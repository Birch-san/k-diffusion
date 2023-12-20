import accelerate
import argparse
from pathlib import Path
import torch
from torch import distributed as dist, multiprocessing as mp, FloatTensor, ByteTensor, Tensor
from torch.utils import data
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms
from torchvision.transforms import _functional_pil as F_pil
from typing import Optional, Union, List, Iterator, Callable
from tqdm import tqdm
from os import makedirs
import gc
from welford_torch import Welford
import math
import numpy as np
from numpy.typing import NDArray
from PIL import Image

import k_diffusion as K
from kdiff_trainer.dataset.get_dataset import get_dataset

def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())

_mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
def _to_pil_style_tensor(pic: Image.Image) -> ByteTensor:
    """
    Converts PIL image to tensor with the fewest changes possible (no rescaling, no normalization, no permute, no casting).
    The reasoning is that rescaling and normalization would be faster on-GPU.
    And this particular script (compute_channel_dist.py) can compute averages fine with channels-last data, so no permute is needed.
    To be run in the DataLoader's CPU worker.
    Returns:
        ByteTensor range: [0, 255], shape: [h, w, c]
    """
    assert pic.mode != "1", "for performance, we do not supoprt 1-bit images; we cannot performantly rescale [0, 1] to [0, 255] inside the CPU worker, and we don't have an easy way to tell the GPU worker that rescaling is required."
    arr: NDArray = np.array(pic, _mode_to_nptype.get(pic.mode, np.uint8), copy=False)
    t: ByteTensor = torch.from_numpy(arr)
    # torchvision ToTensor() use .view() here but to me it looked like a tautology. were they trying to remove batch dims?
    assert t.shape == (pic.size[1], pic.size[0], F_pil.get_image_num_channels(pic))
    return t

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str, required=True,
                   help='configuration file detailing a dataset of ground-truth examples, for example configs/dataset/imagenet.tanishq.jsonc')
    p.add_argument('--batch-size', type=int, default=4,
                   help='the batch size')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--side-length', type=int, default=None,
                   help='square side length to which to resize-and-crop image')
    p.add_argument('--seed', type=int,
                   help='the random seed')
    p.add_argument('--log-average-every-n', type=int, default=1000,
                   help='how noisy do you want your logs to be (log the online average per-channel mean and std of latents every n batches)')
    p.add_argument('--save-average-every-n', type=int, default=10000,
                   help="how frequently to save the welford averages. the main reason we do it on an interval is just so there's no nasty surprise at the end of the run.")
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--out-dir', type=str, default="./out/avg",
                   help='[in inference-only mode] directory into which to output WDS .tar files')

    args = p.parse_args()
    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True

    accelerator = accelerate.Accelerator()
    ensure_distributed()
    
    config = K.config.load_config(args.config, use_json5=args.config.endswith('.jsonc'))
    dataset_config = config['dataset']

    resize_crop: List[Callable] = [] if args.side_length is None else [
        transforms.Resize(args.side_length, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(args.side_length)
    ]

    tf = transforms.Compose(
        [
            *resize_crop,
            # transforms.ToTensor(),
            _to_pil_style_tensor,
        ]
    )

    # for supported dataset types (i.e. non-class-conditional)
    output_tuples: bool = dataset_config['type'] not in ('wds', 'npz', 'imagefolder')
    train_set: Union[Dataset, IterableDataset] = get_dataset(
        dataset_config,
        config_dir=Path(args.config).parent,
        uses_crossattn=False,
        tf=tf,
        class_captions=None,
        # try to prevent memory leak described in
        # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
        # by returning Tensor instead of Tuple[Tensor]
        output_tuples=output_tuples,
    )
    try:
        dataset_len_estimate: int = len(train_set)
    except TypeError:
        # WDS datasets are IterableDataset, so do not implement __len__()
        if 'estimated_samples' in dataset_config:
            dataset_len_estimate: int = dataset_config['estimated_samples']
        else:
            raise ValueError("we need to know how the dataset is, in order to avoid the bias introduced by DataLoader's wraparound (it tries to ensure consistent batch size by drawing samples from a new epoch)")
    batches_estimate: int = math.ceil(dataset_len_estimate/(args.batch_size*accelerator.num_processes))

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=False, drop_last=False,
                               num_workers=args.num_workers, persistent_workers=args.num_workers>0, pin_memory=True)
    train_dl = accelerator.prepare(train_dl)

    if accelerator.is_main_process:
        makedirs(args.out_dir, exist_ok=True)
        w_val = Welford(device=accelerator.device)
        w_sq = Welford(device=accelerator.device)
    else:
        w_val: Optional[Welford] = None
        w_sq: Optional[Welford] = None

    samples_analysed = 0
    if output_tuples:
        image_key = dataset_config.get('image_key', 0)
        it: Iterator[List[Tensor]] = iter(train_dl)
    else:
        it: Iterator[ByteTensor] = iter(train_dl)
    for batch_ix, batch in enumerate(tqdm(it, total=batches_estimate)):
        if output_tuples:
            images: ByteTensor = batch[image_key]
        else:
            images: ByteTensor = batch
        del batch
        # dataset types such as 'imagefolder' will be lists, 'huggingface' will be dicts
        assert torch.is_tensor(images)
        images = images.float()
        # dataloader gives us [0, 255]. we normalize to [-1, 1]
        images.div_(127.5)
        images.sub_(1)
        samples_analysed += images.shape[0]*accelerator.num_processes

        # if we'd converted PIL->Tensor via transforms.ToTensor(), the height and width dimensions would've been -1, -2:
        # per_channel_val_mean: FloatTensor = images.mean((-1, -2))
        # per_channel_sq_mean: FloatTensor = images.square().mean((-1, -2))
        per_channel_val_mean: FloatTensor = images.mean((-2, -3))
        per_channel_sq_mean: FloatTensor = images.square().mean((-2, -3))
        per_channel_val_mean = accelerator.gather(per_channel_val_mean)
        per_channel_sq_mean = accelerator.gather(per_channel_sq_mean)
        if accelerator.is_main_process:
            w_val.add_all(per_channel_val_mean)
            w_sq.add_all(per_channel_sq_mean)

            if batch_ix % args.log_average_every_n == 0:
                print('per-channel val:', w_val.mean.tolist())
                print('per-channel  sq:', w_sq.mean.tolist())
                print('per-channel std:', torch.sqrt(w_sq.mean - w_val.mean**2).tolist())
            if batch_ix % args.save_average_every_n == 0:
                print(f'Saving averages to {args.out_dir}')
                torch.save(w_val.mean, f'{args.out_dir}/val.pt')
                torch.save(w_sq.mean,  f'{args.out_dir}/sq.pt')
        del images, per_channel_val_mean, per_channel_sq_mean
        gc.collect()
    print(f"r{accelerator.process_index} done.")
    if accelerator.is_main_process:
        print(f'Measured {samples_analysed} samples. We wanted {dataset_len_estimate}.')
        print("Over-run *is* possible (if batch_size*num_processes doesn't divide into dataset length, or [as I think can happen with wds] batch size can vary from batch-to-batch)")
        print('per-channel val:', w_val.mean.tolist())
        print('per-channel  sq:', w_sq.mean.tolist())
        print('per-channel std:', torch.sqrt(w_sq.mean - w_val.mean**2).tolist())
        print(f'Saving averages to {args.out_dir}')
        torch.save(w_val.mean, f'{args.out_dir}/val.pt')
        torch.save(w_sq.mean,  f'{args.out_dir}/sq.pt')

if __name__ == '__main__':
    main()