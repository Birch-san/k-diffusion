import accelerate
import argparse
from pathlib import Path
import torch
from torch import distributed as dist, multiprocessing as mp, FloatTensor, Tensor
from torch.utils import data
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms
from typing import Optional, Union, Dict, List, Iterator, Callable
from tqdm import tqdm
from os import makedirs
import gc
from welford_torch import Welford
import math

import k_diffusion as K
from kdiff_trainer.dataset.get_dataset import get_dataset

def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())

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
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

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
            transforms.ToTensor(),
        ]
    )

    train_set: Union[Dataset, IterableDataset] = get_dataset(
        dataset_config,
        config_dir=Path(args.config).parent,
        uses_crossattn=False,
        tf=tf,
        class_captions=None,
        # try to prevent memory leak described in
        # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
        # by returning Tensor instead of Tuple[Tensor]
        output_tuples=False,
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
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    train_dl = accelerator.prepare(train_dl)

    if accelerator.is_main_process:
        makedirs(args.out_dir, exist_ok=True)
        w_val = Welford(device=accelerator.device)
        w_sq = Welford(device=accelerator.device)
    else:
        w_val: Optional[Welford] = None
        w_sq: Optional[Welford] = None

    samples_output = 0
    it: Iterator[Union[List[Tensor], Dict[str, Tensor]]] = iter(train_dl)
    for batch_ix, batch in enumerate(tqdm(it, total=batches_estimate)):
        # dataset types such as 'imagefolder' will be lists, 'huggingface' will be dicts
        assert torch.is_tensor(batch)

        per_channel_val_mean: FloatTensor = images.mean((-1, -2))
        per_channel_sq_mean: FloatTensor = images.square().mean((-1, -2))
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
            del per_channel_val_mean, per_channel_sq_mean
        del per_channel_val_mean, per_channel_sq_mean
        gc.collect()
    print(f"r{accelerator.process_index} done")
    if accelerator.is_main_process:
        print(f'Output {samples_output} samples. We wanted {dataset_len_estimate}.')
        print('per-channel val:', w_val.mean.tolist())
        print('per-channel  sq:', w_sq.mean.tolist())
        print('per-channel std:', torch.sqrt(w_sq.mean - w_val.mean**2).tolist())
        print(f'Saving averages to {args.out_dir}')
        torch.save(w_val.mean, f'{args.out_dir}/val.pt')
        torch.save(w_sq.mean,  f'{args.out_dir}/sq.pt')

if __name__ == '__main__':
    main()