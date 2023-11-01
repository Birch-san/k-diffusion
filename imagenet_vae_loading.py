import accelerate
from accelerate.state import AcceleratorState
import argparse
from pathlib import Path
import torch
from torch import distributed as dist, multiprocessing as mp, inference_mode, FloatTensor
from torch.utils import data
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms
from typing import Optional, Callable, Union, TypedDict, Dict, Any, List, Iterator
from diffusers import AutoencoderKL
from diffusers.models.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.vae import DiagonalGaussianDistribution
from tqdm import tqdm
from os import makedirs
from contextlib import nullcontext
from welford_torch import Welford
import math

import k_diffusion as K
from kdiff_trainer.dataset.get_dataset import get_dataset

SinkOutput = TypedDict('SinkOutput', {
  '__key__': str,
  'img.png': FloatTensor,
  'txt': bytes,
})

def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())

def collate_fn(data):

    images = []
    paths = []
    for entry in data:

        image = entry["image"]
        if type(image) == int:
            if image == -1:
                continue
        images.append(image)
        paths.append(entry["path"])

    images = preprocess_train(images)
    images = torch.stack(images)
    return {"images": images, "paths": paths}

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str, required=True,
                   help='configuration file detailing a dataset of ground-truth examples, for example configs/dataset/imagenet.tanishq.jsonc')
    p.add_argument('--batch-size', type=int, default=4,
                   help='the batch size')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--side-length', type=int, default=256,
                   help='square side length to which to resize-and-crop image')
    p.add_argument('--seed', type=int,
                   help='the random seed')
    p.add_argument('--use-ollin-vae', action='store_true',
                   help="use Ollin's fp16 finetune of SDXL 0.9 VAE")
    p.add_argument('--compile', action='store_true',
                   help="accelerate VAE with torch.compile()")
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--out-root', type=str, default="./shards",
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

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])
    latent_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())

    tf = transforms.Compose(
        [
            transforms.Resize(args.side_length, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(args.side_length),
            transforms.ToTensor(),
        ]
    )

    train_set: Union[Dataset, IterableDataset] = get_dataset(
        dataset_config,
        config_dir=Path(args.config).parent,
        uses_crossattn=False,
        tf=tf,
        class_captions=None,
    )
    # this is just to try and get a good progress bar.
    try:
        dataset_len_estimate: int = len(train_set)
    except TypeError:
        # WDS datasets are IterableDataset, so do not implement __len__()
        if 'estimated_samples' in dataset_config:
            dataset_len_estimate: int = dataset_config['estimated_samples']
        else:
            dataset_len_estimate: Optional[int] = None
    if dataset_len_estimate is None:
        batches_estimate: Optional[int] = None
    else:
        batches_estimate: int = math.ceil(dataset_len_estimate/args.batch_size)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=not isinstance(train_set, data.IterableDataset), drop_last=False,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    
    vae_kwargs: Dict[str, Any] = {
        'torch_dtype': torch.float16,
    } if args.use_ollin_vae else {
        'subfolder': 'vae',
        'torch_dtype': torch.bfloat16,
    }
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix' if args.use_ollin_vae else 'stabilityai/stable-diffusion-xl-base-0.9',
        use_safetensors=True,
        **vae_kwargs,
    )
    vae.to(accelerator.device).eval()
    if args.compile:
        vae = torch.compile(vae, fullgraph=True, mode='max-autotune')

    train_dl = accelerator.prepare(train_dl)

    if accelerator.is_main_process:
        from webdataset import ShardWriter
        makedirs(args.out_root, exist_ok=True)
        print(f'process {accelerator.process_index}')
        print(f'num AcceleratorState().num_processes')
        sink_ctx = ShardWriter(f'{args.out_root}/%05d.tar', maxcount=10000)
        def sink_sample(sink: ShardWriter, ix: int, image, key) -> None:
            out: SinkOutput = {
                '__key__': f'{AcceleratorState().process_index}/{ix}',
                'img.pth': image,
                'txt': key
            }
            sink.write(out)
    else:
        sink_ctx = nullcontext()
        sink_sample: Callable[[Optional[ShardWriter], int, object], None] = lambda *_: ...

    it: Iterator[List[FloatTensor]] = iter(train_dl)
    with sink_ctx as sink:
        for batch in tqdm(it, total=batches_estimate):
            images, *_ = batch
            images = images.to(vae.dtype)
            # transform pipeline's ToTensor() gave us [0, 1]
            # but VAE wants [-1, 1]
            images.mul_(2).sub_(1)
            with inference_mode():
                encoded: AutoencoderKLOutput = vae.encode(images)
                dist: DiagonalGaussianDistribution = encoded.latent_dist
                latents: FloatTensor = dist.sample(generator=latent_gen)
                # you can verify correctness by saving the sample out like so:
                #   from torchvision.utils import save_image
                #   save_image(vae.decode(latents).sample.div(2).add_(.5).clamp_(0,1), 'test.png')
                # let's not multiply by scale factor, and opt instead to measure a per-channel scale-and-shift
                #   latents.mul_(vae.config.scaling_factor)

            #We set image_key and class_cond_key in the CONFIG
            #What we are storing is a random key, and the sample is a dict with two items
            #That's image_key, image and class_cond_key, label. For us, the label needs to be extracted from the path
            for ix, sample in enumerate(latents):
                class_key = batch["paths"][ix].split("/")[-2] #This should be the name of the folder, which is just a random ID for us
                sink_sample(sink_ctx, count, sample, class_key)
                count += 1
    print(f"r{accelerator.process_index} done")

if __name__ == '__main__':
    main()