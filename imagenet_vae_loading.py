import accelerate
import argparse
from pathlib import Path
import torch
from torch import distributed as dist, multiprocessing as mp, inference_mode, FloatTensor, Tensor
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
from kdiff_trainer.vae.attn.null_attn_processor import NullAttnProcessor
from kdiff_trainer.vae.attn.natten_attn_processor import NattenAttnProcessor
from kdiff_trainer.vae.attn.qkv_fusion import fuse_vae_qkv

SinkOutput = TypedDict('SinkOutput', {
  '__key__': str,
  'latent.pth': FloatTensor,
  'cls': bytes,
})

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
    p.add_argument('--side-length', type=int, default=256,
                   help='square side length to which to resize-and-crop image')
    p.add_argument('--seed', type=int,
                   help='the random seed')
    p.add_argument('--log-average-every-n', type=int, default=1000,
                   help='how noisy do you want your logs to be (log the online average per-channel mean and std of latents every n batches)')
    p.add_argument('--save-average-every-n', type=int, default=10000,
                   help="how frequently to save the welford averages. the main reason we do it on an interval is just so there's no nasty surprise at the end of the run.")
    p.add_argument('--use-ollin-vae', action='store_true',
                   help="use Ollin's fp16 finetune of SDXL 0.9 VAE")
    p.add_argument('--compile', action='store_true',
                   help="accelerate VAE with torch.compile()")
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--vae-attn-impl', type=str, default='original',
                   choices=['original', 'natten', 'null'],
                   help='use more lightweight attention in VAE (https://github.com/Birch-san/sdxl-play/pull/3)')
    p.add_argument('--out-root', type=str, default="./out/latents",
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

    if args.vae_attn_impl == 'natten':
        fuse_vae_qkv(vae)
        # NATTEN seems to output identical output to global self-attention at kernel size 17
        # even kernel size 3 looks good (not identical, but very close).
        # I haven't checked what's the smallest kernel size that can look identical. 15 looked good too.
        # seems to speed up encoding of 1024x1024px images by 11%
        vae.set_attn_processor(NattenAttnProcessor(kernel_size=17))
    elif args.vae_attn_impl == 'null':
        for attn in [*vae.encoder.mid_block.attentions, *vae.decoder.mid_block.attentions]:
            # you won't be needing these
            del attn.to_q, attn.to_k
        # doesn't mix information between tokens via QK similarity. just projects every token by V and O weights.
        # looks alright, but is by no means identical to global self-attn.
        vae.set_attn_processor(NullAttnProcessor())
    elif args.vae_attn_impl == 'original':
        # leave it as global self-attention
        pass
    else:
        raise ValueError(f"Never heard of --vae-attn-impl '{args.vae_attn_impl}'")

    del vae.decoder
    vae.to(accelerator.device).eval()
    if args.compile:
        vae = torch.compile(vae, fullgraph=True, mode='max-autotune')

    train_dl = accelerator.prepare(train_dl)

    wds_out_dir = f'{args.out_root}/wds'
    avg_out_dir = f'{args.out_root}/avg'

    if accelerator.is_main_process:
        from webdataset import ShardWriter
        makedirs(wds_out_dir, exist_ok=True)
        makedirs(avg_out_dir, exist_ok=True)
        sink_ctx = ShardWriter(f'{wds_out_dir}/%05d.tar', maxcount=10000)
        w_mean = Welford(device=accelerator.device)
        w_std = Welford(device=accelerator.device)
        def sink_sample(sink: ShardWriter, ix: int, latent: FloatTensor, cls: int) -> None:
            out: SinkOutput = {
                '__key__': str(ix),
                'latent.pth': latent,
                'cls': cls,
            }
            sink.write(out)
    else:
        sink_ctx = nullcontext()
        sink_sample: Callable[[Optional[ShardWriter], int, FloatTensor, int], None] = lambda *_: ...
        w_mean: Optional[Welford] = None
        w_std: Optional[Welford] = None

    it: Iterator[List[Tensor]] = iter(train_dl)
    with sink_ctx as sink:
        for batch_ix, batch in enumerate(tqdm(it, total=batches_estimate)):
            assert isinstance(batch, list)
            assert len(batch) == 2, "batch item was not the expected length of 2. perhaps class labels are missing. use dataset type imagefolder-class or wds-class, to get class labels."
            images, classes = batch
            images = images.to(vae.dtype)
            # transform pipeline's ToTensor() gave us [0, 1]
            # but VAE wants [-1, 1]
            images.mul_(2).sub_(1)
            with inference_mode():
                encoded: AutoencoderKLOutput = vae.encode(images)
            dist: DiagonalGaussianDistribution = encoded.latent_dist
            latents: FloatTensor = dist.sample(generator=latent_gen)
            all_latents: FloatTensor = accelerator.gather(latents)
            # you can verify correctness by saving the sample out like so:
            #   from torchvision.utils import save_image
            #   save_image(vae.decode(all_latents).sample.div(2).add_(.5).clamp_(0,1), 'test.png')
            # let's not multiply by scale factor. opt instead to measure a per-channel scale-and-shift
            #   all_latents.mul_(vae.config.scaling_factor)

            if accelerator.is_main_process:
                per_channel_means: FloatTensor = all_latents.mean((-1, -2))
                per_channel_stds: FloatTensor = all_latents.std((-1, -2))
                w_mean.add_all(per_channel_means)
                w_std.add_all(per_channel_stds)

                if batch_ix % args.log_average_every_n == 0:
                    print('per-channel mean:', w_mean.mean)
                    print('per-channel  std:', w_std.mean)
                if batch_ix % args.save_average_every_n == 0:
                    print(f'Saving averages to {avg_out_dir}')
                    torch.save(w_mean.mean, f'{avg_out_dir}/mean.pt')
                    torch.save(w_std.mean,  f'{avg_out_dir}/std.pt')
                del per_channel_means, per_channel_stds

            all_classes: FloatTensor = accelerator.gather(classes)
            for sample_ix_in_batch, (latent, cls) in enumerate(zip(all_latents.unbind(), all_classes.unbind())):
                sample_ix_in_corpus: int = batch_ix * args.batch_size + sample_ix_in_batch
                sink_sample(sink, sample_ix_in_corpus, latent, cls.item())
            del all_latents, latents, all_classes, classes
    print(f"r{accelerator.process_index} done")
    if accelerator.is_main_process:
        print(f'Saving averages to {avg_out_dir}')
        torch.save(w_mean.mean, f'{avg_out_dir}/mean.pt')
        torch.save(w_std.mean,  f'{avg_out_dir}/std.pt')

if __name__ == '__main__':
    main()