#!/usr/bin/env python3

"""Trains Karras et al. (2022) diffusion models."""

import argparse
from copy import deepcopy
from functools import partial
import math
import json
from pathlib import Path
import time
from os import makedirs
from os.path import relpath

import accelerate
import safetensors.torch as safetorch
import torch
import torch._dynamo
from torch import distributed as dist
from torch.nn import MSELoss
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch import multiprocessing as mp
from torch import optim, FloatTensor, LongTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils import data
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms, utils
from tqdm.auto import tqdm
from typing import Any, List, Optional, Union, Protocol
from PIL import Image
from dataclasses import dataclass
from typing import Optional, TypedDict, Generator, Callable, Literal, Dict, Any
from contextlib import nullcontext
from itertools import islice
from tqdm import tqdm
import numpy as np
import gc

import k_diffusion as K
from kdiff_trainer.dimensions import Dimensions
from kdiff_trainer.make_default_grid_captioner import make_default_grid_captioner
from kdiff_trainer.make_captioned_grid import GridCaptioner
from kdiff_trainer.xattn.precompute_conds import precompute_conds, PrecomputedConds
from kdiff_trainer.xattn.precomputed_cond_cfg_args import get_precomputed_cond_cfg_args
from kdiff_trainer.xattn.make_cfg_crossattn_model import make_cfg_crossattn_model_fn
from kdiff_trainer.xattn.get_precomputed_conds_by_ix import get_precomputed_conds_by_ix
from kdiff_trainer.xattn.crossattn_cfg_args import CrossAttnCFGArgs
from kdiff_trainer.xattn.crossattn_extra_args import CrossAttnExtraArgs
from kdiff_trainer.to_pil_images import to_pil_images
from kdiff_trainer.tqdm_ctx import tqdm_environ, TqdmOverrides
from kdiff_trainer.iteration.batched import batched
from kdiff_trainer.dataset_meta.get_class_captions import get_class_captions, ClassCaptions
from kdiff_trainer.dataset.get_dataset import get_dataset
from kdiff_trainer.dataset.get_latent_dataset import get_latent_dataset
from kdiff_trainer.normalize import Normalize
from kdiff_trainer.migrations.migrate_model import register_load_hooks, should_discard_optim_state

SinkOutput = TypedDict('SinkOutput', {
    '__key__': str,
    'img.png': Image.Image,
})

ClassCondSinkOutput = TypedDict('ClassCondSinkOutput', {
    '__key__': str,
    'img.png': Image.Image,
    'cls.txt': str,
})

@dataclass
class Samples:
    x_0: FloatTensor

@dataclass
class ClassConditionalSamples(Samples):
    class_cond: LongTensor

@dataclass
class Sample:
    pil: Image.Image

@dataclass
class ClassConditionalSample(Sample):
    class_cond: int

class Sampler(Protocol):
    @staticmethod
    def __call__(model_fn: Callable, x: FloatTensor, sigmas: FloatTensor, extra_args: Dict[str, Any]) -> Any: ...

def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size')
    p.add_argument('--cfg-scale', type=float, default=1.,
                   help='CFG scale used for demo samples and FID evaluation')
    p.add_argument('--checkpointing', action='store_true',
                   help='enable gradient checkpointing')
    p.add_argument('--clip-model', type=str, default='ViT-B/16',
                   choices=K.evaluation.CLIPFeatureExtractor.available_models(),
                   help='the CLIP model to use to evaluate')
    p.add_argument('--compile', action='store_true',
                   help='compile the model')
    p.add_argument('--config', type=str, required=True,
                   help='the configuration file')
    p.add_argument('--torchmetrics-fid', action='store_true',
                   help='whether to use torchmetrics FID (in addition to CleanFID)')
    p.add_argument('--out-root', type=str, default='.',
                   help='outputs (checkpoints, demo samples, state, metrics) will be saved under this directory')
    p.add_argument('--output-to-subdir', action='store_true',
                   help='for outputs (checkpoints, demo samples, state, metrics): whether to use {{out_root}}/{{name}}/ subdirectories. When True: saves/loads from/to {{out_root}}/{{name}}/[{{product}}/]*, ({{product}}/ subdirectories are used for demo samples and checkpoints). When False: saves/loads from/to {{out_root}}/{{name}}_*')
    p.add_argument('--demo-every', type=int, default=500,
                   help='save a demo grid every this many steps')
    p.add_argument('--demo-classcond-include-uncond', action='store_true',
                   help='when producing demo grids for class-conditional tasks: allow the generation of uncond demo samples (class chosen from num_classes+1 instead of merely num_classes)')
    p.add_argument('--goose-mode', action='store_true',
                   help='very important option for scientists at EleutherAI')
    p.add_argument('--dinov2-model', type=str, default='vitl14',
                   choices=K.evaluation.DINOv2FeatureExtractor.available_models(),
                   help='the DINOv2 model to use to evaluate')
    p.add_argument('--end-step', type=int, default=None,
                   help='the step to end training at')
    p.add_argument('--evaluate-every', type=int, default=10000,
                   help='evaluate every this many steps')
    p.add_argument('--evaluate-n', type=int, default=2000,
                   help='the number of samples to draw to evaluate')
    p.add_argument('--evaluate-only', action='store_true',
                   help='evaluate instead of training')
    p.add_argument('--evaluate-with', type=str, default='inception',
                   choices=['inception', 'clip', 'dinov2'],
                   help='the feature extractor to use for evaluation')
    p.add_argument('--gns', action='store_true',
                   help='measure the gradient noise scale (DDP only, disables stratified sampling)')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                   help='the number of gradient accumulation steps')
    p.add_argument('--inference-only', action='store_true',
                   help='run demo sample instead of training')
    p.add_argument('--inference-n', type=int, default=None,
                   help='[in inference-only mode] the number of samples to generate in total (in batches of up to --sample-n)')
    p.add_argument('--inference-out-wds-root', type=str, default=None,
                   help='[in inference-only mode] directory into which to output WDS .tar files')
    p.add_argument('--inference-out-wds-shard', type=int, default=None,
                   help="[in inference-only mode] the directory within the WDS dataset .tar to place each sample. this enables you to prevent key clashes if you were to tell multiple nodes to independently produce .tars and collect them together into a single dataset afterward (poor man's version of multi-node support).")
    p.add_argument('--lr', type=float,
                   help='the learning rate')
    p.add_argument('--mixed-precision', type=str,
                   help='the mixed precision type')
    p.add_argument('--name', type=str, default='model',
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--reset-ema', action='store_true',
                   help='reset the EMA')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--resume-inference', type=str,
                   help='the inference checkpoint to resume from')
    p.add_argument('--demo-steps', type=int, default=50,
                   help='the number of steps to sample for demo grids')
    p.add_argument('--sample-n', type=int, default=64,
                   help='the number of images to sample for demo grids')
    p.add_argument('--sampler-preset', type=str, default='dpm3', choices=['dpm2', 'dpm3', 'ddpm'],
                   help='whether to use the original DPM++(2M) SDE, sampler_type="heun" eta=0. config or to use DPM++(3M) SDE eta=1., which seems to get lower FID')
    p.add_argument('--save-every', type=int, default=10000,
                   help='save every this many steps')
    p.add_argument('--seed', type=int,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--text-model-hf-cache-dir', type=str, default=None,
                   help='disk directory into which HF should download text model checkpoints')
    p.add_argument('--text-model-trust-remote-code', action='store_true',
                   help="whether to access model code via HF's Code on Hub feature (required for text encoders such as Phi)")
    p.add_argument('--font', type=str, default='./kdiff_trainer/font/DejaVuSansMono.ttf',
                   help='font used for drawing demo grids (e.g. /usr/share/fonts/dejavu/DejaVuSansMono.ttf or ./kdiff_trainer/font/DejaVuSansMono.ttf). Pass empty string for ImageFont.load_default().')
    p.add_argument('--demo-title-qualifier', type=str, default=None,
                   help='Additional text to include in title printed in demo grids')
    p.add_argument('--demo-img-compress', action='store_true',
                   help='Demo image file format. False: .png; True: .jpg')
    p.add_argument('--wandb-entity', type=str,
                   help='the wandb entity name')
    p.add_argument('--wandb-group', type=str,
                   help='the wandb group name')
    p.add_argument('--wandb-project', type=str,
                   help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--wandb-run-name', type=str, default=None,
                   help='the wandb run name')
    p.add_argument('--wandb-save-model', action='store_true',
                   help='save model to wandb')
    p.add_argument('--enable-vae-slicing', action='store_true',
                   help='limit VAE decode (demo, eval) to batch-of-1 to save memory')
    p.add_argument('--input-size-override', type=int, default=None,
                   help="specify a different input size, overriding what's specified in the model's config. intended only as a convenient way of measuring how the model's FLOPs scale with input size.")
    p.add_argument('--grow-levels-for-input-size', action='store_true',
                   help="[when --input-size-override is specified] for each power-of-2 by which the 'override' input size exceeds the config-specified input size, grow the model by duplicating its shallowest level. intended only as a convenient way of measuring how the model's FLOPs *and parameter count* scale with input size, under a 'duplicate shallowest level' scheme for growing the model. for image_transformer_v2 models only.")
    p.add_argument('--use-x0-loss', action='store_true',
                   help="compute loss in x0-space, i.e. mse(denoised, reals) rather than via the EDM formulation mse(model_output, effective_target). this feature is intended only for verifying whether EDM loss weightings give the same results as corresponding x0 loss weightings.")
    p.add_argument('--log-dataset-distribution', action='store_true',
                   help="Measure/log how close the dataset is to a standard Gaussian. this feature is intended for verifying that you have configured a sane scale-and-shift, or for comparing latent scale-and-shifts against the official SD/SDXL VAE scale factor")
    args = p.parse_args()

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass

    do_train: bool = not args.evaluate_only and not args.inference_only
    if args.inference_only and args.evaluate_only:
        raise ValueError('Cannot fulfil both --inference-only and --evaluate-only; they are mutually-exclusive')

    # use json5 parser if we wish to load .jsonc (commented) config
    config = K.config.load_config(args.config, use_json5=args.config.endswith('.jsonc'))
    model_config = config['model']

    if args.input_size_override is not None:
        assert model_config['type'] == 'image_transformer_v2'
        assert model_config['input_size'][0] == model_config['input_size'][1], "implemented for squares only"
        orig_input_size: int = model_config['input_size'][0]
        model_config['input_size'][0] = args.input_size_override
        model_config['input_size'][1] = args.input_size_override
        if args.grow_levels_for_input_size:
            shallowest_width: int = model_config['widths'][0]
            shallowest_depth: int = model_config['depths'][0]
            shallowest_ff: int = model_config['d_ffs'][0]
            shallowest_self_attn = model_config['self_attns'][0]
            # duplicate shallowest level of the model for every power-of-2 larger the overriden input_size is compared to the originally-specified config
            extra_levels = int(math.log2(args.input_size_override)-math.log2(orig_input_size))
            for _ in range(extra_levels):
                model_config['widths'].insert(0, shallowest_width)
                model_config['depths'].insert(0, shallowest_depth)
                model_config['d_ffs'].insert(0, shallowest_ff)
                model_config['self_attns'].insert(0, deepcopy(shallowest_self_attn))
                model_config['dropout_rate'].insert(0, 0.0)

    dataset_config = config['dataset']
    if do_train:
        opt_config = config['optimizer']
        sched_config = config['lr_sched']
        ema_sched_config = config['ema_sched']
    else:
        opt_config = sched_config = ema_sched_config = None

    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.grad_accum_steps, mixed_precision=args.mixed_precision)
    ensure_distributed()
    device = accelerator.device
    unwrap = accelerator.unwrap_model
    print(f'Process {accelerator.process_index} using device: {device}', flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f'World size: {accelerator.num_processes}', flush=True)
        print(f'Batch size: {args.batch_size * accelerator.num_processes}', flush=True)

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])
    demo_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())
    elapsed = 0.0

    class_captions: Optional[ClassCaptions] = get_class_captions(dataset_config['classes_to_captions']) if 'classes_to_captions' in dataset_config else None
    
    uses_crossattn: bool = 'cross_attn' in model_config and model_config['cross_attn']
    if uses_crossattn:
        assert 'demo_uncond' in dataset_config
        assert dataset_config['demo_uncond'] == 'allzeros' or dataset_config['demo_uncond'] == 'emptystr'
        assert class_captions is not None, "cross-attention is currently only implemented for precomputed conditions based on class labels, but your dataset config does not tell us what convention to use to produce captions from your class indices. set your config's dataset.classes_to_captions to a dataset name for which we have a mapping"

    if uses_crossattn and class_captions is not None:
        precomputed_conds: PrecomputedConds = precompute_conds(
            accelerator=accelerator,
            class_captions=class_captions.embed_class_captions,
            uncond_class_ix=class_captions.uncond_class_ix,
            encoder=model_config['cross_attn']['encoder'],
            trust_remote_code=args.text_model_trust_remote_code,
            hf_cache_dir=args.text_model_hf_cache_dir,
        )
        masked_uncond: FloatTensor = precomputed_conds.allzeros_masked_uncond if dataset_config['demo_uncond'] == 'allzeros' else precomputed_conds.emptystr_masked_uncond
    else:
        precomputed_conds: Optional[PrecomputedConds] = None
        masked_uncond: Optional[FloatTensor] = None

    if model_config['type'] == 'guided_diffusion':
        from kdiff_trainer.load_diffusion_model import construct_diffusion_model
        # can't easily put this into K.config.make_model; would change return type and introduce dependency
        model_, guided_diff = construct_diffusion_model(model_config['config'])
    else:
        model_ = K.config.make_model(config)
        guided_diff = None

    if do_train:
        inner_model, inner_model_ema = model_, deepcopy(model_)
    else:
        inner_model, inner_model_ema = None, model_
    del model_

    if args.compile:
        (inner_model or inner_model_ema).compile()
        # inner_model_ema.compile()

    if accelerator.is_main_process:
        print(f'Parameters: {K.utils.n_params((inner_model or inner_model_ema)):,} for {args.config}')

    # If logging to wandb, initialize the run
    use_wandb = accelerator.is_main_process and args.wandb_project
    if use_wandb:
        import wandb
        log_config = vars(deepcopy(args))
        log_config['config'] = config
        log_config['parameters'] = K.utils.n_params((inner_model or inner_model_ema))
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group, config=log_config, name=args.wandb_run_name, save_code=True)

    if do_train:
        lr = opt_config['lr'] if args.lr is None else args.lr
        groups = inner_model.param_groups(lr)
        # for FSDP support: models must be prepared separately and before optimizers
        inner_model, inner_model_ema = accelerator.prepare(inner_model, inner_model_ema)
        if opt_config['type'] == 'adamw':
            opt = optim.AdamW(groups,
                            lr=lr,
                            betas=tuple(opt_config['betas']),
                            eps=opt_config['eps'],
                            weight_decay=opt_config['weight_decay'])
        elif opt_config['type'] == 'adam8bit':
            import bitsandbytes as bnb
            opt = bnb.optim.Adam8bit(groups,
                                    lr=lr,
                                    betas=tuple(opt_config['betas']),
                                    eps=opt_config['eps'],
                                    weight_decay=opt_config['weight_decay'])
        elif opt_config['type'] == 'sgd':
            opt = optim.SGD(groups,
                            lr=lr,
                            momentum=opt_config.get('momentum', 0.),
                            nesterov=opt_config.get('nesterov', False),
                            weight_decay=opt_config.get('weight_decay', 0.))
        else:
            raise ValueError('Invalid optimizer type')

        if sched_config['type'] == 'inverse':
            sched = K.utils.InverseLR(opt,
                                    inv_gamma=sched_config['inv_gamma'],
                                    power=sched_config['power'],
                                    warmup=sched_config['warmup'])
        elif sched_config['type'] == 'exponential':
            sched = K.utils.ExponentialLR(opt,
                                        num_steps=sched_config['num_steps'],
                                        decay=sched_config['decay'],
                                        warmup=sched_config['warmup'])
        elif sched_config['type'] == 'constant':
            sched = K.utils.ConstantLRWithWarmup(opt, warmup=sched_config['warmup'])
        else:
            raise ValueError('Invalid schedule type')

        assert ema_sched_config['type'] == 'inverse'
        ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                    max_value=ema_sched_config['max_value'])
        ema_stats = {}
    else:
        opt: Optional[Optimizer] = None
        sched: Optional[LRScheduler] = None
        ema_sched: Optional[K.utils.EMAWarmup] = None
        # for FSDP support: model must be prepared separately and before optimizers
        inner_model_ema = accelerator.prepare(inner_model_ema)

    if 'channel_means' in dataset_config or 'channel_squares' in dataset_config or 'channel_stds' in dataset_config:
        assert 'channel_means' in dataset_config, 'dataset channel normalization attributes are incomplete; channel_means is missing'
        channel_means: FloatTensor = torch.tensor(dataset_config['channel_means'])
        if 'channel_stds' in dataset_config:
            channel_stds: FloatTensor = torch.tensor(dataset_config['channel_stds'])
        elif 'channel_squares' in dataset_config:
            channel_squares: FloatTensor = torch.tensor(dataset_config['channel_squares'])
            channel_stds: FloatTensor = torch.sqrt(channel_squares - channel_means**2)
            del channel_squares
        else:
            raise ValueError('dataset channel normalization attributes are incomplete; neither channel_stds nor channel_squares was found.')
        normalizer = Normalize(channel_means, channel_stds)
        del channel_means, channel_stds
        accelerator.prepare(normalizer)
    else:
        normalizer = None

    is_latent: bool = dataset_config.get('latents', False)
    if is_latent:
        assert normalizer is not None, "we assume for now that latents require scale-and-shift. remove this assertion if you've taken countermeasures such as increasing sigma_data to compensate"
        # we don't do resize & center-crop, because our latent datasets are precomputed
        # (via imagenet_vae_loading.py) for a given canvas size
        train_set: Union[Dataset, IterableDataset] = get_latent_dataset(dataset_config)

        # we're just using this to decode demo latents
        from diffusers import AutoencoderKL
        from diffusers.models.autoencoder_kl import DecoderOutput
        vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-0.9',
            use_safetensors=True,
            subfolder='vae',
            torch_dtype=torch.bfloat16,
        )
        if args.enable_vae_slicing:
            # serialize VAE decode jobs into batch-of-1
            vae.enable_slicing()
        # 8x upsample
        maybe_vae_upsample: int = 1 << (len(vae.config.block_out_channels) - 1)
        del vae.encoder
        vae.eval()
        accelerator.prepare(vae)
    else:
        tf = transforms.Compose([
            transforms.Resize(size[0], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size[0]),
            K.augmentation.KarrasAugmentationPipeline(model_config['augment_prob'], disable_all=model_config['augment_prob'] == 0),
        ])
        train_set: Union[Dataset, IterableDataset] = get_dataset(
            dataset_config,
            config_dir=Path(args.config).parent,
            uses_crossattn=uses_crossattn,
            tf=tf,
            class_captions=class_captions,
        )
        maybe_vae_upsample = 1

    if accelerator.is_main_process:
        try:
            print(f'Number of items in dataset: {len(train_set):,}')
        except TypeError:
            pass

    image_key = dataset_config.get('image_key', 0)
    num_classes = dataset_config.get('num_classes', 0)
    cond_dropout_rate = dataset_config.get('cond_dropout_rate', 0.1)
    class_key = dataset_config.get('class_key', 1)

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=not isinstance(train_set, data.IterableDataset), drop_last=True,
                               num_workers=args.num_workers, persistent_workers=args.num_workers>0, pin_memory=True)

    if do_train:
        opt, train_dl = accelerator.prepare(opt, train_dl)
        if use_wandb:
            wandb.watch(inner_model)
    else:
        train_dl = accelerator.prepare(train_dl)

    if accelerator.num_processes == 1:
        args.gns = False
    if args.gns and do_train:
        gns_stats_hook = K.gns.DDPGradientStatsHook(inner_model)
        gns_stats = K.gns.GradientNoiseScale()
    else:
        gns_stats = None
    if guided_diff is None:
        sigma_min = model_config['sigma_min']
        sigma_max = model_config['sigma_max']
        if do_train:
            sample_density = K.config.make_sample_density(model_config)
            model = K.config.make_denoiser_wrapper(config)(inner_model)
        model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)
        class_cond_key = 'class_cond'
    else:
        from kdiff_trainer.load_diffusion_model import wrap_diffusion_model
        model_ema = wrap_diffusion_model(inner_model_ema, guided_diff, device=accelerator.device)
        sigma_min = model_ema.sigma_min.item()
        sigma_max = model_ema.sigma_max.item()
        if do_train:
            sample_density = partial(K.utils.rand_uniform, min_value=0, max_value=guided_diff.num_timesteps-1)
            model = wrap_diffusion_model(inner_model, guided_diff, device=accelerator.device)
        else:
            model_ema.requires_grad_(False).eval()
        class_cond_key = 'y'
    
    with torch.no_grad(), K.models.flops.flop_counter() as fc:
        x = torch.zeros([1, model_config['input_channels'], size[0], size[1]], device=device)
        sigma = torch.ones([1], device=device)
        extra_args = {}
        if getattr(unwrap((inner_model or inner_model_ema)), "num_classes", 0):
            extra_args[class_cond_key] = torch.zeros([1], dtype=torch.long, device=device)
        (inner_model or inner_model_ema)(x, sigma, **extra_args)
        if accelerator.is_main_process:
            print(f"Forward pass GFLOPs: {fc.flops / 1_000_000_000:,.3f} for {args.config}", flush=True)

    if args.output_to_subdir:
        run_root = f'{args.out_root}/{args.name}'
        state_root = metrics_root = run_root
        demo_root = f'{run_root}/demo'
        ckpt_root = f'{run_root}/ckpt'
        demo_file_qualifier = ''
    else:
        run_root = demo_root = ckpt_root = state_root = metrics_root = args.out_root
        demo_file_qualifier = 'demo_'
    run_qualifier = f'{args.name}_'

    if accelerator.is_main_process:
        makedirs(run_root, exist_ok=True)
        makedirs(state_root, exist_ok=True)
        makedirs(metrics_root, exist_ok=True)
        makedirs(demo_root, exist_ok=True)
        makedirs(ckpt_root, exist_ok=True)

    state_path = Path(f'{state_root}/{run_qualifier}state.json')

    if state_path.exists() or args.resume:
        if args.resume:
            ckpt_path = args.resume
        if not args.resume:
            state = json.load(open(state_path))
            ckpt_path = f"{state_root}/{state['latest_checkpoint']}"
        if accelerator.is_main_process:
            print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt_config = ckpt['config']
        # merge in any new config defaults that have been introduced since the checkpoint was created
        ckpt_config = K.config.load_config(ckpt_config)
        register_load_hooks(unwrap(model_ema.inner_model), ckpt_config['model'], model_config)
        unwrap(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
        if do_train:
            register_load_hooks(unwrap(model.inner_model), ckpt_config['model'], model_config)
            unwrap(model.inner_model).load_state_dict(ckpt['model'])
            if not should_discard_optim_state(unwrap(model.inner_model), ckpt_config['model'], model_config):
                opt.load_state_dict(ckpt['opt'])
            sched.load_state_dict(ckpt['sched'])
            ema_sched.load_state_dict(ckpt['ema_sched'])
            ema_stats = ckpt.get('ema_stats', ema_stats)
            epoch = ckpt['epoch'] + 1
            step = ckpt['step'] + 1
            if args.gns and ckpt.get('gns_stats', None) is not None:
                gns_stats.load_state_dict(ckpt['gns_stats'])
            demo_gen.set_state(ckpt['demo_gen'])
            elapsed = ckpt.get('elapsed', 0.0)

        del ckpt
    else:
        epoch = 0
        step = 0

    if args.reset_ema:
        if not do_train:
            raise ValueError("Training is disabled (this can happen as a result of options such as --evaluate-only). Accordingly we did not construct a trainable model, and consequently cannot load the EMA model's weights onto said trainable model. Disable --reset-ema, or enable training.")
        unwrap(model.inner_model).load_state_dict(unwrap(model_ema.inner_model).state_dict())
        ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                      max_value=ema_sched_config['max_value'])
        ema_stats = {}

    if args.resume_inference:
        if accelerator.is_main_process:
            print(f'Loading {args.resume_inference}...')
        if guided_diff is None:
            ckpt = safetorch.load_file(args.resume_inference)
            if do_train:
                unwrap(model.inner_model).load_state_dict(ckpt)
            unwrap(model_ema.inner_model).load_state_dict(ckpt)
            del ckpt
        else:
            from kdiff_trainer.load_diffusion_model import load_diffusion_model
            if do_train:
                load_diffusion_model(args.resume_inference, unwrap(model.inner_model))
            load_diffusion_model(args.resume_inference, unwrap(model_ema.inner_model))

    evaluate_enabled = do_train and args.evaluate_every > 0 and args.evaluate_n > 0 or args.evaluate_only
    metrics_log = None
    if evaluate_enabled:
        if args.evaluate_with == 'inception':
            extractor = K.evaluation.InceptionV3FeatureExtractor(device=device)
        elif args.evaluate_with == 'clip':
            extractor = K.evaluation.CLIPFeatureExtractor(args.clip_model, device=device)
        elif args.evaluate_with == 'dinov2':
            extractor = K.evaluation.DINOv2FeatureExtractor(args.dinov2_model, device=device)
        else:
            raise ValueError('Invalid evaluation feature extractor')
        train_iter = iter(train_dl)
        if args.torchmetrics_fid:
            if accelerator.is_main_process:
                from torchmetrics.image.fid import FrechetInceptionDistance
                # "normalize" means "my images are [0, 1] floats"
                # https://torchmetrics.readthedocs.io/en/stable/image/frechet_inception_distance.html
                # we tell it not to obliterate our real features on reset(), because we have no mechanism set up to compute reals again
                fid_obj = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False)
                fid_obj.to(accelerator.device)
            def observe_samples(real: bool, samples: FloatTensor) -> None:
                all_samples: FloatTensor = accelerator.gather(samples)
                if accelerator.is_main_process:
                    fid_obj.update(all_samples, real=real)
            observe_samples_real: Callable[[FloatTensor], None] = partial(observe_samples, True)
            observe_samples_fake: Callable[[FloatTensor], None] = partial(observe_samples, False)
        else:
            observe_samples_real: Optional[Callable[[FloatTensor], None]] = None
            observe_samples_fake: Optional[Callable[[FloatTensor], None]] = None
        if accelerator.is_main_process:
            print('Computing features for reals...')
        if is_latent:
            def sample_fn(cur_batch_size: int) -> FloatTensor:
                samples = next(train_iter)
                # we don't augment latents, because even flipped latents are pretty OOD (they look damaged when decoded).
                # translations sort of work, but decoder only resolves convolution padding artifacts when they're at the canvas edge.
                # basically if we want augmentations: we should do it in RGB before encoding (which is awkward if precomputing, or
                # slow if done live).
                latents: FloatTensor = samples[image_key]
                # adapt from standard gaussian onto VAE's distribution
                normalizer.inverse_(latents)
                with torch.inference_mode():
                    decoded: DecoderOutput = vae.decode(latents.to(vae.dtype))
                del latents
                rgb: FloatTensor = decoded.sample
                # cast here to ensure we have same dtype as our fakes (which come out of the float32 ema_model).
                # rgb datasets have float32 reals because they go through KarrasAugmentationPipeline, which returns a float32 tensor.
                return rgb.float()
        else:
            def sample_fn(cur_batch_size: int) -> FloatTensor:
                samples = next(train_iter)
                augmented_imgs = samples[image_key]
                # KarrasAugmentationPipeline maps images to 3-tuples:
                # image, image_orig, aug_cond
                _, image_orig, _ = augmented_imgs
                return image_orig
        reals_features = K.evaluation.compute_features(accelerator, sample_fn, extractor, args.evaluate_n, args.batch_size, observe_samples=observe_samples_real)
        if accelerator.is_main_process and not args.evaluate_only:
            fid_cols: List[str] = ['fid']
            if args.torchmetrics_fid:
                fid_cols.append('tfid')
            metrics_log = K.utils.CSVLogger(f'{metrics_root}/{run_qualifier}metrics.csv', ['step', 'time', 'loss', *fid_cols, 'kid'])
        del train_iter

    maybe_uncond_class: Literal[1, 0] = 1 if args.demo_classcond_include_uncond else 0
    
    if args.sampler_preset == 'dpm3':
        def do_sample(model_fn: Callable, x: FloatTensor, sigmas: FloatTensor, extra_args: Dict[str, Any], disable: bool) -> FloatTensor:
            return K.sampling.sample_dpmpp_3m_sde(model_fn, x, sigmas, extra_args=extra_args, eta=1.0, disable=disable)
    elif args.sampler_preset == 'dpm2':
        def do_sample(model_fn: Callable, x: FloatTensor, sigmas: FloatTensor, extra_args: Dict[str, Any], disable: bool) -> FloatTensor:
            return K.sampling.sample_dpmpp_2m_sde(model_fn, x, sigmas, extra_args=extra_args, eta=0.0, solver_type='heun', disable=disable)
    elif args.sampler_preset == 'ddpm':
        def do_sample(model_fn: Callable, x: FloatTensor, sigmas: FloatTensor, extra_args: Dict[str, Any], disable: bool) -> FloatTensor:
            return K.sampling.sample_euler_ancestral(model_fn, x, sigmas, extra_args=extra_args, eta=1.0, disable=disable)
    else:
        raise ValueError(f"Unsupported sampler_preset: '{args.sampler_preset}'")

    def make_cfg_model_fn(model):
        def cfg_model_fn(x, sigma, class_cond):
            x_in = torch.cat([x, x])
            sigma_in = torch.cat([sigma, sigma])
            class_uncond = torch.full_like(class_cond, num_classes)
            class_cond_in = torch.cat([class_uncond, class_cond])
            out = model(x_in, sigma_in, class_cond=class_cond_in)
            out_uncond, out_cond = out.chunk(2)
            return out_uncond + (out_cond - out_uncond) * args.cfg_scale
        if args.cfg_scale != 1:
            return cfg_model_fn
        return model
    
    def generate_batch_of_samples() -> Samples:
        n_per_proc = math.ceil(args.sample_n / accelerator.num_processes)
        x = torch.randn([accelerator.num_processes, n_per_proc, model_config['input_channels'], size[0], size[1]], generator=demo_gen).to(device)
        dist.broadcast(x, 0)
        x = x[accelerator.process_index] * sigma_max
        model_fn, extra_args = model_ema, {}
        class_cond: Optional[LongTensor] = None
        if uses_crossattn:
            assert precomputed_conds is not None, "cross-attention is currently only implemented for precomputed conditions"
            assert masked_uncond is not None, "expected that we would have assigned to masked_uncond either an all-zeros uncond, or an uncond generated by embedding an empty string"
            cfg_args: CrossAttnCFGArgs = get_precomputed_cond_cfg_args(
                accelerator=accelerator,
                masked_conds=precomputed_conds.masked_conds,
                uncond_class_ix=class_captions.uncond_class_ix,
                use_allzeros_uncond=dataset_config['demo_uncond'] == 'allzeros',
                n_per_proc=n_per_proc,
                distribute=True,
                include_uncond=args.demo_classcond_include_uncond,
                rng=demo_gen,
            )
            class_cond = cfg_args.caption_ix
            extra_args = {**extra_args, **cfg_args.sampling_extra_args}
            model_fn = make_cfg_crossattn_model_fn(model_ema, masked_uncond=masked_uncond, cfg_scale=args.cfg_scale)
        elif num_classes:
            if args.goose_mode:
                class_cond = torch.full([accelerator.num_processes, n_per_proc], fill_value=99, dtype=torch.int64).to(device)
            else:
                class_cond = torch.randint(0, num_classes + maybe_uncond_class, [accelerator.num_processes, n_per_proc], generator=demo_gen).to(device)
            dist.broadcast(class_cond, 0)
            # print ImageNet class labels like so:
            # from kdiff_trainer.dataset_meta.imagenet_1k import class_labels
            # print('\n'.join([' | '.join([class_labels[cell] for cell in row]) for row in class_cond[accelerator.process_index].unflatten(-1, sizes=(4,4)).tolist()]))
            extra_args[class_cond_key] = class_cond[accelerator.process_index]
            model_fn = make_cfg_model_fn(model_ema)
        sigmas = K.sampling.get_sigmas_karras(args.demo_steps, sigma_min, sigma_max, rho=7., device=device)

        x_0: FloatTensor = do_sample(model_fn, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
        # adapt from training distribution (e.g. sigma_data=1.0) onto VAE's distribution (if latent) or onto reals distribution
        if normalizer is not None:
            normalizer.inverse_(x_0)
        x_0 = accelerator.gather(x_0)[:args.sample_n]
        if class_cond is None:
            return Samples(x_0)
        class_cond = class_cond.flatten(end_dim=1)[:args.sample_n]
        return ClassConditionalSamples(x_0, class_cond)
    
    @torch.inference_mode() # note: inference_mode is lower-overhead than no_grad but disables forward-mode AD
    @K.utils.eval_mode(model_ema)
    def generate_samples() -> Generator[Optional[Sample], None, None]:
        if accelerator.is_main_process:
            tqdm.write('Sampling...')
        with FSDP.summon_full_params(model_ema):
            pass

        while True:
            with tqdm_environ(TqdmOverrides(position=1)):
                batch: Samples = generate_batch_of_samples()
            # denormalization/scale-and-shift (such as for VAE) was already done inside generate_batch_of_samples()
            if is_latent:
                # VAE decoder outputs a FloatTensor with range approx ±1
                decoded: DecoderOutput = vae.decode(batch.x_0.to(vae.dtype))
                batch.x_0 = decoded.sample
                del decoded
            if accelerator.is_main_process:
                pils: List[Image.Image] = to_pil_images(batch.x_0)
                if isinstance(batch, ClassConditionalSamples):
                    for pil, class_cond in zip(pils, batch.class_cond.unbind()):
                        yield ClassConditionalSample(pil, class_cond.item())
                else:
                    for pil in pils:
                        yield Sample(pil)
                del pils
            else:
                yield from [None]*batch.x_0.shape[0]
            del batch # turns out not to be necessary, but this way we avoid having two batches in memory at the point of reassigning, maybe
            gc.collect() # very important, which makes me question when the runtime was *planning* on collecting freed references otherwise
            # torch.cuda.empty_cache() # turns out not to be necessary, but keep this in mind if I'm wrong about that

    @torch.inference_mode() # note: inference_mode is lower-overhead than no_grad but disables forward-mode AD
    @K.utils.eval_mode(model_ema)
    def demo(captioner: GridCaptioner):
        if accelerator.is_main_process:
            tqdm.write('Sampling...')
        with FSDP.summon_full_params(model_ema):
            pass

        batch: Samples = generate_batch_of_samples()
        # denormalization/scale-and-shift (such as for VAE) was already done inside generate_batch_of_samples()
        if is_latent:
            # VAE decoder outputs a FloatTensor with range approx ±1
            decoded: DecoderOutput = vae.decode(batch.x_0.to(vae.dtype))
            batch.x_0 = decoded.sample
            del decoded
        if accelerator.is_main_process:
            if class_captions is not None:
                assert isinstance(batch, ClassConditionalSamples)
                pils: List[Image.Image] = to_pil_images(batch.x_0)
                captions: List[str] = [class_captions.demo_class_captions[caption_ix_.item()] for caption_ix_ in batch.class_cond.flatten().cpu()]
                title = f'[step {step}] {args.name} {args.config}'
                if args.demo_title_qualifier:
                    title += f' {args.demo_title_qualifier}'
                grid_pil: Image.Image = captioner.__call__(
                    imgs=pils,
                    captions=captions,
                    title=title,
                )
            else:
                grid = utils.make_grid(batch.x_0, nrow=math.ceil(args.sample_n ** 0.5), padding=0)
                grid_pil: Image.Image = K.utils.to_pil_image(grid)
            save_kwargs = { 'subsampling': 0, 'quality': 95 } if args.demo_img_compress else {}
            fext = 'jpg' if args.demo_img_compress else 'png'
            filename = f'{demo_root}/{run_qualifier}{demo_file_qualifier}{step:08}.{fext}'
            grid_pil.save(filename, **save_kwargs)

            if use_wandb:
                wandb.log({'demo_grid': wandb.Image(filename)}, step=step)

    @torch.inference_mode() # note: inference_mode is lower-overhead than no_grad but disables forward-mode AD
    @K.utils.eval_mode(model_ema)
    def evaluate():
        if not evaluate_enabled:
            return
        if accelerator.is_main_process:
            tqdm.write('Evaluating...')
        with FSDP.summon_full_params(model_ema):
            pass
        sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)
        def sample_fn(n: int) -> FloatTensor:
            x = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
            model_fn, extra_args = model_ema, {}
            if uses_crossattn:
                assert precomputed_conds is not None, "cross-attention is currently only implemented for precomputed conditions"
                assert masked_uncond is not None, "expected that we would have assigned to masked_uncond either an all-zeros uncond, or an uncond generated by embedding an empty string"
                cfg_args: CrossAttnCFGArgs = get_precomputed_cond_cfg_args(
                    accelerator=accelerator,
                    masked_conds=precomputed_conds.masked_conds,
                    uncond_class_ix=class_captions.uncond_class_ix,
                    use_allzeros_uncond=dataset_config['demo_uncond'] == 'allzeros',
                    n_per_proc=n,
                    distribute=False,
                    include_uncond=False,
                    rng=None,
                )
                extra_args = {**extra_args, **cfg_args.sampling_extra_args}
                model_fn = make_cfg_crossattn_model_fn(model_ema, masked_uncond=masked_uncond, cfg_scale=args.cfg_scale)
            elif num_classes:
                extra_args[class_cond_key] = torch.randint(0, num_classes, [n], device=device)
                model_fn = make_cfg_model_fn(model_ema)
            x_0: FloatTensor = do_sample(model_fn, x, sigmas, extra_args=extra_args, disable=True)
            # adapt from training distribution (e.g. sigma_data=1.0) onto VAE's distribution (if latent) or onto reals distribution
            if normalizer is not None:
                normalizer.inverse_(x_0)
            return x_0
        if is_latent:
            # wrap sample_fn
            sample_latent = sample_fn
            def sample_fn(n: int) -> FloatTensor:
                # sample_latent performs denormalization, so these latents have already been moved from training distribution to VAE distribution
                latents: FloatTensor = sample_latent(n)
                with torch.inference_mode():
                    decoded: DecoderOutput = vae.decode(latents.to(vae.dtype))
                return decoded.sample.type_as(latents)
        fakes_features = K.evaluation.compute_features(accelerator, sample_fn, extractor, args.evaluate_n, args.batch_size, observe_samples=observe_samples_fake)
        if accelerator.is_main_process:
            fid = K.evaluation.fid(fakes_features, reals_features)
            kid = K.evaluation.kid(fakes_features, reals_features)
            cfid: float = fid.item()
            fid_csv_vals: List[float] = [cfid]
            fid_wandb_vals: Dict[str, float] = {'FID': cfid}
            fid_summary = f'FID: {cfid:g}'
            if args.torchmetrics_fid:
                tfid: float = fid_obj.compute().item()
                fid_csv_vals.append(tfid)
                fid_wandb_vals['tFID'] = tfid
                fid_summary += f', tFID: {tfid:g}'
                # this will only reset fake features, because we passed construction param reset_real_features=False
                fid_obj.reset()
            print(f'{fid_summary}, KID: {kid.item():g}')
            if metrics_log is not None:
                metrics_log.write(step, elapsed, ema_stats['loss'], *fid_csv_vals, kid.item())
            if use_wandb:
                wandb.log({**fid_wandb_vals, 'KID': kid.item()}, step=step)

    def save():
        accelerator.wait_for_everyone()
        filename = f'{ckpt_root}/{run_qualifier}{step:08}.pth'
        if accelerator.is_main_process:
            tqdm.write(f'Saving to {filename}...')
        with (
            FSDP.summon_full_params(model.inner_model, rank0_only=True, offload_to_cpu=True, writeback=False),
            FSDP.summon_full_params(model_ema.inner_model, rank0_only=True, offload_to_cpu=True, writeback=False),
        ):
            inner_model = unwrap(model.inner_model)
            inner_model_ema = unwrap(model_ema.inner_model)
            obj = {
                'config': config,
                'model': inner_model.state_dict(),
                'model_ema': inner_model_ema.state_dict(),
                'opt': opt.state_dict(),
                'sched': sched.state_dict(),
                'ema_sched': ema_sched.state_dict(),
                'epoch': epoch,
                'step': step,
                'gns_stats': gns_stats.state_dict() if gns_stats is not None else None,
                'ema_stats': ema_stats,
                'demo_gen': demo_gen.get_state(),
                'elapsed': elapsed,
            }
            accelerator.save(obj, filename)
            if accelerator.is_main_process:
                state_obj = {'latest_checkpoint': relpath(filename, state_root)}
                json.dump(state_obj, open(state_path, 'w'))
            if args.wandb_save_model and use_wandb:
                wandb.save(filename)

    # TODO: are h and w the right way around?
    samp_h, samp_w = model_config['input_size']
    sample_size = Dimensions(height=samp_h * maybe_vae_upsample, width=samp_w * maybe_vae_upsample)
    captioner: GridCaptioner = make_default_grid_captioner(
        font_path=args.font,
        sample_n=args.sample_n,
        sample_size=sample_size,
    )
    
    if args.inference_only:
        if args.demo_classcond_include_uncond:
            if accelerator.is_main_process:
                print('WARN: you have enabled uncond samples to be generated (--demo-classcond-include-uncond), and you are in --inference-only mode. if you are using this mode for demo purposes this can be reasonable. but if you are using this mode for purposes of computing FID: you will not want a mixture of cond and uncond samples.')
        if args.sample_n < 1:
            raise ValueError('--inference-only requested but --sample-n is less than 1')
        if (args.inference_n is None) != (args.inference_out_wds_root is None):
            # this mutual-presence requirement could be relaxed if we were to implement other ways of outputting many images (e.g. folder-of-images)
            # but the only current use-case for "inference more samples than we can fit in a batch" is for _making datasets_, which may as well be WDS.
            raise ValueError('--inference-n and --inference-out-wds-root must both be provided if either are provided.')
        if args.inference_n is None:
            demo(captioner)
        else:
            samples = islice(generate_samples(), args.inference_n)
            if accelerator.is_main_process:
                from webdataset import ShardWriter
                makedirs(args.inference_out_wds_root, exist_ok=True)
                sink_ctx = ShardWriter(f'{args.inference_out_wds_root}/%05d.tar', maxcount=10000)
                shard_qualifier: Optional[str] = '' if args.inference_out_wds_shard is None else f'{args.inference_out_wds_shard}/'
                if num_classes:
                    def sink_sample(sink: ShardWriter, ix: int, sample: ClassConditionalSample) -> None:
                        out: ClassCondSinkOutput = {
                            '__key__': f'{shard_qualifier}{ix}',
                            'img.png': sample.pil,
                            'cls.txt': str(sample.class_cond),
                        }
                        sink.write(out)
                else:
                    def sink_sample(sink: ShardWriter, ix: int, sample: Sample) -> None:
                        out: SinkOutput = {
                            '__key__': f'{shard_qualifier}{ix}',
                            'img.png': sample.pil,
                        }
                        sink.write(out)
            else:
                sink_ctx = nullcontext()
                sink_sample: Callable[[Optional[ShardWriter], int, Sample], None] = lambda *_: ...
            # turns out there's a little bit of memory we can get back before we begin
            torch.cuda.empty_cache()
            with sink_ctx as sink:
                for batch_ix, batch in enumerate(tqdm(
                    # collating into batches just to get a more reliable progress report from tqdm
                    batched(samples, args.sample_n),
                    'sampling batches',
                    disable=not accelerator.is_main_process,
                    position=0,
                    total=math.ceil(args.inference_n/args.sample_n),
                    unit='batch',
                )):
                    for ix, sample in enumerate(batch):
                        corpus_ix: int = args.sample_n*batch_ix + ix
                        sink_sample(sink, corpus_ix, sample)
                    if accelerator.is_main_process:
                        tqdm.write(f'batch {batch_ix} {torch.cuda.memory_allocated()/1024**2:.2f}MiB allocated, {torch.cuda.memory_reserved()/1024**2:.2f}MiB reserved of {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f}MiB')
        if accelerator.is_main_process:
            tqdm.write('Finished inferencing!')
        return

    if args.evaluate_only:
        if args.evaluate_n < 1:
            raise ValueError('--evaluate-only requested but --evaluate-n is less than 1')
        evaluate()
        if accelerator.is_main_process:
            tqdm.write('Finished evaluating!')
        return

    losses_since_last_print = []

    if do_train and args.use_x0_loss:
        assert model_config['loss_weighting'] in ('snr', 'soft-min-snr', 'min-snr', 'karras')
        mse_loss_fn = MSELoss(reduction='none')

    try:
        while True:
            for batch in tqdm(train_dl, smoothing=0.1, disable=not accelerator.is_main_process):
                if device.type == 'cuda':
                    start_timer = torch.cuda.Event(enable_timing=True)
                    end_timer = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_timer.record()
                else:
                    start_timer = time.time()

                with accelerator.accumulate(model):
                    if is_latent:
                        # we are using a dataset of precomputed latents, which means any augmenting would've had to have been done before it was encoded
                        reals: FloatTensor = batch[image_key]
                        aug_cond = None
                    else:
                        reals, _, aug_cond = batch[image_key]

                    if args.log_dataset_distribution:
                        prenorm_means: FloatTensor = reals.mean((0, -1, -2))
                        prenorm_vars: FloatTensor = reals.var((0, -1, -2))
                        if is_latent:
                            sdscaled: FloatTensor = reals * vae.config.scaling_factor
                            sdscaled_means: FloatTensor = sdscaled.mean((0, -1, -2))
                            sdscaled_vars: FloatTensor = sdscaled.var((0, -1, -2))
                            del sdscaled

                    # scale-and-shift. usually in order to obtain a standard Gaussian.
                    # latents may do this (to move from VAE distribution to standard Gaussian)
                    # or you may do this in order to shift to sigma_data=1 to use a loss weighting not parameterized on sigma_data
                    if normalizer is not None:
                        normalizer.forward_(reals)
                        if args.log_dataset_distribution:
                            normed_means: FloatTensor = reals.mean((0, -1, -2))
                            normed_vars: FloatTensor = reals.var((0, -1, -2))

                    class_cond, extra_args = None, {}
                    if uses_crossattn:
                        assert precomputed_conds is not None, "cross-attention is currently only implemented for precomputed conditions"
                        xattn_extra_args: CrossAttnExtraArgs = get_precomputed_conds_by_ix(
                            device=accelerator.device,
                            text_uncond_ix=class_captions.uncond_class_ix,
                            masked_conds=precomputed_conds.masked_conds,
                            embed_ix=batch[class_key],
                            cond_dropout_rate=cond_dropout_rate,
                            allzeros_uncond_rate=dataset_config['allzeros_uncond_rate'],
                        )
                        extra_args = {**extra_args, **xattn_extra_args}
                    elif num_classes:
                        class_cond = batch[class_key]
                        drop = torch.rand(class_cond.shape, device=class_cond.device)
                        class_cond.masked_fill_(drop < cond_dropout_rate, num_classes)
                        extra_args[class_cond_key] = class_cond
                    noise = torch.randn_like(reals)
                    with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                        sigma = sample_density([reals.shape[0]], device=device)
                    with K.models.checkpointing(args.checkpointing):
                        if args.use_x0_loss:
                            noised_reals: FloatTensor = reals + noise * K.utils.append_dims(sigma, reals.ndim)
                            denoiseds: FloatTensor = model.forward(noised_reals, sigma, aug_cond=aug_cond, **extra_args)
                            c_weight: FloatTensor = K.layers.loss_weighting_x0(
                                loss_weighting=model_config['loss_weighting'],
                                loss_weighting_params=model_config['loss_weighting_params'],
                                sigma_data=model_config['sigma_data'],
                                sigma=sigma,
                            )
                            mse_losses: FloatTensor = mse_loss_fn(denoiseds, reals).mean(tuple(range(1, denoiseds.ndim)))
                            losses: FloatTensor = mse_losses * c_weight
                        else:
                            losses = model.loss(reals, noise, sigma, aug_cond=aug_cond, **extra_args)
                    loss = accelerator.gather(losses).mean().item()
                    losses_since_last_print.append(loss)
                    accelerator.backward(losses.mean())
                    if args.gns:
                        sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, reals.shape[0], reals.shape[0] * accelerator.num_processes)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    sched.step()
                    opt.zero_grad()

                    ema_decay = ema_sched.get_value()
                    K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay ** (1 / args.grad_accum_steps))
                    if accelerator.sync_gradients:
                        K.utils.ema_update(model, model_ema, ema_decay)
                        ema_sched.step()

                if device.type == 'cuda':
                    end_timer.record()
                    torch.cuda.synchronize()
                    elapsed += start_timer.elapsed_time(end_timer) / 1000
                else:
                    elapsed += time.time() - start_timer

                if step % 25 == 0:
                    loss_disp = sum(losses_since_last_print) / len(losses_since_last_print)
                    losses_since_last_print.clear()
                    avg_loss = ema_stats['loss']
                    if accelerator.is_main_process:
                        if args.gns:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}, gns: {gns_stats.get_gns():g}')
                        else:
                            tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss_disp:g}, avg loss: {avg_loss:g}')

                if use_wandb:
                    log_dict = {
                        'epoch': epoch,
                        'loss': loss,
                        'lr': sched.get_last_lr()[0],
                        'ema_decay': ema_decay,
                    }
                    if args.gns:
                        log_dict['gradient_noise_scale'] = gns_stats.get_gns()
                    if args.log_dataset_distribution:
                        for channel_ix, (prenorm_mean, prenorm_var, normed_mean, normed_var) in enumerate(zip(
                            prenorm_means.unbind(),
                            prenorm_vars.unbind(),
                            normed_means.unbind(),
                            normed_vars.unbind(),
                        )):
                            log_dict[f'latent_reals_prenorm/mean/{channel_ix}'] = prenorm_mean.item()
                            log_dict[f'latent_reals_prenorm/var/{channel_ix}'] = prenorm_var.item()
                            log_dict[f'latent_reals_normed/mean/{channel_ix}'] = normed_mean.item()
                            log_dict[f'latent_reals_normed/var/{channel_ix}'] = normed_var.item()
                        if is_latent:
                            for channel_ix, (sdscaled_mean, sdscaled_var) in enumerate(zip(
                                sdscaled_means.unbind(),
                                sdscaled_vars.unbind(),
                            )):
                                log_dict[f'latent_reals_sdscaled/mean/{channel_ix}'] = sdscaled_mean.item()
                                log_dict[f'latent_reals_sdscaled/var/{channel_ix}'] = sdscaled_var.item()
                    wandb.log(log_dict, step=step)

                step += 1

                if step % args.demo_every == 0:
                    demo(captioner)

                if evaluate_enabled and step > 0 and step % args.evaluate_every == 0:
                    evaluate()

                if step == args.end_step or (step > 0 and step % args.save_every == 0):
                    save()

                if step == args.end_step:
                    if accelerator.is_main_process:
                        tqdm.write('Done!')
                    return

            epoch += 1
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
