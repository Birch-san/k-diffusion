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
import torch
import torch.nn.functional as F
import torch._dynamo
from torch import distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch import multiprocessing as mp
from torch import optim, FloatTensor
from torch.nn import Module, MSELoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils import data
from torch.utils.data.dataset import Dataset, IterableDataset
from tqdm.auto import tqdm
from typing import Any, List, Optional, Union, Protocol, Iterator
from PIL import Image
from dataclasses import dataclass
from typing import Optional, TypedDict, Generator, Callable, Dict, Any, Tuple
from contextlib import nullcontext
from itertools import islice
from tqdm import tqdm
import numpy as np
from diffusers import ConsistencyDecoderVAE
from diffusers.models.unet_2d_blocks import ResnetDownsampleBlock2D, ResnetUpsampleBlock2D
from peft import LoraConfig, get_peft_model
from lpips import LPIPS
import gc
from peft import PeftModel
import re

import k_diffusion as K
from kdiff_trainer.dimensions import Dimensions
from kdiff_trainer.make_default_grid_captioner import make_default_grid_captioner
from kdiff_trainer.make_captioned_grid import GridCaptioner
from kdiff_trainer.to_pil_images import to_pil_images, to_pil_images_from_0_1
from kdiff_trainer.iteration.batched import batched
from kdiff_trainer.dataset.get_latent_dataset import get_latent_dataset, LatentImgPair
from kdiff_trainer.normalize import Normalize

from sdxl_diff_dec.schedule import betas_for_alpha_bar, alpha_bar, get_alphas
from sdxl_diff_dec.sd_denoiser import SDDecoderDistilled
from sdxl_diff_dec.prune_conv_out import prune_conv_out

SinkOutput = TypedDict('SinkOutput', {
    '__key__': str,
    'img.png': Image.Image,
})

@dataclass
class Samples:
    # [-1., 1.] (standard Gaussian)
    x_0: FloatTensor
    #  [0., 1.] (torchvision-style)
    reals: FloatTensor

@dataclass
class Sample:
    pred: Image.Image
    real: Image.Image

class Sampler(Protocol):
    @staticmethod
    def __call__(model_fn: Callable, x: FloatTensor, sigmas: FloatTensor, extra_args: Dict[str, Any]) -> Any: ...

def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())

def find_all_conv_names(model: ConsistencyDecoderVAE) -> List[str]:
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)

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
    p.add_argument('--demo-steps', type=int, default=50,
                   help='the number of steps to sample for demo grids')
    p.add_argument('--sample-n', type=int, default=64,
                   help='the number of images to sample for demo grids')
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
    args = p.parse_args()

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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
    assert model_config['type'] == 'diff_dec'
    train_dataset_config = config['dataset']
    test_dataset_config = config['dataset_test']
    if do_train:
        opt_config = config['optimizer']
        sched_config = config['lr_sched']
        ema_sched_config = config['ema_sched']
        lora_config = model_config['lora']
    else:
        opt_config = sched_config = ema_sched_config = lora_config = None

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

    cvae: ConsistencyDecoderVAE = ConsistencyDecoderVAE.from_pretrained(
        'openai/consistency-decoder',
        # variant='fp16',
        # torch_dtype=torch.float16,
        # device_map={'': f'{device.type}:{device.index or 0}'},
    ).train()
    del cvae.encoder, cvae.means, cvae.stds, cvae.decoder_scheduler

    # prune away the unused channels
    prune_conv_out(cvae.decoder_unet)

    vae_scale_factor: int = 1 << (len(cvae.config.block_out_channels) - 1)
    out_size: Tuple[int, int] = size[0]*vae_scale_factor, size[1]*vae_scale_factor

    if args.checkpointing:
        # one day they'll support this
        if cvae.decoder_unet._supports_gradient_checkpointing:
            cvae.decoder_unet.enable_gradient_checkpointing()
        else:
            def enable_ckpt(module: Module) -> None:
                match(module):
                    case ResnetDownsampleBlock2D() | ResnetUpsampleBlock2D():
                        module.gradient_checkpointing = True
            cvae.decoder_unet.apply(enable_ckpt)

    state_path = Path(f'{state_root}/{run_qualifier}state.json')
    if state_path.exists():
        loaded_state: Dict = json.load(open(state_path))
        ckpt_path = f"{state_root}/{loaded_state['latest_checkpoint']}"
        if accelerator.is_main_process:
            print(f'Resuming LoRA from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        lora_ckpt_dir: str = ckpt['lora_ckpt_dir']
        lora_ckpt_dir_qualified: str = f'{state_root}/{lora_ckpt_dir}'
        lora_config_obj: LoraConfig = ckpt['lora_config']
        if do_train:
            inner_model = PeftModel.from_pretrained(cvae.decoder_unet, f'{lora_ckpt_dir_qualified}/model', adapter_name='model', is_trainable=True)
        else:
            inner_model = None
        inner_model_ema = PeftModel.from_pretrained(cvae.decoder_unet, f'{lora_ckpt_dir_qualified}/model_ema', adapter_name='model_ema', is_trainable=False)
        torch.cuda.empty_cache()
    else:
        assert do_train
        loaded_state: Optional[Dict] = None
        if accelerator.is_main_process:
            print(f'adding LoRA modules...')
        modules = find_all_conv_names(cvae.decoder_unet)
        lora_config_obj = LoraConfig(
            r=lora_config['rank'],
            lora_alpha=lora_config['alpha'],
            target_modules=modules,
            lora_dropout=lora_config['dropout'],
            bias="none",
        )
        inner_model = get_peft_model(cvae.decoder_unet, lora_config_obj, adapter_name='model')
        inner_model_ema = get_peft_model(cvae.decoder_unet, lora_config_obj, adapter_name='model_ema')
        for name, param in cvae.decoder_unet.named_parameters():
            if re.search(K.utils.model_ema_lora_param_match, name):
                param.requires_grad_(False)
        ckpt = None
    if args.compile:
        torch.compile(inner_model, fullgraph=True, mode='max-autotune')
        torch.compile(inner_model_ema, fullgraph=True, mode='max-autotune')

    # If logging to wandb, initialize the run
    use_wandb = accelerator.is_main_process and args.wandb_project
    if use_wandb:
        import wandb
        log_config = vars(deepcopy(args))
        log_config['config'] = config
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group, config=log_config, name=args.wandb_run_name, save_code=True)

    if do_train:
        lr = opt_config['lr'] if args.lr is None else args.lr
        groups = [
            {"params": [p for p in inner_model.parameters() if p.requires_grad], "lr": lr}
        ]
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

    is_latent: bool = train_dataset_config.get('latents', False)
    assert is_latent
    assert test_dataset_config.get('latents', False) == is_latent

    # we don't do resize & center-crop, because our latent datasets are precomputed
    # (via imagenet_vae_loading.py) for a given canvas size
    train_set: Union[Dataset, IterableDataset] = get_latent_dataset(train_dataset_config)
    test_set: Union[Dataset, IterableDataset] = get_latent_dataset(test_dataset_config)

    train_channel_means: FloatTensor = torch.tensor(train_dataset_config['channel_means'])
    train_channel_squares: FloatTensor = torch.tensor(train_dataset_config['channel_squares'])
    train_channel_stds: FloatTensor = torch.sqrt(train_channel_squares - train_channel_means**2)
    train_normalizer = Normalize(train_channel_means, train_channel_stds)
    del train_channel_means, train_channel_squares, train_channel_stds

    test_channel_means: FloatTensor = torch.tensor(test_dataset_config['channel_means'])
    test_channel_squares: FloatTensor = torch.tensor(test_dataset_config['channel_squares'])
    test_channel_stds: FloatTensor = torch.sqrt(test_channel_squares - test_channel_means**2)
    test_normalizer = Normalize(test_channel_means, test_channel_stds)
    del test_channel_means, test_channel_squares, test_channel_stds

    accelerator.prepare(train_normalizer, test_normalizer)

    if accelerator.is_main_process:
        try:
            print(f'Number of items in dataset: {len(train_set):,}')
        except TypeError:
            pass

    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=not isinstance(train_set, data.IterableDataset), drop_last=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    n_demo_samples_per_proc: int = math.ceil(args.sample_n / accelerator.num_processes)
    test_dl = data.DataLoader(test_set, n_demo_samples_per_proc, shuffle=not isinstance(test_set, data.IterableDataset), drop_last=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)

    # TODO: work out whether there's situations where we wouldn't need both dataloaders
    if do_train:
        opt, train_dl, test_dl = accelerator.prepare(opt, train_dl, test_dl)
        if use_wandb:
            wandb.watch(inner_model)
    else:
        train_dl, test_dl = accelerator.prepare(train_dl, test_dl)

    if accelerator.num_processes == 1:
        args.gns = False
    if args.gns and do_train:
        gns_stats_hook = K.gns.DDPGradientStatsHook(inner_model)
        gns_stats = K.gns.GradientNoiseScale()
    else:
        gns_stats = None

    num_timesteps: int = model_config['num_timesteps']
    betas = betas_for_alpha_bar(num_timesteps, alpha_bar, device=device)
    alphas: FloatTensor = get_alphas(betas)
    alphas_cumprod: FloatTensor = alphas.cumprod(dim=0)

    if do_train:
        sample_density = partial(K.utils.rand_uniform, min_value=0, max_value=num_timesteps-1)
        model = SDDecoderDistilled(
            inner_model,
            alphas_cumprod,
            total_timesteps=num_timesteps,
            n_distilled_steps=64,
            dtype=torch.float32,
        )
    model_ema = SDDecoderDistilled(
        inner_model_ema,
        alphas_cumprod,
        total_timesteps=num_timesteps,
        n_distilled_steps=64,
        dtype=torch.float32,
    )
    sigma_min: float = model_ema.sigma_min.item()
    sigma_max: float = model_ema.sigma_max.item()

    sampling_steps = 2
    consistency_sigmas: FloatTensor = model.get_sigmas_rounded(n=sampling_steps+1, include_sigma_min=False, t_max_exclusion='shift')

    if loaded_state is not None:
        if accelerator.is_main_process:
            print(f'Resuming optimiser and schedules from {ckpt_path}...')
        assert ckpt is not None
        if do_train:
            opt.load_state_dict(ckpt['opt'])
            sched.load_state_dict(ckpt['sched'])
            ema_sched.load_state_dict(ckpt['ema_sched'])
            ema_stats = ckpt.get('ema_stats', ema_stats)
            # I goofed in one of the checkpoints I saved
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
        # it seems to be common practice to eval against train set
        train_reals_iter: Iterator[LatentImgPair] = iter(train_dl)
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
        def sample_fn(cur_batch_size: int) -> FloatTensor:
            _, rgb = next(train_reals_iter)
            # shift from [0., 1.] to [-1., 1.]
            # cast here to ensure we have same dtype as our fakes (which come out of the float32 ema_model).
            # rgb datasets have float32 reals because they go through KarrasAugmentationPipeline, which returns a float32 tensor.
            return rgb.mul_(2).sub_(1).float()
        reals_features = K.evaluation.compute_features(accelerator, sample_fn, extractor, args.evaluate_n, args.batch_size, observe_samples=observe_samples_real)
        if accelerator.is_main_process and not args.evaluate_only:
            fid_cols: List[str] = ['fid']
            if args.torchmetrics_fid:
                fid_cols.append('tfid')
            metrics_log = K.utils.CSVLogger(f'{metrics_root}/{run_qualifier}metrics.csv', ['step', 'time', 'loss', *fid_cols, 'kid'])
        del train_reals_iter
    
    test_iter: Iterator[LatentImgPair] = iter(test_dl)
    def generate_batch_of_samples() -> Samples:
        batch: LatentImgPair = next(test_iter)
        latents, reals = batch
        assert latents.shape[0] == n_demo_samples_per_proc
        assert reals.shape[0] == n_demo_samples_per_proc
        # scale-and-shift from VAE distribution, to standard Gaussian
        test_normalizer.forward_(latents)
        latents: FloatTensor = F.interpolate(latents, mode="nearest", scale_factor=8).detach()
        extra_args = {'latents': latents}
        x = torch.randn([accelerator.num_processes, n_demo_samples_per_proc, 3, out_size[0], out_size[1]], generator=demo_gen).to(device)
        dist.broadcast(x, 0)
        x = x[accelerator.process_index] * sigma_max

        with torch.inference_mode():
            x_0: FloatTensor = K.sampling.sample_consistency(model_ema, x, consistency_sigmas, extra_args=extra_args, disable=True)
        x_0 = accelerator.gather(x_0)[:args.sample_n]
        reals = accelerator.gather(reals)[:args.sample_n]

        return Samples(x_0, reals)
    
    @torch.inference_mode() # note: inference_mode is lower-overhead than no_grad but disables forward-mode AD
    @K.utils.eval_mode(model_ema)
    def generate_samples() -> Generator[Optional[Sample], None, None]:
        if accelerator.is_main_process:
            tqdm.write('Sampling...')
        with FSDP.summon_full_params(model_ema):
            pass

        while True:
            batch: Samples = generate_batch_of_samples()
            if accelerator.is_main_process:
                preds: List[Image.Image] = to_pil_images(batch.x_0)
                reals: List[Image.Image] = to_pil_images_from_0_1(batch.reals)
                for pred, real in zip(preds, reals):
                    yield Sample(pred, real)
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

        sample_gen: Iterator[Optional[Sample]] = islice(generate_samples(), args.sample_n)
        samples: List[Sample] = list(sample_gen)
        del sample_gen
        if accelerator.is_main_process:
            captions: List[str] = [caption for ix in range(len(samples)) for caption in (f'real {ix}', f'pred {ix}')]
            imgs: List[Image.Image] = [img for sample in samples for img in (sample.real, sample.pred)]
            title = f'[step {step}] {args.name} {args.config}'
            if args.demo_title_qualifier:
                title += f' {args.demo_title_qualifier}'
            grid_pil: Image.Image = captioner.__call__(
                imgs=imgs,
                captions=captions,
                title=title,
            )
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
            # TODO
            x = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
            model_fn, extra_args = model_ema, {}
            x_0: FloatTensor = do_sample(model_fn, x, sigmas, extra_args=extra_args, disable=True)
            return x_0
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

            ckpt_step_dir = f'{ckpt_root}/{run_qualifier}{step:08}'
            inner_model.save_pretrained(save_directory=ckpt_step_dir, safe_serialization=True, selected_adapters=['model'])
            inner_model_ema.save_pretrained(save_directory=ckpt_step_dir, safe_serialization=True, selected_adapters=['model_ema'])
            obj = {
                'config': config,
                'lora_config': lora_config_obj,
                # 'model': inner_model.state_dict(),
                # 'model_ema': inner_model_ema.state_dict(),
                'lora_ckpt_dir': relpath(ckpt_step_dir, state_root),
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
    samp_h, samp_w = out_size
    sample_size = Dimensions(height=samp_h, width=samp_w)
    captioner: GridCaptioner = make_default_grid_captioner(
        font_path=args.font,
        sample_n=args.sample_n,
        sample_size=sample_size,
        cols=(math.ceil(args.sample_n ** .5)//2)*2
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
                def sink_sample(sink: ShardWriter, ix: int, sample: Sample) -> None:
                    out: SinkOutput = {
                        '__key__': f'{shard_qualifier}{ix}',
                        'pred.png': sample.pred,
                        'real.png': sample.real,
                    }
                    sink.write(out)
            else:
                sink_ctx = nullcontext()
                sink_sample: Callable[[Optional[ShardWriter], int, Sample], None] = lambda *_: ...
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
            tqdm.write('Finished inferencing!')
        return

    if args.evaluate_only:
        if args.evaluate_n < 1:
            raise ValueError('--evaluate-only requested but --evaluate-n is less than 1')
        evaluate()
        if accelerator.is_main_process:
            tqdm.write('Finished evaluating!')
        return

    mse = MSELoss()
    lpips = LPIPS(net='vgg').to(device)
    mse_weight = 0.1
    lpips_weight = 1-mse_weight

    losses_since_last_print: List[float] = []

    try:
        while True:
            train_iter: Iterator[LatentImgPair] = iter(train_dl)
            for batch in tqdm(train_iter, smoothing=0.1, disable=not accelerator.is_main_process):
                if device.type == 'cuda':
                    start_timer = torch.cuda.Event(enable_timing=True)
                    end_timer = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start_timer.record()
                else:
                    start_timer = time.time()

                with accelerator.accumulate(model):
                    # we are using a dataset of precomputed latents, which means any augmenting would've had to have been done before it was encoded
                    latents, reals = batch
                    # scale-and-shift from VAE distribution, to standard Gaussian
                    train_normalizer.forward_(latents)
                    # move from torchvision [0., 1.] to standard [-1., 1.]
                    reals.mul_(2).sub_(1).detach()
                    latents: FloatTensor = F.interpolate(latents, mode="nearest", scale_factor=8).detach()
                    extra_args = {'latents': latents}
                    noise = torch.randn_like(reals)
                    # with K.utils.enable_stratified_accelerate(accelerator, disable=args.gns):
                    #     sigma = sample_density([reals.shape[0]], device=device)
                    # with K.models.checkpointing(args.checkpointing):
                    #     losses = model.loss(reals, noise, sigma, **extra_args)
                    sample: FloatTensor = K.sampling.sample_consistency(model, noise, consistency_sigmas, extra_args=extra_args, disable=True)
                    mse_losses: FloatTensor = mse(sample, reals)
                    lpips_losses: FloatTensor = lpips(sample, reals)
                    mse_losses.mul_(mse_weight)
                    lpips_losses.mul_(lpips_weight)
                    lpips_losses.add_(mse_losses)
                    losses: FloatTensor = accelerator.gather(lpips_losses)
                    del mse_losses, lpips_losses
                    loss: FloatTensor = losses.mean()
                    losses_since_last_print.append(loss.item())
                    accelerator.backward(loss)
                    if args.gns:
                        sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                        gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, reals.shape[0], reals.shape[0] * accelerator.num_processes)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    sched.step()
                    opt.zero_grad()

                    ema_decay = ema_sched.get_value()
                    K.utils.ema_update_dict(ema_stats, {'loss': loss.item()}, ema_decay ** (1 / args.grad_accum_steps))
                    if accelerator.sync_gradients:
                        K.utils.lora_ema_update(model.inner_model.base_model, model_ema.inner_model.base_model, ema_decay)
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
