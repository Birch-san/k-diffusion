import accelerate
import argparse
import k_diffusion as K
from pathlib import Path
import torch
from torch import distributed as dist, multiprocessing as mp, Tensor, FloatTensor
from torch.utils import data
from torchvision import transforms
from typing import Dict, Optional, Literal, Callable
import math
from kdiff_trainer.dataset.get_dataset import get_dataset

def ensure_distributed():
    if not dist.is_initialized():
        dist.init_process_group(world_size=1, rank=0, store=dist.HashStore())

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size')
    p.add_argument('--config-pred', type=str, required=True,
                   help='configuration file detailing a dataset of predictions from a model')
    p.add_argument('--config-target', type=str, required=True,
                   help='configuration file detailing a dataset of ground-truth examples')
    p.add_argument('--torchmetrics-fid', action='store_true',
                   help='whether to use torchmetrics FID (in addition to CleanFID)')
    p.add_argument('--evaluate-n', type=int, default=2000,
                   help='the number of samples to draw to evaluate')
    p.add_argument('--evaluate-with', type=str, default='inception',
                   choices=['inception', 'clip', 'dinov2'],
                   help='the feature extractor to use for evaluation')
    p.add_argument('--clip-model', type=str, default='ViT-B/16',
                   choices=K.evaluation.CLIPFeatureExtractor.available_models(),
                   help='the CLIP model to use to evaluate')
    p.add_argument('--dinov2-model', type=str, default='vitl14',
                   choices=K.evaluation.DINOv2FeatureExtractor.available_models(),
                   help='the DINOv2 model to use to evaluate')
    p.add_argument('--mixed-precision', type=str,
                   choices=['no', 'fp16', 'bf16', 'fp8'],
                   help='the mixed precision type')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--seed', type=int,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    args = p.parse_args()

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    
    accelerator = accelerate.Accelerator(mixed_precision=args.mixed_precision)
    ensure_distributed()
    device = accelerator.device

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])

    config_pred, config_target = (K.config.load_config(config, use_json5=config.endswith('.jsonc')) for config in (args.config_pred, args.config_target))
    model_config = config_pred['model']
    pred_dataset_config = config_pred['dataset']

    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    target_dataset_config = config_target['dataset']

    tf = transforms.Compose([
        transforms.Resize(size[0], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(size[0]),
        transforms.ToTensor(),
    ])

    pred_train_set, target_train_set = (
        get_dataset(
            dataset_config,
            config_dir=Path(config_path).parent,
            uses_crossattn=False,
            tf=tf,
            class_captions=None,
        ) for dataset_config, config_path in (
            (pred_dataset_config, args.config_pred),
            (target_dataset_config, args.config_target),
        )
    )

    if accelerator.is_main_process:
        for set_name, train_set in zip(('pred', 'target'), (pred_train_set, target_train_set)):
            try:
                print(f'Number of items in {set_name} dataset: {len(train_set):,}')
            except TypeError:
                pass

    pred_train_dl, target_train_dl = (data.DataLoader(train_set, args.batch_size, shuffle=not isinstance(train_set, data.IterableDataset), drop_last=False, num_workers=args.num_workers, persistent_workers=True, pin_memory=True) for train_set in (pred_train_set, target_train_set))
    pred_train_dl, target_train_dl = accelerator.prepare(pred_train_dl, target_train_dl)

    match args.evaluate_with:
        case 'inception':
            extractor = K.evaluation.InceptionV3FeatureExtractor(device=device)
        case 'clip':
            extractor = K.evaluation.CLIPFeatureExtractor(args.clip_model, device=device)
        case 'dinov2':
            extractor = K.evaluation.DINOv2FeatureExtractor(args.dinov2_model, device=device)
        case _:
            raise ValueError(f"Invalid evaluation feature extractor '{args.evaluate_with}'")
    
    pred_train_iter, target_train_iter = (iter(dl) for dl in (pred_train_dl, target_train_dl))
    if args.torchmetrics_fid:
        if accelerator.is_main_process:
            from torchmetrics.image.fid import FrechetInceptionDistance
            # "normalize" means "my images are [0, 1] floats"
            # https://torchmetrics.readthedocs.io/en/stable/image/frechet_inception_distance.html
            fid_obj = FrechetInceptionDistance(feature=2048, normalize=True)
            fid_obj.to(accelerator.device)
        def observe_samples(samples: FloatTensor) -> None:
            accelerator.gather(samples)
            if accelerator.is_main_process:
                fid_obj.update(samples, real=source_name == 'target')
    else:
        observe_samples: Optional[Callable[[FloatTensor], None]] = None
    features: Dict[Literal['pred', 'target'], Optional[Tensor]] = { 'pred': None, 'target': None }
    for source_name, iter_ in zip(('pred', 'target'), (pred_train_iter, target_train_iter)):
        print(f'Computing features for {source_name}...')
        # to anybody who wants to shorten this to a lambda: have you tried putting a breakpoint in a lambda?
        def sample_fn(_) -> Tensor:
            samp = next(iter_)
            return samp[0]
        features[source_name] = K.evaluation.compute_features(accelerator, sample_fn, extractor, args.evaluate_n, args.batch_size, observe_samples=observe_samples)
    if accelerator.is_main_process:
        if args.torchmetrics_fid:
            fid = fid_obj.compute()
            print(f'torchmetrics FID: {fid.item():g}')
        fid = K.evaluation.fid(features['pred'], features['target'])
        kid = K.evaluation.kid(features['pred'], features['target'])
        print(f'CleanFID FID: {fid.item():g}, KID: {kid.item():g}')
        print(f"Finished evaluating {features['pred'].shape[0]} samples.")
        assert features['pred'].shape[0] == args.evaluate_n, f"you requested --evaluate-n={args.evaluate_n}, but we found {features['pred'].shape[0]} samples. probably the trange() inside K.evaluation.compute_features skipped the final batch due to rounding problems. try ensuring that evaluate_n is divisible by batch_size*procs without a remainder."
        assert features['pred'].shape[0] == features['target'].shape[0], f"somehow we have a mismatch between number of ground-truth samples ({features['target'].shape[0]}) and model-predicted samples ({features['pred'].shape[0]})."
    del iter_

if __name__ == '__main__':
    main()
