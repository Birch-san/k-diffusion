import accelerate
import argparse
import k_diffusion as K
from k_diffusion.evaluation import InceptionV3FeatureExtractor, CLIPFeatureExtractor, DINOv2FeatureExtractor
from pathlib import Path
import torch
from torch import distributed as dist, multiprocessing as mp, Tensor, FloatTensor, cat
from torch.utils import data
from torchvision import transforms
from typing import Dict, Optional, Literal, Callable, List, Iterator, Union
from tqdm import tqdm
from itertools import islice
import math
from kdiff_trainer.dataset.get_dataset import get_dataset

ExtractorName = Literal['inception', 'clip', 'dinov2']
ExtractorType = Callable[[FloatTensor], FloatTensor]

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
    p.add_argument('--evaluate-with', type=str, nargs='+', default=['inception'],
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
    p.add_argument('--result-description', type=str, default='',
                   help='preample to include on any result that is written to a file ()')
    p.add_argument('--result-out-file', type=str, default=None,
                   help='file into which to output result')
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

    def get_extractor(e: str) -> Union[InceptionV3FeatureExtractor, CLIPFeatureExtractor, DINOv2FeatureExtractor]:
        if e == 'inception':
            return InceptionV3FeatureExtractor(device=device)
        if e == 'clip':
            return CLIPFeatureExtractor(args.clip_model, device=device)
        if e == 'dinov2':
            return DINOv2FeatureExtractor(args.dinov2_model, device=device)
        raise ValueError(f"Invalid evaluation feature extractor '{e}'")
    extractors: Dict[ExtractorName, ExtractorType] = {e: accelerator.prepare(get_extractor(e)) for e in args.evaluate_with}
    
    pred_train_iter: Iterator[List[FloatTensor]] = iter(pred_train_dl)
    target_train_iter: Iterator[List[FloatTensor]] = iter(target_train_dl)
    if args.torchmetrics_fid:
        from torchmetrics.image.fid import FrechetInceptionDistance, NoTrainInceptionV3
        # "normalize" means "my images are [0, 1] floats"
        normalize=True
        # https://torchmetrics.readthedocs.io/en/stable/image/frechet_inception_distance.html
        fid_obj = FrechetInceptionDistance(feature=2048, normalize=normalize)
        fid_obj: FrechetInceptionDistance = accelerator.prepare(fid_obj)
        inception: NoTrainInceptionV3 = fid_obj.inception
        # first part of FrechetInceptionDistance#update
        def compute_features(imgs: FloatTensor) -> FloatTensor:
            imgs = (imgs * 255).byte() if normalize else imgs
            features: FloatTensor = inception(imgs)
            return accelerator.gather(features)

        if accelerator.is_main_process:
            # this is just FrechetInceptionDistance#update, except with support for gather.
            # only main process accumulates.
            def update_torchmetrics_fid(imgs: FloatTensor, real: bool) -> None:
                features: FloatTensor = compute_features(imgs)
                fid_obj.orig_dtype = features.dtype
                features = features.double()
                if features.dim() == 1:
                    features = features.unsqueeze(0)
                sum = features.sum(dim=0)
                cov_sum = features.t().mm(features)
                num_samples: int = features.shape[0]
                if real:
                    fid_obj.real_features_sum += sum
                    fid_obj.real_features_cov_sum += cov_sum
                    fid_obj.real_features_num_samples += num_samples
                else:
                    fid_obj.fake_features_sum += sum
                    fid_obj.fake_features_cov_sum += cov_sum
                    fid_obj.fake_features_num_samples += num_samples
        else:
            del fid_obj
            def update_torchmetrics_fid(imgs: FloatTensor, _) -> None:
                # participate in gather
                compute_features(imgs)
    else:
        def update_torchmetrics_fid(*_) -> None: pass
    samples_per_batch_all_proc: int = args.batch_size * accelerator.num_processes
    batches_per_proc: int = math.ceil(args.evaluate_n / samples_per_batch_all_proc)
    if accelerator.is_main_process:
        features: Dict[ExtractorName, Dict[Literal['pred', 'target'], List[FloatTensor]]] = {
            extractor: { 'pred': [], 'target': [] } for extractor in extractors.keys()
        }
    for subject, iter_ in zip(('pred', 'target'), (pred_train_iter, target_train_iter)):
        for batch_ix, batch in enumerate(tqdm(
            islice(iter_, batches_per_proc),
            f'Computing features for {subject}...',
            disable=not accelerator.is_main_process,
            position=0,
            total=batches_per_proc,
            unit='batch',
        )):
            samples, *_ = batch
            curr_total: int = batch_ix * samples_per_batch_all_proc
            samples_remaining: int = args.evaluate_n-curr_total
            # (optimization): use a smaller per-proc batch when possible for the final batch
            curr_batch_per_proc: int = min(args.batch_size, math.ceil(samples_remaining / accelerator.num_processes))
            samples = samples[:curr_batch_per_proc]
            for extractor_name, extractor in extractors.items():
                extracted: FloatTensor = extractor(samples)
                extracted = accelerator.gather(extracted)
                if accelerator.is_main_process:
                    extracted = extracted[:min(samples_per_batch_all_proc, samples_remaining)]
                    features[extractor_name][subject].append(extracted)
                del extracted
            update_torchmetrics_fid(samples, subject == 'target')
    if accelerator.is_main_process:
        # using new variable name rather than reassigning, to get type inference to use the new type
        features_: Dict[ExtractorName, Dict[Literal['pred', 'target'], FloatTensor]] = {
            extractor_name: {
                subject: cat(features_list) for subject, features_list in features_by_subject.items()
            } for extractor_name, features_by_subject in features.items()
        }
        del features
        if args.torchmetrics_fid:
            fid: FloatTensor = fid_obj.compute()
            print(f'torchmetrics, {fid_obj.fake_features_num_samples} samples:')
            print(f'  FID: {fid.item()}')
            assert fid_obj.fake_features_num_samples == args.evaluate_n, f"[torchmetrics FID] you requested --evaluate-n={args.evaluate_n}, but we found {fid_obj.fake_features_num_samples} samples. perhaps the final batch was skipped due to rounding problems. try ensuring that evaluate_n is divisible by batch_size*procs without a remainder, or try simplifying the multi-processing (i.e. single-node or single-GPU)."
            assert fid_obj.fake_features_num_samples == fid_obj.real_features_num_samples, f"[torchmetrics FID] somehow we have a mismatch between number of ground-truth samples ({fid_obj.real_features_num_samples}) and model-predicted samples ({fid_obj.fake_features_num_samples})."
        for extractor_name, features_by_subject in features_.items():
            pred, target = features_by_subject['pred'], features_by_subject['target']
            initial: Literal['I', 'C', 'D'] = extractor_name[0].upper()
            fid: FloatTensor = K.evaluation.fid(pred, target)
            kid: FloatTensor = K.evaluation.kid(pred, target)
            print(f'{extractor_name}, {pred.shape[0]} samples:')
            print(f'  F{initial}D: {fid.item():g}')
            print(f'  K{initial}D: {kid.item():g}')
            assert pred.shape[0] == args.evaluate_n, f"you requested --evaluate-n={args.evaluate_n}, but we found {pred.shape[0]} samples. perhaps the final batch was skipped due to rounding problems. try ensuring that evaluate_n is divisible by batch_size*procs without a remainder, or try simplifying the multi-processing (i.e. single-node or single-GPU)."
            assert pred.shape[0] == target.shape[0], f"somehow we have a mismatch between number of ground-truth samples ({target.shape[0]}) and model-predicted samples ({pred.shape[0]})."
    del iter_

if __name__ == '__main__':
    main()
