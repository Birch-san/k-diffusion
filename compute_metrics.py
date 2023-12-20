import accelerate
import argparse
import k_diffusion as K
from k_diffusion.evaluation import InceptionV3FeatureExtractor, CLIPFeatureExtractor, DINOv2FeatureExtractor
from pathlib import Path
import torch
from torch import distributed as dist, multiprocessing as mp, FloatTensor, zeros, inference_mode
from torch.nn import Sequential
from torch.nn.functional import kl_div
from torch.utils import data
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms
from typing import Dict, Literal, Callable, List, Iterator, Union, Set, TypedDict, Optional, Any, NamedTuple, Tuple
from tqdm import tqdm
import math
from os.path import dirname
from os import makedirs
from functools import lru_cache
import gc

from kdiff_trainer.dataset.get_dataset import get_dataset
from kdiff_trainer.eval.sfid import SFID
from kdiff_trainer.eval.inception_score import InceptionLogits
from kdiff_trainer.eval.bicubic_resize import BicubicResize
from kdiff_trainer.eval.inceptionv3_resize import InceptionV3Resize
from kdiff_trainer.normalize import Normalize_

ExtractorName = Literal['inception', 'clip', 'dinov2', 'sfid-bicubic', 'sfid-bilinear', 'inception-score-bicubic', 'inception-score-bilinear']
ExtractorType = Callable[[FloatTensor], FloatTensor]

class FeaturesBySubject(TypedDict):
    pred: FloatTensor
    # inception score doesn't require targets
    target: Optional[FloatTensor]

class DatasetConfigAndPath(NamedTuple):
    dataset_config: Dict[str, Any]
    config_path: str

inception_feature_size=2048
imagenet_classes=1000
sfid_feature_size=2023
extractor_feature_size: Dict[ExtractorName, int] = {
    'inception': inception_feature_size,
    'clip': 512,
    'dinov2': 1024,
    'sfid-bicubic': sfid_feature_size,
    'sfid-bilinear': sfid_feature_size,
    'inception-score-bicubic': imagenet_classes,
    'inception-score-bilinear': imagenet_classes,
}
does_extractor_need_targets: Callable[[str], bool]  = lambda extractor: not extractor.startswith('inception-score')
extractor_target_needs: Dict[ExtractorName, int] = { extractor: does_extractor_need_targets(extractor) for extractor in extractor_feature_size.keys() }

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
    p.add_argument('--config-target', type=str, required=False,
                   help='configuration file detailing a dataset of ground-truth examples')
    p.add_argument('--torchmetrics-fid', action='store_true',
                   help='whether to use torchmetrics FID (in addition to CleanFID)')
    p.add_argument('--compile', action='store_true',
                   help='whether to torch.compile() the feature extractors')
    p.add_argument('--evaluate-n', type=int, default=2000,
                   help='the number of samples to draw to evaluate')
    p.add_argument('--evaluate-n-reals', type=int, default=None,
                   help='number of reals to draw (if different from --evaluate-n)')
    p.add_argument('--evaluate-with', type=str, nargs='+', default=['inception'],
                   choices=['inception', 'clip', 'dinov2', 'sfid-bicubic', 'sfid-bilinear', 'inception-score-bicubic', 'inception-score-bilinear'],
                   help='the feature extractor to use for evaluation')
    p.add_argument('--inception-score-splits', type=int, default=10,
                   help='for Inception Score metric: how into how many cohorts should we partition our corpus of samples')
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
    p.add_argument('--shuffle-fakes', default=True, action=argparse.BooleanOptionalAction)
    p.add_argument('--shuffle-reals', default=True, action=argparse.BooleanOptionalAction)
    args = p.parse_args()

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    
    accelerator = accelerate.Accelerator(mixed_precision=args.mixed_precision)
    ensure_distributed()
    device = accelerator.device
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            print("!!!\n[WARNING] multi-GPU is known to exhaust the dataloader iterator without yielding all the expected samples, at least for wds datasets. An assertion will tell you if this happens. You may need to try single-GPU.\n!!!")

    if args.seed is not None:
        seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
        torch.manual_seed(seeds[accelerator.process_index])

    if any((extractor.startswith('sfid-') for extractor in args.evaluate_with)):
        print("!!!\n[WARNING] This sFID implementation gives results much higher than those of OpenAI evaluator.py. Probably wrong. You probably shouldn't use it, unless you are here to fix it.\n!!!")

    any_needs_targets: bool = any((extractor_target_needs[extractor] for extractor in args.evaluate_with))
    if any_needs_targets:
        if args.config_target is None:
            raise ValueError('--config-target must be specified because one of the evaluators specified in --evaluate-with compares predictions to targets')

    configs: List[Dict[str, Any]] = [K.config.load_config(config, use_json5=config.endswith('.jsonc')) for config in (args.config_pred, *(args.config_target,)*any_needs_targets)]
    config_pred, *_ = configs
    model_config = config_pred['model']

    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    tf = transforms.Compose([
        transforms.Resize(size[0], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(size[0]),
        transforms.ToTensor(),
    ])

    dataset_config_and_paths: List[DatasetConfigAndPath] = []
    dataset_config_and_paths.append(DatasetConfigAndPath(config_pred['dataset'], args.config_pred))
    if any_needs_targets:
        _, config_target = configs
        dataset_config_and_paths.append(DatasetConfigAndPath(config_target['dataset'], args.config_target))
    wants_shuffles: Tuple[bool, ...] = (args.shuffle_fakes, *(args.shuffle_reals,)*any_needs_targets)
    datasets: List[Union[Dataset, IterableDataset]] = [get_dataset(
        dataset_config,
        config_dir=Path(config_path).parent,
        uses_crossattn=False,
        tf=tf,
        class_captions=None,
        shuffle_wds=wants_shuffle,
    ) for (dataset_config, config_path), wants_shuffle in zip(dataset_config_and_paths, wants_shuffles)]

    if accelerator.is_main_process:
        for set_name, train_set in zip(('pred', *('target',)*any_needs_targets), datasets):
            try:
                print(f'Number of items in {set_name} dataset: {len(train_set):,}')
            except TypeError:
                pass

    dls: List[data.DataLoader] = [data.DataLoader(train_set, args.batch_size, shuffle=wants_shuffle and not isinstance(train_set, data.IterableDataset), drop_last=False, num_workers=args.num_workers, persistent_workers=args.num_workers>0, pin_memory=True) for train_set, wants_shuffle in zip(datasets, wants_shuffles)]
    dls = accelerator.prepare(*dls)
    # if you give prepare() a list (n>1), it returns a Tuple[DataLoader, ...].
    # if you give prepare() a list (n==1), it returns a Dataloader.
    # it's an inconsistent contract. we tuple-ize it to compensate for that.
    if not isinstance(dls, tuple):
        dls = (dls,)

    # taking an iterator is superfluous but gives us a type hint
    iters: List[Iterator[List[FloatTensor]]] = [iter(dl) for dl in dls]
    del datasets, dls

    @lru_cache(maxsize=1)
    def get_inception() -> InceptionV3FeatureExtractor:
        return InceptionV3FeatureExtractor(device=device)
    
    @lru_cache(maxsize=1)
    def get_sfid() -> SFID:
        inception: InceptionV3FeatureExtractor = get_inception()
        sfid = SFID(inception.model.base)
        return sfid
    
    @lru_cache(maxsize=1)
    def get_inception_logits_model() -> InceptionLogits:
        inception: InceptionV3FeatureExtractor = get_inception()
        inception_logits = InceptionLogits(inception.model.base)
        return inception_logits
    
    @lru_cache(maxsize=1)
    def get_normalize() -> Normalize_:
        # (0, 1) to (-1, 1)
        normalize = Normalize_(.5, .5).to(device)
        return normalize

    def get_extractor(e: str) -> Union[InceptionV3FeatureExtractor, CLIPFeatureExtractor, DINOv2FeatureExtractor, Sequential]:
        if e == 'inception':
            return get_inception()
        if e == 'clip':
            return CLIPFeatureExtractor(args.clip_model, device=device)
        if e == 'dinov2':
            return DINOv2FeatureExtractor(args.dinov2_model, device=device)
        if e == 'sfid-bicubic':
            sfid = get_sfid()
            bicubic_resize = BicubicResize()
            normalize: Normalize_ = get_normalize()
            return Sequential(bicubic_resize, normalize, sfid)
        if e == 'sfid-bilinear':
            sfid: SFID = get_sfid()
            inception_resize = InceptionV3Resize()
            normalize: Normalize_ = get_normalize()
            return Sequential(inception_resize, normalize, sfid)
        if e == 'inception-score-bicubic':
            inception_logits: InceptionLogits = get_inception_logits_model()
            bicubic_resize = BicubicResize()
            normalize: Normalize_ = get_normalize()
            return Sequential(bicubic_resize, normalize, inception_logits)
        if e == 'inception-score-bilinear':
            inception_logits: InceptionLogits = get_inception_logits_model()
            inception_resize = InceptionV3Resize()
            normalize: Normalize_ = get_normalize()
            return Sequential(inception_resize, normalize, inception_logits)
        raise ValueError(f"Invalid evaluation feature extractor '{e}'")
    
    extractors: Dict[ExtractorName, ExtractorType] = {e: accelerator.prepare(get_extractor(e)) for e in args.evaluate_with}

    if args.compile:
        for extractor in extractors.values():
            torch.compile(extractor, fullgraph=True, mode='max-autotune')

    if args.torchmetrics_fid:
        from torchmetrics.image.fid import FrechetInceptionDistance
        # "normalize" means "my images are [0, 1] floats"
        # https://torchmetrics.readthedocs.io/en/stable/image/frechet_inception_distance.html
        fid_obj = FrechetInceptionDistance(feature=2048, normalize=True)
        fid_obj: FrechetInceptionDistance = accelerator.prepare(fid_obj)
        if args.compile:
            torch.compile(fid_obj, fullgraph=True, mode='max-autotune')
    
    fakes_quota: int = args.evaluate_n
    reals_quota: int = args.evaluate_n if args.evaluate_n_reals is None else args.evaluate_n_reals

    if accelerator.is_main_process:
        reals_msg = f' against {reals_quota} reals' if any_needs_targets else ''
        quota_msg = f'Evaluating {fakes_quota} fakes{reals_msg}'
        print(quota_msg)
    
    if accelerator.is_main_process:
        features: Dict[ExtractorName, FeaturesBySubject] = {}
        for extractor in extractors.keys():
            feature_dim: int = extractor_feature_size[extractor]
            needs_targets: bool = extractor_target_needs[extractor]
            pred_features: FloatTensor = zeros([fakes_quota, feature_dim], device=device, dtype=torch.float32)
            # inception-score doesn't need targets
            target_features: Optional[FloatTensor] = zeros([reals_quota, feature_dim], device=device, dtype=torch.float32) if needs_targets else None
            features_by_subject_ = FeaturesBySubject(pred=pred_features, target=target_features)
            features[extractor] = features_by_subject_

    for subject, iter_, quota in zip(
        ('pred', *('target',)*any_needs_targets),
        iters,
        (fakes_quota, *(reals_quota,)*any_needs_targets),
    ):
        # batch sizes from torch dataloader are not *necessarily* equal to the args.batch_size you requested!
        # dataloaders based on webdataset will give a smaller batch when they exhaust a .tar shard.
        total_samples = 0
        with tqdm(
            desc=f'Computing features for {subject}...',
            total=quota,
            disable=not accelerator.is_main_process,
            position=0,
            unit='samp',
        ) as pbar:
            for batch in iter_:
                samples, *_ = batch
                del batch
                samples_remaining: int = quota-total_samples
                # (optimization): use a smaller per-proc batch if possible (only relevant for the final batch)
                curr_batch_per_proc: int = min(samples.shape[0], math.ceil(samples_remaining / accelerator.num_processes))
                samples = samples[:curr_batch_per_proc]
                # we may be forced to overshoot in order to give every process an equal batch size. we discard the excess.
                samples_kept: int = min(curr_batch_per_proc * accelerator.num_processes, samples_remaining)
                for extractor_name, extractor in extractors.items():
                    if subject == 'target':
                        needs_targets: bool = extractor_target_needs[extractor_name]
                        if not needs_targets:
                            break
                    with inference_mode():
                        extracted: FloatTensor = extractor(samples)
                    extracted = accelerator.gather(extracted)
                    if accelerator.is_main_process:
                        extracted = extracted[:samples_kept]
                        features[extractor_name][subject][total_samples:total_samples+extracted.shape[0]] = extracted
                    del extracted
                if args.torchmetrics_fid:
                    with inference_mode():
                        fid_obj.update(samples, subject == 'target')
                del samples
                total_samples += samples_kept
                pbar.update(samples_kept)
                assert total_samples <= quota
                gc.collect()
                if total_samples == quota:
                    break
            else:
                raise RuntimeError(f'Exhausted iterator before reaching {quota} {subject} samples. Only got {total_samples}. If you are using multi-GPU: try, uh, not using multi-GPU.')
    del iters, extractors
    gc.collect()

    receipt_lines: List[str] = [args.result_description]
    def add_to_receipt(line: str) -> str:
        receipt_lines.append(line)
        return line

    if args.torchmetrics_fid:
        # all processes must participate in all-gather
        # FrechetInceptionDistance's superclass "Metric" wraps FrechetInceptionDistance#update and FrechetInceptionDistance#compute to support distributed operation
        tm_fid: FloatTensor = fid_obj.compute()
        if accelerator.is_main_process:
            print(add_to_receipt(f'torchmetrics, {fid_obj.fake_features_num_samples} fakes vs {fid_obj.real_features_num_samples} reals:'))
            print(add_to_receipt(f'  FID: {tm_fid.item()}'))
            assert fid_obj.fake_features_num_samples.item() == fakes_quota, f"[torchmetrics FID] you requested --evaluate-n={fakes_quota}, but we found {fid_obj.fake_features_num_samples} samples. perhaps the final batch was skipped due to rounding problems. try ensuring that evaluate_n is divisible by batch_size*procs without a remainder, or try simplifying the multi-processing (i.e. single-node or single-GPU)."
            assert fid_obj.real_features_num_samples.item() == reals_quota, f"[torchmetrics FID] you requested --evaluate-n={reals_quota}, but we found {fid_obj.real_features_num_samples} samples. perhaps the final batch was skipped due to rounding problems. try ensuring that evaluate_n is divisible by batch_size*procs without a remainder, or try simplifying the multi-processing (i.e. single-node or single-GPU)."
        del fid_obj, tm_fid
        gc.collect()
    if accelerator.is_main_process:
        wants_inception_score: Set[str] = { 'inception-score-bicubic', 'inception-score-bilinear' }
        for extractor_name, features_by_subject in features.items():
            pred: FloatTensor = features_by_subject['pred']
            initial: Literal['I', 'C', 'D'] = extractor_name[0].upper()
            print(add_to_receipt(f'{extractor_name}, {pred.shape[0]} samples:'))
            if extractor_name in wants_inception_score:
                kl_divs: FloatTensor = pred.new_zeros((args.inception_score_splits, 1))
                # references:
                # https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
                # https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
                # https://github.com/pytorch/pytorch/commit/8e1ead8e4df3fa8acddd13c8b68d5c26690a9ec1?diff=split&w=0
                for logits_yx, kl_div_out in zip(torch.chunk(pred, args.inception_score_splits), kl_divs.unbind()):
                    # equivalent to:
                    #   p_y = logits_yx.softmax(-1).mean(0)
                    #   kl_div_: FloatTensor = kl_div(p_y.log(), logits_yx.log_softmax(-1), reduction='none', log_target=True).sum(-1).mean().exp()
                    # and hopefully more numerically stable.
                    # thanks to @ad8e for computing the equivalence
                    log_p_yx = logits_yx.log_softmax(-1)
                    log_p_y = log_p_yx.logsumexp(0) - math.log(log_p_yx.size(0))
                    kl_div_ = kl_div(log_p_y, log_p_yx, reduction='none', log_target=True).sum(-1).mean().exp()
                    kl_div_out.copy_(kl_div_)
                score_avg, score_std = kl_divs.mean(), kl_divs.std()
                print(add_to_receipt(f'  score: avg={score_avg.item():g} std={score_std.item():g}'))
            else:
                target: FloatTensor = features_by_subject['target']
                fid: FloatTensor = K.evaluation.fid(pred, target)
                kid: FloatTensor = K.evaluation.kid(pred, target)
                print(add_to_receipt(f'  F{initial}D: {fid.item():g}'))
                print(add_to_receipt(f'  K{initial}D: {kid.item():g}'))
                del target
            del pred
            gc.collect()

        if args.result_out_file is not None:
            print(f'Writing receipt to: {args.result_out_file}')
            makedirs(dirname(args.result_out_file), exist_ok=True)
            receipt: str = '\n'.join(receipt_lines)
            with open(args.result_out_file, 'w', encoding='utf-8') as f:
                f.write(receipt)

if __name__ == '__main__':
    main()
