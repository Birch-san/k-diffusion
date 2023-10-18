from functools import partial
import json
import math
from pathlib import Path
from typing import Dict, Union

from jsonmerge import merge

from . import augmentation, layers, models, utils


def round_to_power_of_two(x, tol):
    approxs = []
    for i in range(math.ceil(math.log2(x))):
        mult = 2**i
        approxs.append(round(x / mult) * mult)
    for approx in reversed(approxs):
        error = abs((approx - x) / x)
        if error <= tol:
            return approx
    return approxs[0]


def load_config(path_or_dict: Union[str, Dict], use_json5=False):
    defaults_image_v1 = {
        'model': {
            'patch_size': 1,
            'augment_wrapper': True,
            'mapping_cond_dim': 0,
            'unet_cond_dim': 0,
            'cross_cond_dim': 0,
            'cross_attn_depths': None,
            'skip_stages': 0,
            'has_variance': False,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'betas': [0.95, 0.999],
            'eps': 1e-6,
            'weight_decay': 1e-3,
        },
    }
    defaults_image_transformer_v1 = {
        'model': {
            'd_ff': 0,
            'augment_wrapper': False,
            'skip_stages': 0,
            'has_variance': False,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 5e-4,
            'betas': [0.9, 0.99],
            'eps': 1e-8,
            'weight_decay': 1e-4,
        },
    }
    defaults_image_transformer_v2 = {
        'model': {
            'mapping_width': 256,
            'mapping_depth': 2,
            'mapping_d_ff': None,
            'mapping_cond_dim': 0,
            'd_ffs': None,
            'self_attns': None,
            'augment_wrapper': False,
            'skip_stages': 0,
            'has_variance': False,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 5e-4,
            'betas': [0.9, 0.99],
            'eps': 1e-8,
            'weight_decay': 1e-4,
        },
    }
    defaults = {
        'model': {
            'sigma_data': 1.,
            'dropout_rate': 0.,
            'augment_prob': 0.,
            'loss_config': 'karras',
            'loss_weighting': 'karras',
            'loss_scales': 1,
        },
        'dataset': {
            'type': 'imagefolder',
            'num_classes': 0,
            'cond_dropout_rate': 0.1,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'weight_decay': 1e-4,
        },
        'lr_sched': {
            'type': 'constant',
            'warmup': 0.,
        },
        'ema_sched': {
            'type': 'inverse',
            'power': 0.6667,
            'max_value': 0.9999
        },
    }
    if not isinstance(path_or_dict, dict):
        file = Path(path_or_dict)
        if file.suffix == '.safetensors':
            metadata = utils.get_safetensors_metadata(file)
            json_str: str = metadata['config']
        else:
            json_str: str = file.read_text()
        if use_json5:
            # json5 supports comments, like in .jsonc files
            import json5
            config: Dict = json5.loads(json_str)
        else:
            config: Dict = json.loads(json_str)
    else:
        config: Dict = path_or_dict
    if config['model']['type'] == 'image_v1':
        config = merge(defaults_image_v1, config)
    elif config['model']['type'] == 'image_transformer_v1':
        config = merge(defaults_image_transformer_v1, config)
        if not config['model']['d_ff']:
            config['model']['d_ff'] = round_to_power_of_two(config['model']['width'] * 8 / 3, tol=0.05)
    elif config['model']['type'] == 'image_transformer_v2':
        config = merge(defaults_image_transformer_v2, config)
        if not config['model']['mapping_d_ff']:
            config['model']['mapping_d_ff'] = config['model']['mapping_width'] * 3
        if not config['model']['d_ffs']:
            d_ffs = []
            for width in config['model']['widths']:
                d_ffs.append(width * 3)
            config['model']['d_ffs'] = d_ffs
        if not config['model']['self_attns']:
            self_attns = []
            default_neighborhood = {"type": "neighborhood", "d_head": 64, "kernel_size": 7}
            default_global = {"type": "global", "d_head": 64}
            for i in range(len(config['model']['widths'])):
                self_attns.append(default_neighborhood if i < len(config['model']['widths']) - 1 else default_global)
            config['model']['self_attns'] = self_attns
    return merge(defaults, config)


def make_model(config):
    dataset_config = config['dataset']
    num_classes = dataset_config['num_classes']
    config = config['model']
    if config['type'] == 'image_v1':
        model = models.ImageDenoiserModelV1(
            config['input_channels'],
            config['mapping_out'],
            config['depths'],
            config['channels'],
            config['self_attn_depths'],
            config['cross_attn_depths'],
            patch_size=config['patch_size'],
            dropout_rate=config['dropout_rate'],
            mapping_cond_dim=config['mapping_cond_dim'] + (9 if config['augment_wrapper'] else 0),
            unet_cond_dim=config['unet_cond_dim'],
            cross_cond_dim=config['cross_cond_dim'],
            skip_stages=config['skip_stages'],
            has_variance=config['has_variance'],
        )
        if config['augment_wrapper']:
            model = augmentation.KarrasAugmentWrapper(model)
    elif config['type'] == 'image_transformer_v1':
        model = models.ImageTransformerDenoiserModelV1(
            n_layers=config['depth'],
            d_model=config['width'],
            d_ff=config['d_ff'],
            in_features=config['input_channels'],
            out_features=config['input_channels'],
            patch_size=config['patch_size'],
            num_classes=num_classes + 1 if num_classes else 0,
            dropout=config['dropout_rate'],
            sigma_data=config['sigma_data'],
        )
    elif config['type'] == 'image_transformer_v2':
        assert len(config['widths']) == len(config['depths'])
        assert len(config['widths']) == len(config['d_ffs'])
        assert len(config['widths']) == len(config['self_attns'])
        cross_attn = config['cross_attn'] if 'cross_attn' in config else None
        if cross_attn is None:
            d_cross = xattn_scale_qk = xattn_dropout = None
        else:
            xattn_scale_qk=cross_attn.get('scale_qk', True)
            xattn_dropout=cross_attn.get('dropout', .1)
            match(cross_attn['encoder']):
                case 'clip-vit-l':
                    d_cross = 768
                case 'phi-1-5':
                    d_cross = 2048
                case _:
                    raise ValueError(f"Never heard of cross-attn encoder '{cross_attn['encoder']}'")
        cross_attn_layers = [None]*len(config['widths']) if cross_attn is None else cross_attn['layers']
        levels = []
        for depth, width, d_ff, self_attn, cross_attn_layer in zip(config['depths'], config['widths'], config['d_ffs'], config['self_attns'], cross_attn_layers):
            if self_attn['type'] == 'global':
                self_attn = models.image_transformer_v2.GlobalAttentionSpec(self_attn.get('d_head', 64))
            elif self_attn['type'] == 'neighborhood':
                self_attn = models.image_transformer_v2.NeighborhoodAttentionSpec(self_attn.get('d_head', 64), self_attn.get('kernel_size', 7))
            elif self_attn['type'] == 'shifted-window':
                self_attn = models.image_transformer_v2.ShiftedWindowAttentionSpec(self_attn.get('d_head', 64), self_attn['window_size'])
            elif self_attn['type'] == 'none':
                self_attn = models.image_transformer_v2.NoAttentionSpec()
            else:
                raise ValueError(f'unsupported self attention type {self_attn["type"]}')
            if cross_attn_layer is not None:
                cross_attn_spec = models.image_transformer_v2.CrossAttentionSpec(
                    d_head=cross_attn_layer.get('d_head', 64),
                    d_cross=d_cross,
                    scale_qk=xattn_scale_qk,
                    dropout=xattn_dropout,
                )
            levels.append(models.image_transformer_v2.LevelSpec(depth, width, d_ff, self_attn, cross_attn_spec))
        mapping = models.image_transformer_v2.MappingSpec(config['mapping_depth'], config['mapping_width'], config['mapping_d_ff'])
        model = models.ImageTransformerDenoiserModelV2(
            levels=levels,
            mapping=mapping,
            in_channels=config['input_channels'],
            out_channels=config['input_channels'],
            patch_size=config['patch_size'],
            num_classes=num_classes + 1 if num_classes else 0,
            mapping_cond_dim=config['mapping_cond_dim'],
            dropout=config['dropout_rate'],
        )
    else:
        raise ValueError(f'unsupported model type {config["type"]}')
    return model


def make_denoiser_wrapper(config):
    config = config['model']
    sigma_data = config.get('sigma_data', 1.)
    has_variance = config.get('has_variance', False)
    loss_config = config.get('loss_config', 'karras')
    if loss_config == 'karras':
        weighting = config.get('loss_weighting', 'karras')
        scales = config.get('loss_scales', 1)
        if not has_variance:
            return partial(layers.Denoiser, sigma_data=sigma_data, weighting=weighting, scales=scales)
        return partial(layers.DenoiserWithVariance, sigma_data=sigma_data, weighting=weighting)
    if loss_config == 'simple':
        if has_variance:
            raise ValueError('Simple loss config does not support a variance output')
        return partial(layers.SimpleLossDenoiser, sigma_data=sigma_data)
    raise ValueError('Unknown loss config type')


def make_sample_density(config):
    sd_config = config['sigma_sample_density']
    sigma_data = config['sigma_data']
    if sd_config['type'] == 'lognormal':
        loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
        scale = sd_config['std'] if 'std' in sd_config else sd_config['scale']
        return partial(utils.rand_log_normal, loc=loc, scale=scale)
    if sd_config['type'] == 'loglogistic':
        loc = sd_config['loc'] if 'loc' in sd_config else math.log(sigma_data)
        scale = sd_config['scale'] if 'scale' in sd_config else 0.5
        min_value = sd_config['min_value'] if 'min_value' in sd_config else 0.
        max_value = sd_config['max_value'] if 'max_value' in sd_config else float('inf')
        return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
    if sd_config['type'] == 'loguniform':
        min_value = sd_config['min_value'] if 'min_value' in sd_config else config['sigma_min']
        max_value = sd_config['max_value'] if 'max_value' in sd_config else config['sigma_max']
        return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)
    if sd_config['type'] in {'v-diffusion', 'cosine'}:
        min_value = sd_config['min_value'] if 'min_value' in sd_config else 1e-3
        max_value = sd_config['max_value'] if 'max_value' in sd_config else 1e3
        return partial(utils.rand_v_diffusion, sigma_data=sigma_data, min_value=min_value, max_value=max_value)
    if sd_config['type'] == 'split-lognormal':
        loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
        scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
        scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
        return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
    if sd_config['type'] == 'cosine-interpolated':
        min_value = sd_config.get('min_value', min(config['sigma_min'], 1e-3))
        max_value = sd_config.get('max_value', max(config['sigma_max'], 1e3))
        image_d = sd_config.get('image_d', max(config['input_size']))
        noise_d_low = sd_config.get('noise_d_low', 32)
        noise_d_high = sd_config.get('noise_d_high', max(config['input_size']))
        return partial(utils.rand_cosine_interpolated, image_d=image_d, noise_d_low=noise_d_low, noise_d_high=noise_d_high, sigma_data=sigma_data, min_value=min_value, max_value=max_value)

    raise ValueError('Unknown sample density type')
