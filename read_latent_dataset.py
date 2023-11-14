import torch
from torch import inference_mode, FloatTensor, Tensor
from torch.utils import data
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision.utils import save_image
from typing import Union, Dict, Any, List, Iterator
from diffusers import AutoencoderKL
from diffusers.models.autoencoder_kl import DecoderOutput
from tqdm import tqdm
from os import makedirs

import k_diffusion as K
from kdiff_trainer.dataset.get_latent_dataset import get_latent_dataset
from kdiff_trainer.vae.attn.null_attn_processor import NullAttnProcessor
from kdiff_trainer.vae.attn.natten_attn_processor import NattenAttnProcessor
from kdiff_trainer.vae.attn.qkv_fusion import fuse_vae_qkv

def main():
    # config_path = 'configs/dataset/latent-test.jsonc'
    config_path = 'configs/dataset/imagenet-100-latent.mahouko.jsonc'
    config = K.config.load_config(config_path, use_json5=config_path.endswith('.jsonc'))
    dataset_config = config['dataset']
    train_set: Union[Dataset, IterableDataset] = get_latent_dataset(dataset_config)
    use_ollin_vae = False
    vae_kwargs: Dict[str, Any] = {
        'torch_dtype': torch.float16,
    } if use_ollin_vae else {
        'subfolder': 'vae',
        'torch_dtype': torch.bfloat16,
    }
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        'madebyollin/sdxl-vae-fp16-fix' if use_ollin_vae else 'stabilityai/stable-diffusion-xl-base-0.9',
        use_safetensors=True,
        **vae_kwargs,
    )

    vae_attn_impl = 'original'
    if vae_attn_impl == 'natten':
        fuse_vae_qkv(vae)
        # NATTEN seems to output identical output to global self-attention at kernel size 17
        # even kernel size 3 looks good (not identical, but very close).
        # I haven't checked what's the smallest kernel size that can look identical. 15 looked good too.
        # seems to speed up encoding of 1024x1024px images by 11%
        vae.set_attn_processor(NattenAttnProcessor(kernel_size=17))
    elif vae_attn_impl == 'null':
        for attn in [*vae.encoder.mid_block.attentions, *vae.decoder.mid_block.attentions]:
            # you won't be needing these
            del attn.to_q, attn.to_k
        # doesn't mix information between tokens via QK similarity. just projects every token by V and O weights.
        # looks alright, but is by no means identical to global self-attn.
        vae.set_attn_processor(NullAttnProcessor())
    elif vae_attn_impl == 'original':
        # leave it as global self-attention
        pass
    else:
        raise ValueError(f"Never heard of --vae-attn-impl '{vae_attn_impl}'")

    del vae.encoder
    device = torch.device('cuda')
    vae.to(device).eval()

    # shuffle=not isinstance(train_set, data.IterableDataset)
    shuffle=False
    batch_size=16
    train_dl = data.DataLoader(train_set, batch_size, shuffle=shuffle, drop_last=False,
                               num_workers=8, persistent_workers=True, pin_memory=True)

    out_root = 'out/imagenet100-validate'
    makedirs(out_root, exist_ok=True)
    class_sample_count: Dict[int, int] = {}
    it: Iterator[List[Tensor]] = iter(train_dl)
    for batch in tqdm(it, desc='exporting', total=10000//batch_size, unit='batch'):
        assert isinstance(batch, list)
        assert len(batch) == 2, "batch item was not the expected length of 2. perhaps class labels are missing. use dataset type imagefolder-class or wds-class, to get class labels."
        latents, classes = batch
        latents = latents.to(device, vae.dtype)
        with inference_mode():
            decoded: DecoderOutput = vae.decode(latents)
        del latents
        # note: if you wanted to _train_ on these latents, you would want to scale-and-shift them after this
        rgb: FloatTensor = decoded.sample
        del decoded
        rgb: FloatTensor = rgb.div(2).add_(.5).clamp_(0,1)
        for sample, cls in zip(rgb.unbind(), classes.unbind()):
            cls_: int = cls.item()
            out_dir = f'{out_root}/{cls_}'
            if cls_ not in class_sample_count:
                makedirs(out_dir, exist_ok=True)
                class_sample_count[cls_] = 0
            sample_ix: int = class_sample_count[cls_]
            save_image(sample, f'{out_dir}/{sample_ix}.png')
            class_sample_count[cls_] += 1
        del sample, cls
    
    print('class sample count:', class_sample_count)

if __name__ == '__main__':
    main()