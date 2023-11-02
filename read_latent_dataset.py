import torch
from torch import inference_mode, FloatTensor, Tensor
from torch.utils import data
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision.utils import save_image
from typing import Union, Dict, Any, List, Iterator
from diffusers import AutoencoderKL
from diffusers.models.autoencoder_kl import DecoderOutput
from tqdm import tqdm

import k_diffusion as K
from kdiff_trainer.dataset.get_latent_dataset import get_latent_dataset
from kdiff_trainer.vae.attn.null_attn_processor import NullAttnProcessor
from kdiff_trainer.vae.attn.natten_attn_processor import NattenAttnProcessor
from kdiff_trainer.vae.attn.qkv_fusion import fuse_vae_qkv


def main():
    config_path = 'configs/dataset/latent-test.jsonc'
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

    train_dl = data.DataLoader(train_set, 2, shuffle=not isinstance(train_set, data.IterableDataset), drop_last=False,
                               num_workers=2, persistent_workers=True, pin_memory=True)

    it: Iterator[List[Tensor]] = iter(train_dl)
    for batch_ix, batch in enumerate(tqdm(it)):
        assert isinstance(batch, list)
        assert len(batch) == 2, "batch item was not the expected length of 2. perhaps class labels are missing. use dataset type imagefolder-class or wds-class, to get class labels."
        latents, classes = batch
        latents = latents.to(device, vae.dtype)
        with inference_mode():
            decoded: DecoderOutput = vae.decode(latents)
        # note: if you wanted to _train_ on these latents, you would want to scale-and-shift them after this
        rgb: FloatTensor = decoded.sample
        rgb: FloatTensor = rgb.div(2).add_(.5).clamp_(0,1)
        print('batch', batch_ix, 'classes:', classes.tolist())
        save_image(rgb, f'out/vae-decode-test/batch.{batch_ix}.jpg')

if __name__ == '__main__':
    main()