from torch import FloatTensor, ByteTensor
from PIL import Image
from typing import List
from numpy.typing import NDArray
from functorch.einops import rearrange

def to_pil_images(samples: FloatTensor) -> Image.Image:
    rgb_imgs: ByteTensor = samples.clamp(-1, 1).add(1).mul(127.5).byte()
    imgs_np: NDArray = rearrange(rgb_imgs, 'b rgb row col -> b row col rgb').contiguous().cpu().numpy()
    pils: List[Image.Image] = [Image.fromarray(img, mode='RGB') for img in imgs_np]
    return pils