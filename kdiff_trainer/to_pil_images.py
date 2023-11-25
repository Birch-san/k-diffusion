from torch import FloatTensor, ByteTensor
from PIL import Image
from typing import List
from numpy.typing import NDArray
from functorch.einops import rearrange

def byte_tensor_to_pils(rgb_imgs: ByteTensor) -> List[Image.Image]:
    """
    Args:
        rgb_imgs `ByteTensor`: 0 to 255 (batch, channels, height, width)
    """
    imgs_np: NDArray = rearrange(rgb_imgs, 'b rgb row col -> b row col rgb').contiguous().cpu().numpy()
    pils: List[Image.Image] = [Image.fromarray(img, mode='RGB') for img in imgs_np]
    return pils

def to_pil_images(samples: FloatTensor) -> List[Image.Image]:
    rgb_imgs: ByteTensor = samples.clamp(-1, 1).add(1).mul(127.5).byte()
    pils: List[Image.Image] = byte_tensor_to_pils(rgb_imgs)
    return pils

def to_pil_images_from_0_1(samples: FloatTensor) -> List[Image.Image]:
    rgb_imgs: ByteTensor = samples.mul(255).byte()
    pils: List[Image.Image] = byte_tensor_to_pils(rgb_imgs)
    return pils