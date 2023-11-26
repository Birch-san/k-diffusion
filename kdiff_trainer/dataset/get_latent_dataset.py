from typing import TypedDict, NotRequired, Union, Dict, NamedTuple
import torch
from torch import FloatTensor
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset, IterableDataset
from PIL import Image
from io import BytesIO

class LatentImgPair(NamedTuple):
    latents: FloatTensor
    # [0.0, 1.0]
    img_t: FloatTensor

class _MapWdsSample:
    to_tensor: ToTensor
    def __init__(self):
        self.to_tensor = ToTensor()

    def __call__(self, sample: Dict) -> LatentImgPair:
        latent_data: bytes = sample['latent.pth']
        with BytesIO(latent_data) as stream:
            latents: FloatTensor = torch.load(stream, weights_only=True)
        img_data: bytes = sample['img.png']
        with BytesIO(img_data) as stream:
            img: Image.Image = Image.open(stream)
            img_t: FloatTensor = self.to_tensor(img)
        return latents, img_t

class DatasetConfig(TypedDict):
    type: str
    location: NotRequired[str]
    wds_latent_key: NotRequired[str]

def get_latent_dataset(
    dataset_config: DatasetConfig,
) -> Union[Dataset, IterableDataset]:
    assert dataset_config['type'] == 'wds'
    mapper = _MapWdsSample()
    from webdataset import WebDataset, split_by_node
    return WebDataset(dataset_config['location'], nodesplitter=split_by_node).map(mapper).shuffle(1000)