from typing import TypedDict, NotRequired, Union, Dict, Tuple, Callable
import torch
from torch import Tensor, FloatTensor
from torch.utils.data.dataset import Dataset, IterableDataset
from PIL import Image
from io import BytesIO
from dataclasses import dataclass

@dataclass
class _LatentsFromSample:
    latent_key: str
    def __call__(self, sample: Dict) -> Tensor:
        latent_data: bytes = sample[self.latent_key]
        with BytesIO(latent_data) as stream:
            latents: FloatTensor = torch.load(stream, weights_only=True)
        return latents

@dataclass
class _MapClassCondWdsSample:
    class_cond_key: str
    latents_from_sample: _LatentsFromSample
    def __call__(self, sample: Dict) -> Tuple[Image.Image, int]:
        latents: FloatTensor = self.latents_from_sample(sample)
        class_bytes: bytes = sample[self.class_cond_key]
        class_str: str = class_bytes.decode('utf-8')
        class_cond = int(class_str)
        return (latents, class_cond)

@dataclass
class _MapWdsSample:
    latents_from_sample: _LatentsFromSample
    def __call__(self, sample: Dict) -> Tuple[Image.Image]:
        latents: FloatTensor = self.latents_from_sample(sample)
        return (latents,)

class DatasetConfig(TypedDict):
    type: str
    location: NotRequired[str]
    wds_latent_key: NotRequired[str]
    class_cond_key: NotRequired[str]

def get_latent_dataset(
    dataset_config: DatasetConfig,
) -> Union[Dataset, IterableDataset]:
    if dataset_config['type'] == 'wds' or dataset_config['type'] == 'wds-class':
        from webdataset import WebDataset, split_by_node
        latents_from_sample = _LatentsFromSample(
            latent_key=dataset_config['wds_latent_key'],
        )
        if dataset_config['type'] == 'wds':
            mapper = _MapWdsSample(latents_from_sample)
        elif dataset_config['type'] == 'wds-class':
            mapper = _MapClassCondWdsSample(
                class_cond_key=dataset_config['class_cond_key'],
                latents_from_sample=latents_from_sample,
            )
        else:
            raise ValueError('')
        return WebDataset(dataset_config['location'], nodesplitter=split_by_node).map(mapper).shuffle(1000)
    raise ValueError(f"Unsupported dataset type '{dataset_config['type']}'")