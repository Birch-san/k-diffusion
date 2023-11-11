from typing import TypedDict, NotRequired, Union, Dict, Tuple, Callable, Generic, TypeVar
import torch
from torch import FloatTensor
from torch.utils.data.dataset import Dataset, IterableDataset
from io import BytesIO
from dataclasses import dataclass

Aug = TypeVar('Aug')
TensorMapper = Callable[[FloatTensor], Aug]

@dataclass
class _LatentsFromSample(Generic[Aug]):
    latent_key: str
    augment: TensorMapper[Aug]
    def __call__(self, sample: Dict) -> Aug:
        latent_data: bytes = sample[self.latent_key]
        with BytesIO(latent_data) as stream:
            latents: FloatTensor = torch.load(stream, weights_only=True)
        augmented_latents: Aug = self.augment(latents)
        return augmented_latents

@dataclass
class _MapClassCondWdsSample(Generic[Aug]):
    class_cond_key: str
    latents_from_sample: _LatentsFromSample[Aug]
    def __call__(self, sample: Dict) -> Tuple[Aug, int]:
        augmented_latents: Aug = self.latents_from_sample(sample)
        class_bytes: bytes = sample[self.class_cond_key]
        class_str: str = class_bytes.decode('utf-8')
        class_cond = int(class_str)
        return (augmented_latents, class_cond)

@dataclass
class _MapWdsSample(Generic[Aug]):
    latents_from_sample: _LatentsFromSample[Aug]
    def __call__(self, sample: Dict) -> Tuple[Aug]:
        latents: Aug = self.latents_from_sample(sample)
        return (latents,)

class DatasetConfig(TypedDict):
    type: str
    location: NotRequired[str]
    wds_latent_key: NotRequired[str]
    class_cond_key: NotRequired[str]

_identity: TensorMapper[FloatTensor] = lambda x: x

def get_latent_dataset(
    dataset_config: DatasetConfig,
    augment: TensorMapper[Aug] = _identity,
) -> Union[Dataset, IterableDataset]:
    if dataset_config['type'] == 'wds' or dataset_config['type'] == 'wds-class':
        from webdataset import WebDataset, split_by_node
        latents_from_sample = _LatentsFromSample[Aug](
            latent_key=dataset_config['wds_latent_key'],
            augment=augment,
        )
        if dataset_config['type'] == 'wds':
            mapper = _MapWdsSample[Aug](latents_from_sample)
        elif dataset_config['type'] == 'wds-class':
            mapper = _MapClassCondWdsSample[Aug](
                class_cond_key=dataset_config['class_cond_key'],
                latents_from_sample=latents_from_sample,
            )
        else:
            raise ValueError('')
        return WebDataset(dataset_config['location'], nodesplitter=split_by_node).map(mapper)#.shuffle(1000)
    raise ValueError(f"Unsupported dataset type '{dataset_config['type']}'")