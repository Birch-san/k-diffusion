import k_diffusion as K
from k_diffusion.utils import DataSetTransform, BatchData
from typing import Any, TypedDict, NotRequired, Union, Protocol, TypeAlias, Dict, Optional, List, Tuple, Sequence
from torch import Tensor
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision import transforms, datasets
from pathlib import Path
from functools import partial
from PIL import Image
from io import BytesIO
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from ..dataset_meta.get_class_captions import ClassCaptions
from .npz_dataset import NpzDataset

CustomDatasetConfig: TypeAlias = Dict[str, Any]

@dataclass
class _ImgFromSample:
    image_key: str
    tf: transforms.Compose
    def __call__(self, sample: Dict) -> Tensor:
        img_bytes: bytes = sample[self.image_key]
        with BytesIO(img_bytes) as stream:
            img: Image.Image = Image.open(stream)
            img.load()
        # type depends on what's in the `transforms.Compose`.
        # if it's KarrasAugmentationPipeline, then:
        # Tuple[Tensor, Tensor, Tensor]
        transformed_tensor: Any = self.tf(img)
        return transformed_tensor

@dataclass
class _MapAsciiClassCondWdsSample:
    class_cond_key: str
    img_from_sample: _ImgFromSample
    def __call__(self, sample: Dict) -> Tuple[Image.Image, int]:
        img: Any = self.img_from_sample(sample)
        class_cond = int(sample[self.class_cond_key])
        return (img, class_cond)

@dataclass
class _MapNumpyClassCondWdsSample:
    class_cond_key: str
    img_from_sample: _ImgFromSample
    def __call__(self, sample: Dict) -> Tuple[Image.Image, int]:
        img: Any = self.img_from_sample(sample)
        cls_bytes: bytes = sample[self.class_cond_key]
        with BytesIO(cls_bytes) as stream:
            class_cond_arr: NDArray = np.load(stream)
            class_cond: int = class_cond_arr.item()
        return (img, class_cond)

@dataclass
class _MapWdsSample:
    img_from_sample: _ImgFromSample
    def __call__(self, sample: Dict) -> Tuple[Image.Image]:
        img: Any = self.img_from_sample(sample)
        return (img,)

def _label_extractor(batch: BatchData) -> BatchData:
    return { 'label': batch['label'] }

def _class_ix_extractor(batch: BatchData) -> BatchData:
    return { 'class_ix': batch['label'] }

class ClassIxExtractorWithCanonicalization(DataSetTransform):
    dataset_label_to_canonical_label: Sequence[int]
    def __init__(self, dataset_label_to_canonical_label: Sequence[int]):
        self.dataset_label_to_canonical_label = dataset_label_to_canonical_label
    
    def __call__(self, batch: BatchData) -> BatchData:
        labels_nominal: List[int] = batch['label']
        labels_canonical: List[int] = [self.dataset_label_to_canonical_label[o] for o in labels_nominal]
        return { 'class_ix': labels_canonical }

class GetDataset(Protocol):
    @staticmethod
    def __call__(
        custom_dataset_config: CustomDatasetConfig,
        tf: Optional[transforms.Compose] = None,
    ) -> Union[Dataset, IterableDataset]: ...

class DatasetConfig(TypedDict):
    type: str
    location: NotRequired[str]
    image_key: NotRequired[str]
    wds_image_key: NotRequired[str]
    class_cond_key: NotRequired[str]
    get_dataset: NotRequired[GetDataset]

def get_dataset(
    dataset_config: DatasetConfig,
    config_dir: Path,
    uses_crossattn: bool,
    tf: transforms.Compose,
    class_captions: Optional[ClassCaptions] = None,
    shuffle_wds = True,
    # returning tuples probably causes a memory leak
    # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
    # but since we support batches including text-conditioning and karras aug conditioning: the trainer currently expects tuples
    output_tuples = True,
) -> Union[Dataset, IterableDataset]:
    if not output_tuples:
        assert dataset_config['type'] in ('wds', 'npz', 'imagefolder'), f"non-tuple data output not implemented for dataset type '{dataset_config['type']}'"
    if dataset_config['type'] == 'imagefolder':
        return K.utils.FolderOfImages(dataset_config['location'], transform=tf, output_tuples=output_tuples)
    if dataset_config['type'] == 'imagefolder-class':
        return datasets.ImageFolder(dataset_config['location'], transform=tf)
    if dataset_config['type'] == 'cifar10':
        return datasets.CIFAR10(dataset_config['location'], train=True, download=True, transform=tf)
    if dataset_config['type'] == 'mnist':
        return datasets.MNIST(dataset_config['location'], train=True, download=True, transform=tf)
    if dataset_config['type'] == 'huggingface':
        from datasets import load_dataset
        train_set = load_dataset(dataset_config['location'])
        ds_transforms: List[DataSetTransform] = []
        if class_captions is None:
            if uses_crossattn:
                ds_transforms.append(_label_extractor)
        else:
            if class_captions.dataset_label_to_canonical_label is None:
                class_ix_extractor: DataSetTransform = _class_ix_extractor
            else:
                class_ix_extractor: DataSetTransform = ClassIxExtractorWithCanonicalization(
                    class_captions.dataset_label_to_canonical_label
                )
            ds_transforms.append(class_ix_extractor)
        img_augs: DataSetTransform = partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key=dataset_config['image_key'])
        ds_transforms.append(img_augs)
        multi_transform: DataSetTransform = partial(K.utils.hf_datasets_multi_transform, transforms=ds_transforms)
        train_set.set_transform(multi_transform)
        return train_set['train']
    if dataset_config['type'] == 'wds' or dataset_config['type'] == 'wds-class':
        from webdataset import WebDataset, split_by_node
        img_from_sample = _ImgFromSample(
            image_key=dataset_config['wds_image_key'],
            tf=tf,
        )
        if dataset_config['type'] == 'wds':
            if output_tuples:
                mapper = _MapWdsSample(img_from_sample)
            else:
                mapper = _ImgFromSample(img_from_sample)
        elif dataset_config['type'] == 'wds-class':
            if dataset_config['class_cond_key'].endswith('.npy'):
                # legacy
                mapper = _MapNumpyClassCondWdsSample(
                    class_cond_key=dataset_config['class_cond_key'],
                    img_from_sample=img_from_sample,
                )
            else:
                # .txt, etc
                mapper = _MapAsciiClassCondWdsSample(
                    class_cond_key=dataset_config['class_cond_key'],
                    img_from_sample=img_from_sample,
                )
        else:
            raise ValueError('')
        wds = WebDataset(dataset_config['location'], nodesplitter=split_by_node).map(mapper)
        if shuffle_wds:
            wds = wds.shuffle(1000)
        return wds
    if dataset_config['type'] == 'npz':
        dataset = NpzDataset(
            dataset_config['location'],
            dataset_config['npz_image_key'],
            transform=tf,
            output_tuples=output_tuples,
            )
        return dataset
    if dataset_config['type'] == 'custom':
        import importlib.util
        location = (config_dir / dataset_config['location']).resolve()
        spec = importlib.util.spec_from_file_location('custom_dataset', location)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        get_dataset = getattr(module, dataset_config.get('get_dataset', 'get_dataset'))
        custom_dataset_config = dataset_config.get('config', {})
        return get_dataset(custom_dataset_config, transform=tf)
    raise ValueError('Invalid dataset type')