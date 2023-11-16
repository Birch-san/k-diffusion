from webdataset import ShardWriter
from typing import TypedDict, Dict, List#, Iterator, Union
from tqdm import tqdm
from os import makedirs
from torch.utils.data import DataLoader#, Dataset, IterableDataset
# from pathlib import Path
from torchvision import transforms#, datasets
# from torch import Tensor
from PIL import Image

import k_diffusion as K
from k_diffusion.utils import FolderOfImages
from kdiff_trainer.to_pil_images import to_pil_images_from_0_1
# from kdiff_trainer.dataset.get_dataset import get_dataset

ClassCondSinkOutput = TypedDict('ClassCondSinkOutput', {
  '__key__': str,
  'img.png': Image.Image,
  'cls.txt': str,
})

config_path = 'configs/dataset/imagenet.mahouko.jsonc'
config = K.config.load_config(config_path, use_json5=config_path.endswith('.jsonc'))
dataset_config = config['dataset']

size_px = 64

tf = transforms.Compose([
  transforms.Resize(size_px, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
  transforms.CenterCrop(size_px),
  transforms.ToTensor(),
])

# dataset: Union[Dataset, IterableDataset] = get_dataset(
#   dataset_config,
#   config_dir=Path(config_path).parent,
#   uses_crossattn=False,
#   tf=tf,
#   class_captions=None,
# )
# geese = datasets.ImageFolder(dataset_config['location'], is_valid_file=lambda file: Path(file).parent.name == 'n01855672', transform=tf)
geese = FolderOfImages(f"{dataset_config['location']}/n01855672", transform=tf)

batch_size=16
train_dl = DataLoader(geese, batch_size, shuffle=False, drop_last=False,
                            num_workers=8, persistent_workers=True, pin_memory=True)

# it: Iterator[List[Tensor]] = iter(train_dl)

out_root = f'/sdb/ml-data/imagenet-{size_px}-geese'
makedirs(out_root, exist_ok=True)

cls = 99
# counts: Dict[int, int] = { cls: 0 for cls in range(100) }
counts: Dict[int, int] = { 99: 0 }
with ShardWriter(f'{out_root}/%05d.tar', maxcount=10000) as sink:
  for batch in tqdm(train_dl, 'subsetting', total=1281167, unit='samp'):
    rgbs, *_ = batch
    pils: List[Image.Image] = to_pil_images_from_0_1(rgbs)
    del rgbs
    for pil in pils:
      count: int = counts[cls]
      out: ClassCondSinkOutput = {
        '__key__': f'{cls}/{count}',
        'cls.txt': str(cls),
        'img.png': pil,
      }
      sink.write(out)
      new_count: int = count + 1
      if new_count > 1000:
        del counts[cls]
        if not counts:
          break
      else:
        counts[cls] = new_count
    if not counts:
      break