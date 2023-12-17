import argparse
from pathlib import Path
from typing import Dict, Any, Union, Callable
from torch import IntTensor, LongTensor
from torch.utils import data
from torch.utils.data.dataset import Dataset, IterableDataset
from PIL import Image
from numpy.typing import NDArray
import numpy as np

import k_diffusion as K
from kdiff_trainer.dataset.get_dataset import get_dataset

def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('--config', type=str, required=True,
                  help='configuration file detailing a dataset of predictions from a model')
  p.add_argument('--out', type=str, required=True, help='Where to save the .npz')
  p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
  p.add_argument('--batch-size', type=int, default=64)
  args = p.parse_args()
  
  config = K.config.load_config(args.config, use_json5=args.config.endswith('.jsonc'))
  model_config = config['model']
  assert len(model_config['input_size']) == 2
  size_h, size_w = model_config['input_size']

  dataset_config: Dict[str, Any] = config['dataset']
  sample_count: int = dataset_config['estimated_samples']
  
  # note: np.asarray() is zero-copy. but the collation will probably copy. either way we are not planning any mutation.
  tf: Callable[[Image.Image], NDArray] = lambda pil: np.asarray(pil)
  dataset: Union[Dataset, IterableDataset] = get_dataset(
    dataset_config,
    config_dir=Path(args.config).parent,
    uses_crossattn=False,
    tf=tf,
    class_captions=None,
    shuffle_wds=False,
  )
  dl = data.DataLoader(
    dataset,
    args.batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=args.num_workers,
    persistent_workers=True,
    # we don't pin memory because we don't need GPU for this
    pin_memory=False,
  )

  image_key = dataset_config.get('image_key', 0)
  class_key = dataset_config.get('class_key', 1)

  images = np.zeros((sample_count, size_h, size_w, 3), dtype=np.uint8)
  classes = np.zeros((sample_count), dtype=np.int64)
  # we count samples instead of multiplying batch ix by batch size, because torch dataloader can give varying batch sizes
  # (at least for wds/IterableDataset)
  samples_written = 0
  for batch in dl:
    img: IntTensor = batch[image_key]
    batch_img_count: int = img.size(0)
    images[samples_written:samples_written+batch_img_count] = img
    if len(batch) -1 >= class_key:
      cls: LongTensor = batch[class_key]
      assert cls.size(0) == batch_img_count
      classes[samples_written:samples_written+cls.size(0)] = cls
    samples_written += batch_img_count
  assert samples_written == sample_count
  np.savez(args.out, arr_0=images, arr_1=class_key)
  print(f'Wrote {samples_written} samples to {args.out}')

if __name__ == '__main__':
  main()