import argparse
from pathlib import Path
from typing import Dict, Any, Union, Callable
from torch import IntTensor, LongTensor, Tensor
from torch.utils import data
from torch.utils.data.dataset import Dataset, IterableDataset
from PIL import Image
from numpy.typing import NDArray
import numpy as np
from tqdm import tqdm

import k_diffusion as K
from kdiff_trainer.dataset.get_dataset import get_dataset

def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('--config', type=str, required=True,
                  help='configuration file detailing a dataset of predictions from a model')
  p.add_argument('--out', type=str, required=True, help='Where to save the .npz')
  p.add_argument('--mem-map-out', type=str, default=None, help='Memory-mapped .dat into which to buffer images')
  p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
  p.add_argument('--batch-size', type=int, default=64)
  p.add_argument('--limit', type=int, default=None)
  args = p.parse_args()
  
  config = K.config.load_config(args.config, use_json5=args.config.endswith('.jsonc'))
  model_config = config['model']
  assert len(model_config['input_size']) == 2
  size_h, size_w = model_config['input_size']

  dataset_config: Dict[str, Any] = config['dataset']
  if args.limit is None:
    sample_count: int = dataset_config['estimated_samples']
  else:
    sample_count: int = args.limit
  
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
  wrote_classes = False

  if args.mem_map_out is None:
    images = np.zeros((sample_count, size_h, size_w, 3), dtype=np.uint8)
  else:
    images = np.memmap(args.mem_map_out, shape=(sample_count, size_h, size_w, 3), mode='w+', dtype=np.uint8)
  classes = np.zeros((sample_count), dtype=np.int64)
  # we count samples instead of multiplying batch ix by batch size, because torch dataloader can give varying batch sizes
  # (at least for wds/IterableDataset)
  samples_written = 0
  with tqdm(
    desc=f'Exporting...',
    total=sample_count,
    unit='samp',
  ) as pbar:
    for batch in dl:
      img: IntTensor = batch[image_key]
      batch_img_count: int = img.size(0)
      samples_taken: int = min(batch_img_count, sample_count-samples_written)
      images[samples_written:samples_written+samples_taken] = img[:samples_taken]
      if len(batch) -1 >= class_key:
        wrote_classes = True
        cls: LongTensor = batch[class_key]
        assert cls.size(0) == batch_img_count
        classes[samples_written:samples_written+samples_taken] = cls[:samples_taken]
      samples_written += samples_taken
      pbar.update(samples_taken)
      if samples_written >= sample_count:
        break
  assert samples_written == sample_count
  if args.mem_map_out is not None:
    images.flush()
    # dunno if this is necessary
    images = np.memmap(args.mem_map_out, shape=(sample_count, size_h, size_w, 3), mode='r', dtype=np.uint8)
  arrs: Dict[str, Tensor] = { 'arr_0': images }
  if wrote_classes:
    arrs['arr_1'] = classes
  np.savez(args.out, **arrs)
  print(f'Wrote {samples_written} samples to {args.out}')

if __name__ == '__main__':
  main()