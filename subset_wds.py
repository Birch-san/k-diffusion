from webdataset import WebDataset, ShardWriter
from typing import TypedDict, Iterator, Dict
from tqdm import tqdm

Example = TypedDict('Example', {
  '__key__': str,
  '__url__': str,
  'latent.pth': bytes,
  'cls.txt': bytes,
})

SinkOutput = TypedDict('SinkOutput', {
  '__key__': str,
  'latent.pth': bytes,
  'cls.txt': bytes,
})

wds_in = WebDataset('/sdb/ml-data/imagenet-latents-512/{00000..00128}.tar')
it: Iterator[Example] = iter(wds_in)

out_root = '/sdb/ml-data/imagenet-latents-512-subset'
counts: Dict[int, int] = { cls: 0 for cls in range(100) }
with ShardWriter(f'{out_root}/%05d.tar', maxcount=10000) as sink:
  for item in tqdm(it, 'subsetting', total=1281167, unit='samp'):
    cls: int = int(str(item['cls.txt'], encoding='utf8'))
    if cls not in counts:
      continue
    count: int = counts[cls]
    out: SinkOutput = {
      '__key__': f'{cls}/{count}',
      'cls.txt': item['cls.txt'],
      'latent.pth': item['latent.pth'],
    }
    sink.write(out)
    new_count: int = count + 1
    if new_count > 100:
      del counts[cls]
      if not counts:
        break
    else:
      counts[cls] = new_count