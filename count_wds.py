from webdataset import WebDataset
from typing import TypedDict, Iterable

Example = TypedDict('Example', {
  '__key__': str,
  '__url__': str,
  'img.png': bytes,
})

dataset = WebDataset('/sdb/ml-data/openai-guided-diffusion-256-classcond-unguided-samples-50k/{00000..00004}.tar')

count = 0
it: Iterable[Example] = iter(dataset)
for ix, item in enumerate(it):
  count += 1
print(count)
pass