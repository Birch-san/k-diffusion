from PIL import Image
from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from torch.nn import Identity
from typing import Callable, Optional, Any
from functools import cached_property
import numpy as np
from numpy.typing import NDArray
from .npz_reader import open_npz_array

Transform = Callable[[Image.Image], Any]

@dataclass
class NpzDataset(Dataset):
  """Recursively finds all images in a directory. It does not support
  classes/targets."""
  root: str
  image_key: str
  close_npz: Optional[Callable[[], None]]
  tf: Transform

  def __init__(self, root: str, image_key: str, transform: Optional[Transform]=None):
    super().__init__()
    self.root = root
    self.image_key = image_key
    self.transform = Identity() if transform is None else transform
  
  @cached_property
  def arr(self) -> NDArray:
    ctx = open_npz_array(self.root, self.image_key)
    arr: NDArray = ctx.__enter__()
    def close_npz() -> None:
      ctx.__exit__()
      self.__dict__.pop('arr', None)
    self.close_npz = close_npz
    return arr
  
  def dispose(self) -> None:
    if self.close_npz is not None:
      self.close_npz()

  def __repr__(self):
    return f'NpzDataset(root="{self.root}", len: {len(self)})'

  def __len__(self) -> int:
    # TODO
    # return len(self.paths)
    return 4

  def __getitem__(self, key: int) -> Any:
    npz: NDArray = self.arr
    raise RuntimeError("Haven't finished implementing")
    # path = self.paths[key]
    # with open(path, 'rb') as f:
    #   image = Image.open(f).convert('RGB')
    # image = self.transform(image)
    # return image