from PIL import Image
from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from torch.nn import Identity
from typing import Callable, Optional, Generic, TypeVar
from functools import cached_property
import numpy as np
from numpy.typing import NDArray

T = TypeVar('T')
Transform = Callable[[Image.Image], T]

@dataclass
class NpzDataset(Generic[T], Dataset[T]):
  """Recursively finds all images in a directory. It does not support
  classes/targets."""
  root: str
  image_key: str
  close_npy: Optional[Callable[[], None]]
  # returning tuples probably causes a memory leak
  # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
  # but since we support batches including text-conditioning and karras aug conditioning: the trainer currently expects tuples
  output_tuples: bool
  tf: Transform[T]

  def __init__(self, root: str, image_key: str, output_tuples=True, transform: Optional[Transform[T]]=None):
    super().__init__()
    self.root = root
    self.image_key = image_key
    self.transform = Identity() if transform is None else transform
    self.output_tuples=output_tuples
  
  @cached_property
  def arr(self) -> NDArray:
    with np.load(self.root) as npz_f:
      if self.image_key not in npz_f.files:
        raise ValueError(f"missing {self.image_key} in npz file")
      with npz_f.zip.open(f"{self.image_key}.npy", "r") as arr_f:
        version = np.lib.format.read_magic(arr_f)
        if version == (1, 0):
          header = np.lib.format.read_array_header_1_0(arr_f)
        elif version == (2, 0):
          header = np.lib.format.read_array_header_2_0(arr_f)
        else:
          raise ValueError(f".npy file had unsupported header version {version}.")
        shape, fortran, dtype = header
        assert not fortran and not dtype.hasobject, "Not sure what are the implications for fortran npy or npy with dtype.hasobject. Never seen one, so unable to test. Aborting out of caution. You could consider deleting this guard and seeing what happens."
        arr: NDArray = np.memmap(arr_f._fileobj._file, shape=shape, mode='r', dtype=np.uint8)
    def close_npy() -> None:
      # https://stackoverflow.com/a/6398543/5257399
      # for me, this crashes Python. even without any furher explicit access of arr. perhaps debugger is accessing it implicitly?
      arr._mmap.close()
      self.close_npy = None
    self.close_npy = close_npy
    return arr
  
  def dispose(self) -> None:
    """Release memory-mapped array if present. Might crash Python."""
    if self.close_npy is not None:
      self.close_npy()
    self.__dict__.pop('arr', None)

  def __repr__(self):
    return f'NpzDataset(root="{self.root}", len: {len(self)})'

  def __len__(self) -> int:
    return self.arr.shape[0]

  def __getitem__(self, ix: int) -> T:
    npz: NDArray = self.arr
    sample: NDArray = npz[ix]
    img = Image.fromarray(sample, 'RGB')
    transformed: T = self.transform(img)
    if self.output_tuples:
      return transformed,
    return transformed