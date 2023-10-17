from os import environ
from typing import Dict, Literal, TypeAlias, Set, TypedDict, NotRequired, Any, Callable
from tqdm.std import tqdm
from tqdm.utils import envwrap

TqdmKey: TypeAlias = Literal['total', 'ncols', 'miniters', 'position', 'nrows']
tqdm_keys: Set[TqdmKey] = {'total', 'ncols', 'miniters', 'position', 'nrows'}
tqdm_prefix = 'TQDM_'
prefix_len = len(tqdm_prefix)

class TqdmOverrides(TypedDict):
  total: NotRequired[float]
  ncols: NotRequired[int]
  miniters: NotRequired[float]
  position: NotRequired[int]
  nrows: NotRequired[int]

class tqdm_environ:
    preexisting_clashes_backup: Dict[TqdmKey, str]
    overrides: Dict[TqdmKey, str]
    tqdm_init_backup: Callable
    def __init__(self, overrides: TqdmOverrides) -> None:
        self.overrides = {f'{tqdm_prefix}{k}': str(v) for k, v in overrides.items()}

    def __enter__(self) -> None:
        self.preexisting_clashes_backup = {
            k: v for k, v in environ.items() if f'{tqdm_prefix}{k[prefix_len:].lower()}' in self.overrides.keys()
        }
        # we can't rely on the .update() to remove these via key-clash, because os.environ is case-sensitive whereas TQDM is not
        for key in self.preexisting_clashes_backup.keys():
            del environ[key]
        environ.update(self.overrides)
        wrap = envwrap("TQDM_", is_method=True, types={'total': float, 'ncols': int, 'miniters': float, 'position': int, 'nrows': int})
        self.tqdm_init_backup = tqdm.__init__
        tqdm.__init__ = wrap(tqdm.__init__)
    
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for key in self.overrides.keys():
            del environ[key]
        environ.update(self.preexisting_clashes_backup)
        tqdm.__init__ = self.tqdm_init_backup