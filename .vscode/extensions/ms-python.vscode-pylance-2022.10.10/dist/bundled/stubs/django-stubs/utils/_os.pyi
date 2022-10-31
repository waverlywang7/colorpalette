from os.path import abspath
from pathlib import Path
from typing import Any, Union

abspathu = abspath

def upath(path: Any) -> Any: ...
def npath(path: Any) -> Any: ...
def safe_join(base: Union[bytes, str], *paths: Any) -> str: ...
def symlinks_supported() -> Any: ...
def to_path(value: Union[Path, str]) -> Path: ...
