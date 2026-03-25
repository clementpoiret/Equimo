import importlib
import pkgutil
from pathlib import Path

from .registry import get_model_cls, register_model

# Auto-import every submodule in this package and pull its __all__ into the
# namespace. Adding a new model file with a properly declared __all__ is all
# that is needed: no manual import here required.
_pkg_path = str(Path(__file__).parent)
for _mod_info in pkgutil.iter_modules([_pkg_path]):
    if _mod_info.name in ("registry",):
        continue
    _mod = importlib.import_module(f".{_mod_info.name}", package=__name__)
    if hasattr(_mod, "__all__"):
        globals().update({k: getattr(_mod, k) for k in _mod.__all__})
