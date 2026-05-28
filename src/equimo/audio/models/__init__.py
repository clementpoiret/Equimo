import importlib
import pkgutil
from pathlib import Path

from equimo.registry import get_model_cls as get_model_cls
from equimo.registry import register_model as register_model

__all__ = ["get_model_cls", "register_model"]

_pkg_path = str(Path(__file__).parent)
for _mod_info in pkgutil.iter_modules([_pkg_path]):
    if _mod_info.name in ("registry",):
        continue
    _mod = importlib.import_module(f".{_mod_info.name}", package=__name__)
    if hasattr(_mod, "__all__"):
        _exports = {k: getattr(_mod, k) for k in _mod.__all__}
        globals().update(_exports)
        __all__.extend(_exports)
