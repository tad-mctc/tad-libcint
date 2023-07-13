import os
import sys
import ctypes
import ctypes.util
from typing import Callable, Dict, Any

__all__ = ["CINT", "CGTO"]


_libs: Dict[str, Any] = {}


def _library_loader(name: str, relpath: str) -> Callable:
    curpath = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(curpath, relpath))

    # load the library and cache the handler
    def fcn():
        if name not in _libs:
            try:
                _libs[name] = ctypes.cdll.LoadLibrary(path)
            except OSError as e:
                path2 = ctypes.util.find_library(name)
                if path2 is None:
                    raise e
                _libs[name] = ctypes.cdll.LoadLibrary(path2)
        return _libs[name]

    return fcn


# libraries
_ext = "dylib" if sys.platform == "darwin" else "so"

_libcint_relpath = f"libcint.{_ext}"
CINT = _library_loader("cint", _libcint_relpath)

_libcgto_relpath = f"libcgto.{_ext}"
CGTO = _library_loader("cgto", _libcgto_relpath)

# _libcpbc_relpath = f"libpbc.{_ext}"
# CPBC = _library_loader("cpbc", _libcpbc_relpath)
# _libcsymm_relpath = f"libsymm.{_ext}"
# CSYMM = _library_loader("symm", _libcsymm_relpath)
# _libcvhf_relpath = f"libcvhf.{_ext}"
# CVHF = _library_loader("CVHF", _libcvhf_relpath)
