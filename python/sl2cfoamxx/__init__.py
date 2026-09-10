# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

"""sl2cfoam++ vertex amplitudes and tensors through the public C++ library.

Quantum labels are doubled integers; all angles are in radians. Computations
release the GIL. Configure shared defaults before launching concurrent work,
then select backends and progress callbacks per call.
"""

import sys as _sys

from . import _sl2cfoamxx as _native

if _sys.platform.startswith("linux") and "mkl" in _native.build_info().blas_provider.lower():
    import ctypes as _ctypes
    import os as _os

    # MKL loads CPU-specific kernels lazily. Its separately loaded kernels
    # need the linked MKL libraries in the global symbol scope, whereas Python
    # normally loads extensions locally. Promote the already loaded module
    # and its dependencies without changing Python's flags for other imports.
    _mkl_dependency_handle = _ctypes.CDLL(
        _native.__file__, mode=_os.RTLD_NOLOAD | _ctypes.RTLD_GLOBAL)
    del _ctypes, _os

from ._sl2cfoamxx import *
from ._sl2cfoamxx import __version__

del _native, _sys
