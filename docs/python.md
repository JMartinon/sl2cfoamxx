<!-- SPDX-FileCopyrightText: 2026 Jeremy Martinon -->
<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Python API

The `sl2cfoamxx` package wraps the installed public C++ API. Numerical work runs
in the same shared library as C++ applications, including its CPU and optional
CUDA backends. No numerical algorithms are reimplemented in Python.

## How Python loads the library

A Python source file is a script, for example `calculation.py`. It does not
have a separate compile/link step. At its beginning, write:

```python
import sl2cfoamxx as sf
```

CMake has already compiled and linked the Python extension to the C++ shared
library. When Python executes `import`, it finds the `sl2cfoamxx` package,
loads its compiled `_sl2cfoamxx` extension, and the operating system loads the
linked C++ library. CMake supplies the library search path. You do not add
`-I`, `-L`, or `-lsl2cfoamxx` flags to each Python script.

Python must be able to **find the package**, and its version must match the
one used to build the extension. There are two ways to arrange this:

1. Use the build directly by putting `build-python/python` on `PYTHONPATH`.
   This variable tells Python which additional directories to search.
2. Install into a Python virtual environment. A virtual environment is a
   project-local Python interpreter and package directory. After installation,
   that interpreter finds the package automatically, without `PYTHONPATH`.

For a build configured with your `python3` interpreter, run:

```sh
export PYTHONPATH="$PWD/build-python/python"
python3 examples/python_example.py
python3 /path/to/calculation.py
```

Use the same Python major/minor version that CMake selected. The export applies
to this shell and its child processes; repeat it in a new terminal. Installing
into a virtual environment, as described below, avoids that shell setup. Do not
name your script `sl2cfoamxx.py`, since it would hide the package of the same name.

## Build and install

Use Python 3.10 or newer, its development headers, and the same GNU C++/OpenMP
and BLAS dependencies as the [C++ build](building.md). NumPy is needed
for array results and the Python tests. Set `Python_EXECUTABLE` to the Python
interpreter that will import the extension, including a virtual environment's
interpreter if applicable:

```sh
python3 -m pip install numpy
cmake -S . -B build-python \
  -DCMAKE_CXX_COMPILER=/path/to/g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DSL2CFOAMXX_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$(command -v python3)" \
  -DBUILD_TESTING=OFF
cmake --build build-python --parallel
PYTHONPATH="$PWD/build-python/python" python3 examples/python_example.py
```

The single [`python_example.py`](../examples/python_example.py) covers fixed
and coherent amplitudes, full tensors, complete Euclidean boundary angles,
closure, coefficient generation, NumPy arrays, persistence, and CPU/GPU
progress. Add `--einsum` for optional coherent and tensor/tensor contractions.
Its default half-integer coherent amplitude is checked against the retained
Julia reference. Run it with the same `PYTHONPATH` and interpreter.

GPU builds add `-DENABLE_CUDA=ON`, the appropriate CUDA architectures, and
`CUTENSOR_ROOT` as described in the [build guide](building.md#nvidia-gpus). An
extension built with CUDA still supports explicit CPU calls.

CMake **downloads nanobind 2.13.0 into `third-party/nanobind`**, verifies the
archive's SHA-256, and builds it. The pinned upstream source distribution
includes robin_map. There is no pip-installed nanobind prerequisite. Subsequent
configurations reuse that source tree without network access. Offline builds
may set `SL2CFOAMXX_NANOBIND_SOURCE_DIR` to an existing matching source tree with
its `ext/robin_map` dependency. The download and extracted sources are ignored
by Git. Python support defaults to `OFF`, so C++ builds do not download it or
require Python development headers.

To install both the extension and C++ library into a prefix:

```sh
cmake --install build-python --prefix "$PWD/build-python/install"
# Replace 3.13 with the Python major.minor version selected at configuration.
PYTHONPATH="$PWD/build-python/install/lib/python3.13/site-packages" \
  python3 -c 'import sl2cfoamxx; print(sl2cfoamxx.version())'
```

The default package location is
`<prefix>/<libdir>/pythonX.Y/site-packages/sl2cfoamxx`. Override it with the
prefix-relative `SL2CFOAMXX_PYTHON_INSTALL_DIR` CMake setting if needed. A
relative runtime search path locates the C++ library under the same prefix;
move the entire prefix together. The compiler/OpenMP, BLAS, and optional CUDA
runtime libraries still need to be available on the destination machine.
`BUILD_TESTING=OFF` supports production-only builds and installs as well.

### Install into a virtual environment

The following is a complete one-time CPU setup. Run it from the repository
root, substituting your GNU C++ compiler path. `python3` here selects the
Python version for the new environment and extension:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy
cmake -S . -B build-python-venv \
  -DCMAKE_CXX_COMPILER=/path/to/g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_CUDA=OFF -DBUILD_TESTING=OFF \
  -DSL2CFOAMXX_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_INSTALL_PREFIX="$PWD/.venv" \
  -DCMAKE_INSTALL_LIBDIR=lib
cmake --build build-python-venv --parallel
cmake --install build-python-venv
.venv/bin/python -c 'import sl2cfoamxx; print(sl2cfoamxx.version())'
.venv/bin/python examples/python_example.py
```

Each later script only needs `import sl2cfoamxx`. You may run it using the
environment's interpreter directly, or activate the environment in your
terminal:

```sh
source .venv/bin/activate
python /path/to/calculation.py
```

Activation makes `python` refer to that environment until you run `deactivate`
or close the shell. It does not compile or link anything. An editor or notebook
must likewise use this environment's Python interpreter/kernel. Rebuilding is
needed when you change the native library or Python version, not when you edit
your calculation script. A CUDA installation follows the same steps with
`ENABLE_CUDA=ON` and the CUDA/cuTENSOR settings from the C++ build guide.

## Configure and compute

Names and numerical conventions match C++. Spins are ten **doubled integers**
in `(j12,j13,j14,j15,j23,j24,j25,j34,j35,j45)` order. Intertwiners are five
doubled integers in canonical axis order. A label of `1` represents physical
spin `1/2`. Lists, tuples, and NumPy arrays are accepted for fixed-size inputs.

```python
import sl2cfoamxx as sf

config = sf.default_config()
config.backend = sf.Backend.CPU
config.cpu_threads = 4
config.tensors.directory = "./tensor-cache"
config.tensors.storage = sf.TensorStorage.in_memory_sym
config.coherent_amplitude.pruning_ratio = 0.0
sf.configure(config)

spins = [2] * 10
intertwiners = [0] * 5
amplitude = sf.compute_amplitude(spins, intertwiners, Dl=0, gamma=1.2)
tensor = sf.compute_full_tensor(spins, Dl=0, gamma=1.2)
indices = sf.indices_from_intertwiners(spins, intertwiners)
print(amplitude, tensor[indices])
print(tensor.metadata().scientific.spins)
```

`configure()` publishes a copy. Editing the original configuration or a
`current_config()` result has no effect until another `configure()` call.
See [configuration defaults](configuration.md#configuration-defaults) for every
setting. Configure common settings before launching concurrent computations. The nested
`tensors`, `low_level`, and `coherent_amplitude` settings have the same fields
and validation as C++. Absent paths are `None`; populated paths are
`pathlib.Path` objects. Clear an exact tensor output path by assigning `None`
or `""` to `config.tensors.output_path`.

`compute_coherent_amplitude(spins, Dl, gamma, angles, backend=None)` accepts a
`20 x 2` angle array in radians, following the
[canonical boundary-normal order](scientific-conventions.md).
`compute_coherent_coefficients(spins, angles)` takes four doubled spins and a
`4 x 2` angle array, returning an owned NumPy `complex128` coefficient vector.
`check_closure(spins, angles, absolute_tolerance=1e-12,
relative_tolerance=1e-12)` returns five node reports and `all_closed`; it also
supports `bool(result)`.

The fixed-intertwiner computation is CPU-only. Coherent and tensor calls use
the configured backend when omitted; `backend=sf.Backend.CPU` or `.GPU`
overrides it for one call. There is no silent GPU-to-CPU fallback. The optional
`cache_policy=sf.AmplitudeCachePolicy.prefer_full_tensor` argument to
`compute_amplitude()` enables the C++ exact-metadata persisted-tensor lookup.

## Tensors and NumPy

`FullTensor` retains canonical `(i1,i2,i3,i4,i5)` indexing for every physical
representation. `tensor[i1,i2,i3,i4,i5]`, `tensor.at(indices)`, and
`tensor.at(i1,i2,i3,i4,i5)` all read canonical entries; tuple assignment or
`set_at(indices, value)` modifies writable storage. Indices are nonnegative;
negative indexing and slicing are provided on NumPy arrays instead.

```python
dense = tensor.numpy(copy=True)  # expand/copy into independent canonical storage
assert dense.shape == tensor.shape
assert dense.flags.f_contiguous  # i1 varies fastest, as in C++

physical = tensor.stored_numpy()  # view of the actual one-dimensional payload
print(tensor.logical_size, tensor.stored_size, tensor.storage)

owned_dense = tensor.dense_copy()
view = owned_dense.numpy()       # zero-copy canonical view
view[0, 0, 0, 0, 0] = 1.0
assert owned_dense[0, 0, 0, 0, 0] == 1.0
```

`numpy(copy=False)` is available only for canonical dense tensors; a
reassociated tensor requires explicit `copy=True` for canonical expansion.
`stored_numpy(copy=False)` is available for every representation. It follows
the physical order described by the metadata and must not be reshaped into a
canonical tensor. `size`, `stored_size`, and `at_flat()` also refer to this
physical payload; `shape`, `strides` (in elements), and `logical_size` describe
the logical canonical tensor.

Views keep their tensor and mapping alive even after the original Python
variable is deleted. Read-only mappings produce read-only NumPy views. Copies
are independent and writable. Do not mutate a tensor or shared view from one
thread while another thread reads or writes that storage. Storage-invalidating
operations such as `clear()` and `swap()` are intentionally not exposed.

### Native NumPy arrays and einsum

`FullTensor` implements NumPy's `__array__(dtype=None, copy=None)` protocol.
You can pass the output of `compute_full_tensor()` directly to `np.einsum`,
`np.add`, `np.sum`, `np.transpose`, and other functions that accept arrays:

```python
import numpy as np

left = sf.compute_full_tensor([2] * 10, Dl=0, gamma=1.2)
right = left.clone()  # or compute another tensor with matching contracted axes
matrix = np.einsum("abcde,fbcde->af", left, right, optimize=True)
assert matrix.shape == (left.shape[0], right.shape[0])
```

The labels `b,c,d,e` are summed; `a,f` remain as matrix axes. `einsum` is a
NumPy operation, so no separate tensor class or package is required. For a
native NumPy tensor with standard Python operators and slicing, convert once:

```python
array = np.asarray(left)
first_slice = array[0, :, :, :, :]
scaled = 2.0 * array
single_precision = np.asarray(left, dtype=np.float32)
independent = np.array(left, copy=True)
```

For dense float64 tensors, default conversion returns a view retaining the
C++ owner and any read-only mapping protection. For compressed tensors it
allocates and expands canonical dense storage automatically. A dtype change
also requires a copy. `copy=True` always gives independent writable storage;
`copy=False` rejects any required expansion or dtype conversion instead of
silently copying (`np.asarray(..., copy=False)` requires NumPy 2 or newer).
The existing explicit `tensor.numpy()`/`tensor.numpy(copy=True)` choices remain
available. A mapped compressed tensor converts to a writable independent
array because canonical expansion requires new storage.

Convert a compressed tensor once and reuse the resulting array across many
contractions to avoid repeating expansion. Conversion exports canonical
values, not scientific metadata; retain the `FullTensor` when you need its
metadata or persistence methods. The returned NumPy arrays and `einsum`
results live in host memory, even when the tensor was computed on GPU.

### Save and map tensors

Persistence uses the same `.sl2cxx` format as C++:

```python
from pathlib import Path

tensor.save(Path("vertex.sl2cxx"))
loaded = sf.FullTensor.load("vertex.sl2cxx")  # owned, writable copy
mapped = sf.FullTensor.map("vertex.sl2cxx")  # read-only by default
writable = sf.FullTensor.map("vertex.sl2cxx", sf.TensorAccess.read_write)
writable.flush()
```

`FullTensor(shape)` creates a zero-filled general dense tensor;
`FullTensor.create_mapped(path, shape)` creates a mapped one. Such containers
have `metadata().scientific is None`. Computed tensors retain typed `spins`,
`Dl`, `gamma`, and `simplicity_constraint` metadata through persistence.
`convert_storage(destination_storage, destination_path,
overwrite_existing=False)` performs the C++ lossless on-disk conversion with
the same strict symmetry checks. `clone()` preserves the physical layout;
`dense_copy()` materializes canonical storage. `format()` returns a bounded
canonical diagnostic dump and `print()` sends it through Python's stdout.

## Progress, threads, and exceptions

### Display a progress bar

The built-in progress bar works for both `compute_coherent_amplitude()` and
`compute_full_tensor()`, with **CPU and GPU** backends. It is disabled by
default. Turn it on before calling `configure()`:

```python
config = sf.default_config()
config.backend = sf.Backend.CPU  # use GPU with a CUDA build and visible device
config.progress_bar = True
config.verbosity = sf.Verbosity.summary
sf.configure(config)

tensor = sf.compute_full_tensor([2] * 10, Dl=2, gamma=1.2)
```

`Verbosity.summary` leaves the completed bar and a result summary visible.
With the default `Verbosity.silent`, the live bar is erased on successful
completion. A tiny calculation may finish before you notice the display.
The bar reports the contraction phase; preparation may run before it appears.
An exact persisted-tensor cache hit can finish without entering that phase.
The CPU-only fixed-intertwiner `compute_amplitude()` does not expose progress.

Output goes to native stderr. In a suitable terminal it updates one live line
with percentage, elapsed time, and ETA once enough work has completed. Pipes,
redirected output, and batch logs get newline-delimited progress records. IDEs
and notebooks without a native terminal generally get those records rather
than a live terminal bar. A `progress=` callback is an independent event sink;
collecting events in a list does not draw a bar.

The [Python example](../examples/python_example.py) selects either
backend without silently switching and enables the visible bar:

```sh
python examples/python_example.py --backend CPU --einsum
python examples/python_example.py --backend GPU --einsum
# Increase the workload to keep the progress bar visible for longer:
python examples/python_example.py --two-j 12 --Dl 6
```

Use the interpreter/package setup above. The GPU invocation requires a CUDA
build and an available compatible GPU. The example defaults to CPU, doubled
spin `1` on every face (physical `j=1/2`), `Dl=2`, and two host threads, and
runs all basic operations. It explicitly uses `gamma_j_p1`, gamma 1.2, and
disables coherent pruning for its independent reference comparison. Use
`--help` to adjust the workload; `--quiet` makes the bar disappear after
success. The optional `--einsum` section is off by default.

### Receive progress events

The three numerical computations release Python's GIL. Progress callbacks
reacquire it and receive independent, read-only `ProgressEvent` snapshots,
which can be retained after the call. A callback requires an explicit backend,
matching the native per-call callback overload:

```python
events = []
tensor = sf.compute_full_tensor(
    [2] * 10, 0, 1.2, backend=sf.Backend.CPU, progress=events.append)
for event in events:
    print(event.status, event.completed, event.total)
```

Callbacks are synchronous or arrive on the library's reporter thread. Keep
them short; GUI applications should enqueue events for their own UI thread.
Exceptions in callbacks are reported through `sys.unraisablehook` and do not
change numerical results. Concurrent calls keep separate per-call callbacks.
A compatible tensor cache hit does not perform contraction and may emit no
events. Completion marks the reported phase; file publication may follow it.

`config.progress_bar` enables the native terminal renderer independently.
The borrowed C++ `log_stream` pointer and process-global `progress_callback`
are not exposed as Python configuration fields; Python callbacks are confined
to the lifetime of each computation. Native diagnostics retain C++ stderr
behavior, while `FullTensor.print()` supports Python stdout redirection.

Operational failures raise `sf.Error`, a `RuntimeError` subclass with a typed
`.code`:

```python
try:
    sf.compute_full_tensor([2] * 10, 0, 1.2, backend=sf.Backend.GPU)
except sf.Error as error:
    if error.code == sf.ErrorCode.backend_unavailable:
        print("The requested GPU backend is unavailable")
    else:
        raise
```

Invalid scientific inputs raise `ValueError`; invalid indices raise
`IndexError`; C++ arithmetic/extent overflow raises `OverflowError`; allocation
failures raise `MemoryError`; filesystem failures raise `OSError` (including
its standard errno subclasses). Incompatible Python types/shapes raise
`TypeError`, as can a Python integer outside a native parameter's range.
Use `version()`, `build_info()`, `backend_available()`,
`cuda_device_count()`, and `cuda_device_info(ordinal)` to inspect the loaded
library and hardware.

## Validation

Tests are disabled by default. To include the Python integration suite, build
with `BUILD_TESTING=ON`, `SL2CFOAMXX_BUILD_PYTHON=ON`, and NumPy installed in
the selected interpreter, then run:

```sh
ctest --test-dir build-python -L python --output-on-failure
```

The suite checks complete CPU/GPU computations, NumPy ownership and contractions,
persistence, errors, callbacks, and terminal/redirected progress. GPU checks
explicitly skip when a compatible device is unavailable.
