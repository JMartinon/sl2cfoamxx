# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

"""Numerical and ownership checks for the installed-style Python API.

Run with the CMake build's python directory on PYTHONPATH, or against an
installed package: python tests/test_python_api.py
Native redirected-output and terminal progress checks run separately:
    python tests/test_python_api.py progress --backend CPU
    python tests/test_python_api.py progress --backend GPU
An unavailable requested GPU or pseudo-terminal exits with skip code 77.
"""

import argparse
import errno
import gc
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import sl2cfoamxx as foam


# This unequal-spin, mixed-parity fixture also detects i2/i5 recoupling-order
# errors. The independent magnetic-sum references were recorded by the former
# native recoupling regression and do not depend on the Python coefficient path.
SPINS = (2, 1, 1, 2, 1, 1, 2, 1, 1, 1)
ANGLES = np.array([
    [0.31, 0.13], [0.77, -0.42], [1.29, 1.07], [2.11, 2.21],
    [2.39, 2.48], [1.47, -1.14], [0.43, -0.27], [0.91, 0.61],
    [1.66, 1.52], [2.52, -2.31], [0.58, 0.36], [1.04, -0.83],
    [1.18, 0.94], [1.83, -1.73], [2.67, 2.69], [0.69, -0.48],
    [0.82, 0.57], [1.96, 1.89], [2.78, -2.52], [1.33, -1.06],
], dtype=np.float64)
NODE_SPIN_INDICES = ((0, 1, 2, 3), (4, 5, 6, 0), (7, 8, 1, 4),
                     (9, 2, 5, 7), (3, 6, 8, 9))
NODE_ANGLE_ROWS = ((0, 1, 2, 3), (5, 6, 7, 4), (10, 11, 8, 9),
                   (15, 12, 13, 14), (16, 17, 18, 19))


class PythonApiTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sl2cfoamxx-python-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = foam.default_config()
        self.config.tensors.directory = self.directory / "cache"
        self.config.tensors.storage = foam.TensorStorage.in_memory_dense
        self.config.tensors.reuse_existing = False
        self.config.backend = foam.Backend.CPU
        self.config.cpu_threads = 1
        self.config.verbosity = foam.Verbosity.silent
        self.config.progress_bar = False
        self.config.coherent_amplitude.pruning_ratio = 0.0
        foam.configure(self.config)

    def assertNumericallyEqual(self, actual, expected, rtol=5.0e-11):
        self.assertTrue(np.isfinite(actual))
        self.assertTrue(np.isfinite(expected))
        self.assertLessEqual(abs(actual - expected),
                             rtol * max(abs(expected), 1.0e-300))

    def assertErrorCode(self, code, function, *args, **kwargs):
        with self.assertRaises(foam.Error) as caught:
            function(*args, **kwargs)
        self.assertIsInstance(caught.exception, RuntimeError)
        self.assertEqual(caught.exception.code, code)

    def test_loaded_library_inspection(self):
        info = foam.build_info()
        self.assertEqual(foam.version(), info.version)
        self.assertTrue(info.version)
        self.assertTrue(info.compiler_id)
        self.assertTrue(info.blas_provider)
        self.assertIs(foam.TensorLayout.full_tensor_reassociated_by_i1,
                      foam.TensorLayout.full_tensor_reassociated_by_a1)
        self.assertEqual(
            foam.FullTensorReassociatedLayout.reflection_triangular_by_i1,
            foam.FullTensorReassociatedLayout.reflection_triangular_by_a1)
        self.assertTrue(foam.backend_available(foam.Backend.CPU))
        if not info.cuda_backend_built:
            self.assertFalse(foam.backend_available(foam.Backend.GPU))
            self.assertEqual(foam.cuda_device_count(), 0)
            self.assertErrorCode(foam.ErrorCode.backend_unavailable,
                                 foam.cuda_device_info, 0)

    def test_configuration_nested_fields_and_snapshot_isolation(self):
        # A default-constructed configuration and each public options class
        # must be independently usable from Python.
        self.assertIsInstance(foam.LibraryConfig().tensors,
                              foam.TensorCacheOptions)
        for defaults in (foam.default_config(), foam.LibraryConfig()):
            self.assertTrue(defaults.progress_bar)
            self.assertEqual(defaults.tensors.storage,
                             foam.TensorStorage.on_disk_sym)
        cache_options = foam.TensorCacheOptions()
        self.assertEqual(cache_options.storage, foam.TensorStorage.on_disk_sym)
        self.assertIsNone(cache_options.directory)
        self.assertIsNone(cache_options.output_path)
        for field in ("directory", "output_path"):
            for empty in (None, ""):
                with self.subTest(field=field, empty=empty):
                    setattr(cache_options, field, self.directory / "example")
                    self.assertEqual(getattr(cache_options, field),
                                     self.directory / "example")
                    setattr(cache_options, field, empty)
                    self.assertIsNone(getattr(cache_options, field))
        options = foam.LowLevelOptions()
        options.booster_rho_tile = 8
        options.full_tensor_cpu_p_batch = 128
        options.full_tensor_cpu_final_tile = foam.FullTensorCpuFinalTile.tile_4x8
        self.config.low_level = options
        self.config.coherent_amplitude = foam.CoherentAmplitudeOptions()
        self.config.coherent_amplitude.pruning_ratio = 0.25
        self.config.printing_style = foam.PrintingStyle.ASCII
        foam.configure(self.config)
        snapshot = foam.current_config()
        self.assertEqual(snapshot.low_level.booster_rho_tile, 8)
        self.assertEqual(snapshot.low_level.full_tensor_cpu_p_batch, 128)
        self.assertEqual(snapshot.coherent_amplitude.pruning_ratio, 0.25)
        self.assertEqual(snapshot.printing_style, foam.PrintingStyle.ASCII)
        self.assertEqual(Path(snapshot.tensors.directory).resolve(),
                         (self.directory / "cache").resolve())
        self.assertIsNone(snapshot.tensors.output_path)
        snapshot.low_level.booster_rho_tile = 4
        self.config.coherent_amplitude.pruning_ratio = 0.5
        self.assertEqual(foam.current_config().low_level.booster_rho_tile, 8)
        self.assertEqual(foam.current_config().coherent_amplitude.pruning_ratio,
                         0.25)
        self.assertFalse(hasattr(snapshot, "progress_callback"))
        self.assertFalse(hasattr(snapshot, "log_stream"))
        snapshot.low_level.booster_rho_tile = 3
        with self.assertRaises(ValueError):
            foam.configure(snapshot)
        self.assertEqual(foam.current_config().low_level.booster_rho_tile, 8)

    def test_intertwiner_indices_and_half_integer_fixed_amplitude(self):
        intertwiners = (3, 2, 0, 2, 2)
        indices = foam.indices_from_intertwiners(SPINS, intertwiners)
        self.assertEqual(tuple(indices), (1, 1, 0, 1, 1))
        for simplicity in (foam.SimplicityConstraint.gamma_j,
                           foam.SimplicityConstraint.gamma_j_p1):
            with self.subTest(simplicity=simplicity):
                self.config.simplicity_constraint = simplicity
                foam.configure(self.config)
                tensor = foam.compute_full_tensor(SPINS, 0, 1.2,
                                                   foam.Backend.CPU)
                direct = foam.compute_amplitude(SPINS, intertwiners, 0, 1.2)
                self.assertNumericallyEqual(direct, tensor[tuple(indices)])
                self.assertEqual(tensor.at(*indices), tensor.at(tuple(indices)))
                self.assertEqual(tensor.at(i1=1, i2=1, i3=0, i4=1, i5=1),
                                 tensor.at(a1=1, a2=1, a3=0, a4=1, a5=1))
                self.assertEqual(tensor.shape, (2, 2, 2, 2, 2))

    def test_coefficients_match_independent_magnetic_sum(self):
        expected = {
            1: np.array([0.1282584174511503 + 0.1343652056345688j,
                         0.2484734822500689 + 0.06033324720018829j]),
            4: np.array([0.02331902555825567 - 0.1698302920514597j,
                         0.002828506333065372 - 0.2918289545554249j]),
        }
        for node, reference in expected.items():
            with self.subTest(node=node):
                coefficients = foam.compute_coherent_coefficients(
                    [SPINS[i] for i in NODE_SPIN_INDICES[node]],
                    ANGLES[list(NODE_ANGLE_ROWS[node])])
                self.assertIsInstance(coefficients, np.ndarray)
                self.assertEqual(coefficients.dtype, np.dtype(np.complex128))
                self.assertEqual(coefficients.shape, (2,))
                np.testing.assert_allclose(coefficients, reference,
                                           rtol=5.0e-13, atol=2.0e-15)
                saved = coefficients.copy()
                gc.collect()
                np.testing.assert_array_equal(coefficients, saved)

    def test_coherent_amplitude_matches_canonical_numpy_contraction(self):
        coefficients = [foam.compute_coherent_coefficients(
            tuple(SPINS[i] for i in spin_indices), ANGLES[list(angle_rows)])
            for spin_indices, angle_rows in zip(NODE_SPIN_INDICES,
                                                NODE_ANGLE_ROWS)]
        backends = [foam.Backend.CPU]
        if foam.backend_available(foam.Backend.GPU):
            backends.append(foam.Backend.GPU)
        for simplicity in (foam.SimplicityConstraint.gamma_j,
                           foam.SimplicityConstraint.gamma_j_p1):
            self.config.simplicity_constraint = simplicity
            foam.configure(self.config)
            for Dl in (0, 1, 2):
                with self.subTest(simplicity=simplicity, Dl=Dl):
                    tensor = foam.compute_full_tensor(SPINS, Dl, 1.2)
                    # Direct FullTensor input exercises NumPy conversion as
                    # well as the lifted-shell coherent symmetry regression.
                    expected = np.einsum("abcde,a,b,c,d,e->", tensor, *coefficients,
                                         optimize=True)
                    self.assertGreater(abs(expected), 1.0e-24)
                    actual = foam.compute_coherent_amplitude(SPINS, Dl, 1.2, ANGLES)
                    self.assertIsInstance(actual, complex)
                    self.assertNumericallyEqual(actual, expected)
                    for backend in backends:
                        with self.subTest(backend=backend):
                            explicit = foam.compute_coherent_amplitude(
                                SPINS, Dl, 1.2, ANGLES, backend)
                            self.assertNumericallyEqual(explicit, expected)
                    metadata = tensor.metadata()
                    self.assertEqual(tuple(metadata.scientific.spins), SPINS)
                    self.assertEqual(metadata.scientific.Dl, Dl)
                    self.assertEqual(metadata.scientific.gamma, 1.2)
                    self.assertEqual(metadata.logical_shape, tensor.shape)
                    self.assertEqual(metadata.logical_values, tensor.logical_size)
                    self.assertEqual(metadata.stored_values, tensor.stored_size)

    def test_numpy_sequences_and_strided_angles(self):
        spins = np.array(SPINS, dtype=np.int64)
        padded_angles = np.zeros((20, 4), dtype=np.float64)
        padded_angles[:, ::2] = ANGLES
        angles = padded_angles[:, ::2]
        self.assertFalse(angles.flags.c_contiguous)
        actual = foam.compute_coherent_amplitude(spins, 0, 1.2, angles)
        expected = foam.compute_coherent_amplitude(list(SPINS), 0, 1.2,
                                                    ANGLES.tolist())
        self.assertEqual(actual, expected)
        np.testing.assert_array_equal(
            foam.indices_from_intertwiners(spins, np.array([3, 2, 0, 2, 2])),
            [1, 1, 0, 1, 1])

    def test_closure_uses_physical_spin_weights(self):
        angles = np.zeros((20, 2), dtype=np.float64)
        report = foam.check_closure(SPINS, angles)
        self.assertFalse(report.all_closed)
        self.assertFalse(bool(report))
        # Natural node incidence differs from the recoupling order but has
        # the same sum. All normals point along +z in this simple oracle.
        for node, spin_indices in zip(report.nodes, NODE_SPIN_INDICES):
            scale = sum(SPINS[i] for i in spin_indices) / 2
            np.testing.assert_array_equal(node.vector, [0.0, 0.0, scale])
            self.assertEqual(node.scale, scale)
            self.assertEqual(node.residual_norm, scale)
            self.assertEqual(node.relative_residual, 1.0)
            self.assertFalse(node.closed)
        self.assertTrue(foam.check_closure(SPINS, angles,
                                           absolute_tolerance=4.0).all_closed)
        zero = foam.check_closure([0] * 10, angles, 0.0, 0.0)
        self.assertTrue(bool(zero))
        self.assertTrue(all(node.closed and node.scale == 0.0 and
                            node.relative_residual == 0.0 for node in zero.nodes))

    def test_scientific_input_validation(self):
        for gamma in (math.nan, math.inf, -math.inf, -1.2, -sys.float_info.min):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                foam.compute_full_tensor(SPINS, 0, gamma)
        for call in (
            lambda: foam.compute_amplitude(SPINS, (3, 2, 0, 2, 2), 0, -1.2),
            lambda: foam.compute_coherent_amplitude(SPINS, 0, -1.2, ANGLES),
        ):
            with self.assertRaisesRegex(ValueError, "gamma must be nonnegative"):
                call()
        for spins in ((-1,) + SPINS[1:], (1,) + (0,) * 9):
            with self.subTest(spins=spins), self.assertRaises(ValueError):
                foam.compute_full_tensor(spins, 0, 1.2)
        with self.assertRaises(ValueError):
            foam.indices_from_intertwiners(SPINS, (2, 2, 0, 2, 2))
        for malformed in (SPINS[:-1], SPINS + (0,), [0.5] * 10):
            with self.subTest(spins=malformed):
                with self.assertRaises((TypeError, ValueError)):
                    foam.compute_full_tensor(malformed, 0, 1.2)
        with self.assertRaises((TypeError, ValueError, OverflowError)):
            foam.compute_full_tensor(SPINS, -1, 1.2)
        with self.assertRaises((TypeError, ValueError)):
            foam.compute_coherent_amplitude(SPINS, 0, 1.2, ANGLES[:-1])
        invalid_angles = ANGLES.copy()
        invalid_angles[3, 0] = math.nan
        with self.assertRaises(ValueError):
            foam.compute_coherent_amplitude(SPINS, 0, 1.2, invalid_angles)
        with self.assertRaises(ValueError):
            foam.check_closure(SPINS, ANGLES, relative_tolerance=-1.0)

    def test_invalid_first_use_does_not_create_cache_directory(self):
        cache = self.directory / "never-created"
        script = """
import os
from pathlib import Path
import sl2cfoamxx as foam
spins = [1] + [0] * 9
angles = [[0.0, 0.0]] * 20
invalid_calls = [
    lambda: foam.compute_full_tensor(spins, 0, 1.2),
    lambda: foam.compute_amplitude(spins, [0] * 5, 0, 1.2),
    lambda: foam.compute_coherent_amplitude(spins, 0, 1.2, angles),
    lambda: foam.compute_full_tensor([0] * 10, 0, -1.2),
    lambda: foam.compute_amplitude([0] * 10, [0] * 5, 0, -1.2),
    lambda: foam.compute_coherent_amplitude([0] * 10, 0, -1.2, angles),
]
for call in invalid_calls:
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError('invalid scientific input was accepted')
for call in [
    lambda: foam.compute_full_tensor([0] * 10, 0, 1.2,
                                    progress=lambda event: None),
    lambda: foam.compute_coherent_amplitude([0] * 10, 0, 1.2, angles,
                                           progress=lambda event: None),
]:
    try:
        call()
    except TypeError:
        pass
    else:
        raise AssertionError('a callback requires an explicit backend')
assert not Path(os.environ['SL2CFOAMXX_CACHE_DIR']).exists()
"""
        environment = os.environ.copy()
        environment["SL2CFOAMXX_CACHE_DIR"] = str(cache)
        # Resolve the imported package before changing directories. A caller
        # may have supplied a relative build-tree PYTHONPATH.
        package_root = str(Path(foam.__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = os.pathsep.join(
            [package_root, environment.get("PYTHONPATH", "")])
        completed = subprocess.run([sys.executable, "-c", script],
                                   cwd=self.directory, env=environment,
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0,
                         completed.stdout + completed.stderr)
        self.assertFalse(cache.exists())

    def test_fresh_import_preserves_loader_flags_and_runs_vendor_blas(self):
        # Tiny tensors use local kernels and miss MKL's deferred shared-library
        # loading. This case fails in an unpatched Linux/MKL Python process
        # even though importing the extension itself succeeds.
        script = """
import sys
original_flags = sys.getdlopenflags()
if sys.argv[1] == 'numpy-first':
    import numpy as np
import sl2cfoamxx as foam
assert sys.getdlopenflags() == original_flags

config = foam.default_config()
config.backend = foam.Backend.CPU
config.cpu_threads = 1
config.tensors.directory = sys.argv[2]
config.tensors.storage = foam.TensorStorage.in_memory_dense
config.tensors.reuse_existing = False
foam.configure(config)
spins = [12] * 10
tensor = foam.compute_full_tensor(spins, 1, 1.2, foam.Backend.CPU)
assert tensor.shape == (13, 13, 13, 13, 13)

# In the other order, run the native BLAS path before NumPy can load another
# numerical runtime and accidentally conceal an extension-loader defect.
if sys.argv[1] == 'library-first':
    import numpy as np
for indices in ((0, 0, 0, 0, 0), (6, 5, 4, 3, 2), (12, 12, 12, 12, 12)):
    intertwiners = [2 * index for index in indices]
    assert tuple(foam.indices_from_intertwiners(spins, intertwiners)) == indices
    direct = foam.compute_amplitude(
        spins, intertwiners, 1, 1.2, foam.AmplitudeCachePolicy.disabled)
    assert np.isfinite(direct) and direct != 0.0
    np.testing.assert_allclose(tensor[indices], direct, rtol=5e-10, atol=0)
assert sys.getdlopenflags() == original_flags
"""
        environment = os.environ.copy()
        package_root = str(Path(foam.__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = os.pathsep.join(
            [package_root, environment.get("PYTHONPATH", "")])
        for order in ("numpy-first", "library-first"):
            with self.subTest(import_order=order):
                completed = subprocess.run(
                    [sys.executable, "-c", script, order,
                     str(self.directory / order)],
                    cwd=self.directory, env=environment,
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, 0,
                                 completed.stdout + completed.stderr)

    def test_explicit_backend_overrides_configuration(self):
        expected = foam.compute_full_tensor(SPINS, 0, 1.2).numpy(copy=True)
        self.config.backend = foam.Backend.GPU
        foam.configure(self.config)
        actual = foam.compute_full_tensor(SPINS, 0, 1.2, foam.Backend.CPU)
        np.testing.assert_array_equal(actual.numpy(), expected)
        self.assertEqual(foam.current_config().backend, foam.Backend.GPU)
        if not foam.build_info().cuda_backend_built:
            self.assertErrorCode(foam.ErrorCode.backend_unavailable,
                                 foam.compute_full_tensor, SPINS, 0, 1.2)
            self.assertErrorCode(foam.ErrorCode.backend_unavailable,
                                 foam.compute_coherent_amplitude,
                                 SPINS, 0, 1.2, ANGLES)

    def test_dense_numpy_views_layout_mutation_and_lifetime(self):
        tensor = foam.FullTensor((2, 3, 1, 2, 1))
        self.assertIsNone(tensor.metadata().scientific)
        self.assertIsNone(tensor.path)
        self.assertIsNone(tensor.root_path)
        self.assertIsNone(tensor.metadata().path)
        array = tensor.numpy()
        self.assertEqual(array.dtype, np.dtype(np.float64))
        self.assertEqual(array.shape, (2, 3, 1, 2, 1))
        self.assertTrue(array.flags.f_contiguous)
        self.assertTrue(array.flags.writeable)
        array[...] = np.arange(12).reshape(array.shape, order="F")
        stored = tensor.stored_numpy()
        self.assertTrue(np.shares_memory(array, stored))
        np.testing.assert_array_equal(stored, np.arange(12))
        self.assertEqual(tensor[1, 2, 0, 1, 0], 11.0)
        tensor[0, 1, 0, 0, 0] = -5.0
        self.assertEqual(array[0, 1, 0, 0, 0], -5.0)
        self.assertEqual(tensor.at_flat(2), -5.0)
        copied = tensor.numpy(copy=True)
        physical_copy = tensor.stored_numpy(copy=True)
        self.assertFalse(np.shares_memory(array, copied))
        self.assertFalse(np.shares_memory(stored, physical_copy))
        tensor.fill(3.0)
        self.assertTrue(np.all(array == 3.0))
        self.assertEqual(copied[0, 1, 0, 0, 0], -5.0)
        del tensor
        gc.collect()
        stored[0] = 7.0
        self.assertEqual(array[0, 0, 0, 0, 0], 7.0)

    def test_numpy_array_protocol_dense_copy_and_dtype(self):
        tensor = foam.FullTensor((2, 3, 1, 2, 1))
        expected = (np.arange(12, dtype=np.float64) - 3.25).reshape(
            tensor.shape, order="F")
        tensor.numpy()[...] = expected
        array = np.asarray(tensor)
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.dtype, np.dtype(np.float64))
        self.assertTrue(array.flags.f_contiguous)
        self.assertTrue(np.shares_memory(array, tensor.stored_numpy()))
        np.testing.assert_array_equal(array, expected)
        for dtype in (None, np.float64, "float64", np.dtype("=f8")):
            with self.subTest(dtype=dtype):
                view = tensor.__array__(dtype=dtype, copy=False)
                self.assertTrue(np.shares_memory(view, array))
                if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
                    view = np.asarray(tensor, dtype=dtype, copy=False)
                    self.assertTrue(np.shares_memory(view, array))

        copied = np.array(tensor, copy=True)
        self.assertFalse(np.shares_memory(copied, array))
        self.assertTrue(copied.flags.writeable)
        self.assertFalse(np.shares_memory(tensor.__array__(copy=True), array))
        if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
            self.assertFalse(np.shares_memory(np.asarray(tensor, copy=True), array))
        copied[0, 0, 0, 0, 0] = 99.0
        self.assertEqual(tensor[0, 0, 0, 0, 0], expected[0, 0, 0, 0, 0])
        # A byte-swapped float64 request also needs conversion despite having
        # the same scalar kind and width as the tensor's native payload.
        for dtype in (np.float32, np.complex128, np.dtype("f8").newbyteorder("S")):
            with self.subTest(dtype=dtype):
                converted = np.asarray(tensor, dtype=dtype)
                self.assertEqual(converted.dtype, np.dtype(dtype))
                self.assertTrue(converted.flags.f_contiguous)
                self.assertFalse(np.shares_memory(converted, array))
                np.testing.assert_array_equal(converted, expected.astype(dtype))
                explicit_copy = np.array(tensor, dtype=dtype, copy=True)
                self.assertFalse(np.shares_memory(explicit_copy, array))
                np.testing.assert_array_equal(explicit_copy, converted)
                with self.assertRaises(ValueError):
                    tensor.__array__(dtype=dtype, copy=False)
                if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
                    with self.assertRaises(ValueError):
                        np.asarray(tensor, dtype=dtype, copy=False)

        np.testing.assert_array_equal(np.add(tensor, 2.0), expected + 2.0)
        np.testing.assert_array_equal(np.transpose(tensor), expected.T)
        self.assertEqual(np.sum(tensor), np.sum(expected))
        del tensor
        gc.collect()
        array[1, 2, 0, 1, 0] = 17.0
        self.assertEqual(array[1, 2, 0, 1, 0], 17.0)

    def test_numpy_einsum_accepts_computed_dense_and_compact_tensors(self):
        for storage in (foam.TensorStorage.in_memory_dense,
                        foam.TensorStorage.in_memory_sym):
            with self.subTest(storage=storage):
                self.config.tensors.storage = storage
                foam.configure(self.config)
                first = foam.compute_full_tensor([1] * 10, 0, 1.2)
                second = foam.compute_full_tensor([1] * 10, 0, 1.3)
                first_array = first.numpy(copy=True)
                second_array = second.numpy(copy=True)
                expected = np.einsum("abcde,abfgh->cdefgh",
                                     first_array, second_array)
                self.assertGreater(np.max(np.abs(expected)), 0.0)
                for optimize in (False, True):
                    actual = np.einsum("abcde,abfgh->cdefgh", first, second,
                                       optimize=optimize)
                    np.testing.assert_allclose(actual, expected,
                                               rtol=5.0e-14,
                                               atol=5.0e-16 * np.max(np.abs(expected)))
                np.testing.assert_allclose(np.add(first, second),
                                           first_array + second_array,
                                           rtol=0.0, atol=0.0)

    def test_numpy_array_protocol_compact_expansion_and_copy_policy(self):
        self.config.tensors.storage = foam.TensorStorage.in_memory_sym
        foam.configure(self.config)
        tensor = foam.compute_full_tensor([1] * 10, 0, 1.2)
        self.assertTrue(tensor.is_symmetry_compressed)
        expected = tensor.numpy(copy=True)
        array = np.asarray(tensor)
        self.assertEqual(array.shape, tensor.shape)
        self.assertEqual(array.size, tensor.logical_size)
        self.assertTrue(array.flags.f_contiguous)
        self.assertTrue(array.flags.writeable)
        self.assertFalse(np.shares_memory(array, tensor.stored_numpy()))
        np.testing.assert_array_equal(array, expected)
        path = self.directory / "compact-array-protocol.sl2cxx"
        tensor.save(path)
        mapped = foam.FullTensor.map(path)
        mapped_array = np.asarray(mapped)
        self.assertFalse(mapped.writable)
        self.assertTrue(mapped_array.flags.writeable)
        self.assertFalse(np.shares_memory(mapped_array, mapped.stored_numpy()))
        np.testing.assert_array_equal(mapped_array, expected)
        del mapped
        gc.collect()
        np.testing.assert_array_equal(mapped_array, expected)
        for dtype in (None, np.float64, np.float32):
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError):
                    tensor.__array__(dtype=dtype, copy=False)
                if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
                    with self.assertRaises(ValueError):
                        np.asarray(tensor, dtype=dtype, copy=False)
                converted = np.asarray(tensor, dtype=dtype)
                np.testing.assert_array_equal(converted,
                                               expected.astype(dtype or np.float64))
                copied = np.array(tensor, dtype=dtype, copy=True)
                self.assertFalse(np.shares_memory(copied, converted))
                self.assertFalse(np.shares_memory(copied, tensor.stored_numpy()))
                np.testing.assert_array_equal(copied, converted)
        # Expanded arrays are ordinary independent arrays: reflected canonical
        # entries cease to alias each other or the original compact tensor.
        left, reflected = (0, 1, 0, 1, 0), (0, 0, 1, 0, 1)
        array[left] = 7.25
        self.assertEqual(array[reflected], expected[reflected])
        self.assertEqual(tensor[left], expected[left])
        del tensor
        gc.collect()
        self.assertEqual(array[left], 7.25)

    def test_numpy_array_protocol_readonly_mapping_lifetime(self):
        tensor = foam.FullTensor((2, 1, 1, 1, 1))
        tensor.numpy()[:, 0, 0, 0, 0] = [1.25, -2.5]
        path = self.directory / "array-protocol.sl2cxx"
        tensor.save(path)
        mapped = foam.FullTensor.map(path)
        array = np.asarray(mapped)
        self.assertFalse(array.flags.writeable)
        self.assertTrue(np.shares_memory(array, mapped.stored_numpy()))
        with self.assertRaises(ValueError):
            array.setflags(write=True)
        for copy in (False, None):
            view = mapped.__array__(copy=copy)
            self.assertFalse(view.flags.writeable)
            self.assertTrue(np.shares_memory(view, array))
        for writable in (np.array(mapped, copy=True),
                         np.asarray(mapped, dtype=np.float32)):
            self.assertTrue(writable.flags.writeable)
            self.assertFalse(np.shares_memory(writable, array))
            writable[0, 0, 0, 0, 0] = 99.0
            self.assertEqual(array[0, 0, 0, 0, 0], 1.25)
        del mapped
        gc.collect()
        np.testing.assert_array_equal(array.ravel(), [1.25, -2.5])

    def test_empty_tensor_numpy_views(self):
        for tensor in (foam.FullTensor(), foam.FullTensor((2, 0, 3, 1, 1))):
            with self.subTest(shape=tensor.shape):
                self.assertTrue(tensor.empty)
                self.assertEqual(tensor.logical_size, 0)
                self.assertEqual(tensor.stored_size, 0)
                self.assertIsNone(tensor.path)
                self.assertIsNone(tensor.metadata().scientific)
                for copy in (False, True):
                    array = tensor.numpy(copy=copy)
                    stored = tensor.stored_numpy(copy=copy)
                    self.assertEqual(array.shape, tensor.shape)
                    self.assertEqual(array.size, 0)
                    self.assertEqual(array.dtype, np.dtype(np.float64))
                    self.assertTrue(array.flags.f_contiguous)
                    self.assertEqual(stored.shape, (0,))
                    self.assertEqual(stored.dtype, np.dtype(np.float64))
                converted = np.asarray(tensor)
                self.assertEqual(converted.shape, tensor.shape)
                self.assertEqual(converted.size, 0)

    def test_tensor_checked_indices(self):
        tensor = foam.FullTensor((2, 1, 1, 1, 1))
        for indices in ((2, 0, 0, 0, 0), (-1, 0, 0, 0, 0)):
            with self.subTest(indices=indices):
                with self.assertRaises(IndexError):
                    tensor[indices]
                with self.assertRaises(IndexError):
                    tensor[indices] = 1.0
        for index in (-1, 2):
            with self.assertRaises(IndexError):
                tensor.at_flat(index)
        with self.assertRaises((TypeError, IndexError, ValueError)):
            tensor[0, 0]
        with self.assertRaises((TypeError, IndexError)):
            tensor[0.5, 0, 0, 0, 0]

    def test_compressed_tensor_requires_explicit_dense_copy(self):
        self.config.tensors.storage = foam.TensorStorage.in_memory_sym
        foam.configure(self.config)
        tensor = foam.compute_full_tensor([1] * 10, 0, 1.2)
        self.assertTrue(tensor.is_symmetry_compressed)
        self.assertLess(tensor.stored_size, tensor.logical_size)
        with self.assertRaises((ValueError, foam.Error)):
            tensor.numpy()
        dense = tensor.numpy(copy=True)
        self.assertEqual(dense.shape, tensor.shape)
        self.assertTrue(dense.flags.f_contiguous)
        left, right = (0, 1, 0, 1, 0), (0, 0, 1, 0, 1)
        self.assertEqual(tensor[left], tensor[right])
        original = dense[left]
        tensor[left] = 7.25
        self.assertEqual(tensor[right], 7.25)
        self.assertEqual(dense[left], original)
        physical = tensor.stored_numpy()
        self.assertEqual(physical.ndim, 1)
        self.assertEqual(physical.size, tensor.stored_size)
        physical[:] = 2.0
        self.assertTrue(np.all(tensor.numpy(copy=True) == 2.0))
        del tensor
        gc.collect()
        self.assertTrue(np.all(physical == 2.0))

    def test_persistence_mapping_readonly_and_owned_load(self):
        path = self.directory / "dense.sl2cxx"
        tensor = foam.FullTensor.create_mapped(path, (2, 1, 1, 1, 1))
        tensor.numpy()[:, 0, 0, 0, 0] = [1.25, -2.5]
        tensor.flush()
        mapped = foam.FullTensor.map(path)
        self.assertTrue(mapped.is_mapped)
        self.assertFalse(mapped.writable)
        self.assertEqual(Path(mapped.path), path)
        self.assertIsNone(mapped.metadata().scientific)
        readonly = mapped.numpy()
        stored_readonly = mapped.stored_numpy()
        self.assertFalse(readonly.flags.writeable)
        self.assertFalse(stored_readonly.flags.writeable)
        with self.assertRaises(ValueError):
            readonly[0, 0, 0, 0, 0] = 3.0
        with self.assertRaises(ValueError):
            readonly.setflags(write=True)
        self.assertErrorCode(foam.ErrorCode.unsupported_operation,
                             mapped.fill, 3.0)
        loaded = foam.FullTensor.load(path)
        self.assertFalse(loaded.is_mapped)
        self.assertTrue(loaded.writable)
        loaded[0, 0, 0, 0, 0] = 9.0
        self.assertEqual(mapped[0, 0, 0, 0, 0], 1.25)
        readonly_copy = mapped.numpy(copy=True)
        self.assertTrue(readonly_copy.flags.writeable)
        writable = foam.FullTensor.map(path, foam.TensorAccess.read_write)
        writable[1, 0, 0, 0, 0] = -4.0
        writable.flush()
        self.assertEqual(foam.FullTensor.load(path)[1, 0, 0, 0, 0], -4.0)
        del mapped
        gc.collect()
        self.assertEqual(readonly[0, 0, 0, 0, 0], 1.25)

    def test_lossless_disk_conversion_preserves_scientific_metadata(self):
        self.config.tensors.storage = foam.TensorStorage.in_memory_sym
        foam.configure(self.config)
        tensor = foam.compute_full_tensor([1] * 10, 0, 1.2)
        compact_path = self.directory / "compact.sl2cxx"
        tensor.save(compact_path)
        compact = foam.FullTensor.map(compact_path)
        dense = compact.convert_storage(foam.TensorStorage.on_disk_dense,
                                         self.directory / "dense.sl2cxx")
        restored = dense.convert_storage(foam.TensorStorage.on_disk_sym,
                                          self.directory / "restored.sl2cxx")
        np.testing.assert_array_equal(dense.numpy(), tensor.numpy(copy=True))
        np.testing.assert_array_equal(restored.stored_numpy(),
                                      tensor.stored_numpy())
        self.assertTrue(dense.is_mapped and dense.writable and dense.is_dense)
        self.assertEqual(tuple(restored.metadata().scientific.spins), (1,) * 10)
        self.assertEqual(restored.metadata().scientific.gamma, 1.2)
        with self.assertRaises(OSError):
            compact.convert_storage(foam.TensorStorage.on_disk_dense,
                                    self.directory / "dense.sl2cxx")

    def test_operational_and_filesystem_exception_categories(self):
        malformed = self.directory / "invalid.sl2cxx"
        malformed.write_bytes(b"not a tensor")
        self.assertErrorCode(foam.ErrorCode.invalid_tensor_file,
                             foam.FullTensor.load, malformed)
        with self.assertRaises(OSError) as missing:
            foam.FullTensor.load(self.directory / "missing.sl2cxx")
        self.assertEqual(Path(missing.exception.filename),
                         self.directory / "missing.sl2cxx")
        self.assertIsNone(missing.exception.filename2)
        self.config.memory_budget_bytes = 1
        foam.configure(self.config)
        self.assertErrorCode(foam.ErrorCode.resource_limit,
                             foam.compute_full_tensor, SPINS, 0, 1.2)
        self.assertErrorCode(foam.ErrorCode.resource_limit,
                             foam.compute_amplitude, SPINS, (3, 2, 0, 2, 2),
                             0, 1.2)

    def test_explicit_amplitude_cache_policy(self):
        tensor = foam.compute_full_tensor(SPINS, 0, 1.2)
        intertwiners = (3, 2, 0, 2, 2)
        index = tuple(foam.indices_from_intertwiners(SPINS, intertwiners))
        tensor[index] = 7.25  # Detect the persisted lookup independently.
        path = self.directory / "cached.sl2cxx"
        tensor.save(path)
        self.config.tensors.storage = foam.TensorStorage.on_disk_dense
        self.config.tensors.output_path = path
        foam.configure(self.config)
        cached = foam.compute_amplitude(SPINS, intertwiners, 0, 1.2,
                                         foam.AmplitudeCachePolicy.prefer_full_tensor)
        self.assertEqual(cached, 7.25)
        direct = foam.compute_amplitude(SPINS, intertwiners, 0, 1.2,
                                         foam.AmplitudeCachePolicy.disabled)
        self.assertNotEqual(direct, cached)
        # The exact gamma mismatch must fall back, even when a file exists.
        fallback = foam.compute_amplitude(SPINS, intertwiners, 0, 1.3,
                                           foam.AmplitudeCachePolicy.prefer_full_tensor)
        self.assertEqual(fallback,
                         foam.compute_amplitude(SPINS, intertwiners, 0, 1.3))

    def test_concurrent_progress_callbacks_are_isolated_and_events_owned(self):
        streams = [[], []]
        barrier = threading.Barrier(2)

        def compute(which):
            def progress(event):
                # Reenter a safe read-only API with the reacquired Python GIL.
                self.assertEqual(foam.current_config().backend, foam.Backend.CPU)
                streams[which].append(event)

            barrier.wait(timeout=10)
            if which == 0:
                return foam.compute_full_tensor(SPINS, 0, 1.2, foam.Backend.CPU,
                                                 progress=progress)
            return foam.compute_coherent_amplitude(SPINS, 0, 1.2, ANGLES,
                                                    foam.Backend.CPU,
                                                    progress=progress)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(compute, which) for which in range(2)]
            results = [future.result(timeout=30) for future in futures]
        self.assertTrue(np.isfinite(results[1]))
        operations = (foam.ProgressOperation.full_tensor,
                      foam.ProgressOperation.coherent_amplitude)
        for events, operation in zip(streams, operations):
            self.assertGreaterEqual(len(events), 2)
            self.assertEqual(events[0].status, foam.ProgressStatus.started)
            self.assertEqual(events[-1].status, foam.ProgressStatus.completed)
            self.assertEqual(events[-1].completed, events[-1].total)
            for event in events:
                self.assertEqual(event.operation, operation)
                self.assertEqual(event.backend, foam.Backend.CPU)
                self.assertGreaterEqual(event.attempt, 1)
                self.assertLessEqual(event.completed, event.total)
                self.assertTrue(math.isfinite(event.elapsed_seconds))
        snapshots = [[(e.status, e.completed, e.total) for e in events]
                     for events in streams]
        foam.compute_full_tensor(SPINS, 0, 1.2)
        gc.collect()
        self.assertEqual(snapshots,
                         [[(e.status, e.completed, e.total) for e in events]
                          for events in streams])

    def test_progress_callable_lifetime_and_exception_containment(self):
        events = []

        class Recorder:
            def __call__(self, event):
                events.append(event)
                raise RuntimeError("a progress display failed")

        recorder = Recorder()
        reference = weakref.ref(recorder)
        unraisable = []
        original_hook = sys.unraisablehook
        # Retaining a traceback would itself retain Recorder.__call__'s self.
        sys.unraisablehook = lambda args: unraisable.append(
            (args.exc_type, str(args.exc_value)))
        try:
            actual = foam.compute_full_tensor(SPINS, 0, 1.2, foam.Backend.CPU,
                                               progress=recorder)
        finally:
            sys.unraisablehook = original_hook
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(len(unraisable), len(events))
        self.assertTrue(all(kind is RuntimeError and
                            message == "a progress display failed"
                            for kind, message in unraisable))
        del recorder
        gc.collect()
        self.assertIsNone(reference())
        count = len(events)
        expected = foam.compute_full_tensor(SPINS, 0, 1.2, progress=None)
        self.assertEqual(len(events), count)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
        with self.assertRaises(TypeError):
            foam.compute_full_tensor(SPINS, 0, 1.2, foam.Backend.CPU,
                                     progress=42)


_PROGRESS_BACKEND = foam.Backend.CPU
_PROGRESS_SCRIPT = Path(__file__).resolve()


def _configure_progress(directory, display=False, verbosity="silent", style="ASCII"):
    config = foam.default_config()
    config.backend = _PROGRESS_BACKEND
    config.cpu_threads = 1
    config.tensors.directory = directory
    config.tensors.storage = foam.TensorStorage.in_memory_dense
    config.tensors.reuse_existing = False
    config.progress_bar = display
    config.verbosity = getattr(foam.Verbosity, verbosity)
    config.printing_style = getattr(foam.PrintingStyle, style)
    config.coherent_amplitude.pruning_ratio = 0.0
    foam.configure(config)


def _progress_worker(arguments):
    # Keep C++ stderr separate from the JSON result on Python stdout. This
    # exercises the real file descriptor, which redirect_stderr cannot catch.
    with tempfile.TemporaryDirectory(prefix="sl2cfoamxx-progress-child-") as cache:
        _configure_progress(cache, arguments.display, arguments.verbosity, arguments.style)
        events = []

        def progress(event):
            events.append({
                "operation": event.operation.name,
                "backend": event.backend.name,
                "status": event.status.name,
                "attempt": event.attempt,
                "completed": event.completed,
                "total": event.total,
                "elapsed": event.elapsed_seconds,
            })

        options = {"backend": _PROGRESS_BACKEND}
        if arguments.callback:
            options["progress"] = progress
        tensor = foam.compute_full_tensor(SPINS, 0, 1.2, **options)
        amplitude = foam.compute_coherent_amplitude(
            SPINS, 0, 1.2, ANGLES, **options)
        print(json.dumps({
            "tensor": tensor.numpy().ravel(order="F").tolist(),
            "coherent": [amplitude.real, amplitude.imag],
            "events": events,
        }))


def _open_progress_terminal():
    import fcntl
    import pty
    import struct
    import termios

    master, slave = pty.openpty()
    try:
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 100, 0, 0))
    except BaseException:
        os.close(master)
        os.close(slave)
        raise
    return master, slave


def _run_progress_child(display, callback, terminal=False, verbosity="silent", style="ASCII"):
    environment = os.environ.copy()
    package_root = str(Path(foam.__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = os.pathsep.join(
        [package_root, environment.get("PYTHONPATH", "")])
    environment["TERM"] = "xterm-256color"
    # Native output selection checks this spelling; CPython also accepts it
    # on the supported macOS/Linux runtimes. ASCII is separately forced below.
    environment["LC_ALL"] = "en_US.UTF-8" if sys.platform == "darwin" else "C.UTF-8"
    command = [sys.executable, str(_PROGRESS_SCRIPT), "progress", "--worker",
               "--backend", _PROGRESS_BACKEND.name, "--verbosity", verbosity,
               "--style", style]
    if display:
        command.append("--display")
    if callback:
        command.append("--callback")
    if not terminal:
        completed = subprocess.run(command, env=environment, capture_output=True,
                                   timeout=60)
        return completed.returncode, completed.stdout, completed.stderr

    master, slave = _open_progress_terminal()
    output = bytearray()
    try:
        with subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                              stderr=slave) as process:
            os.close(slave)
            slave = None
            deadline = time.monotonic() + 60
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, 60)
                    readable, _, _ = select.select([master], [], [], min(remaining, 1))
                    if not readable:
                        continue
                    try:
                        chunk = os.read(master, 65536)
                    except OSError as error:
                        if error.errno == errno.EIO:  # Linux PTY end of stream.
                            break
                        raise
                    if not chunk:
                        break
                    output.extend(chunk)
                stdout, _ = process.communicate(timeout=10)
            except BaseException:
                process.kill()
                process.communicate(timeout=10)
                raise
            return process.returncode, stdout, bytes(output)
    finally:
        os.close(master)
        if slave is not None:
            os.close(slave)


class ProgressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory(prefix="sl2cfoamxx-progress-oracle-") as cache:
            _configure_progress(cache)
            tensor = foam.compute_full_tensor(SPINS, 0, 1.2, foam.Backend.CPU)
            cls.expected_tensor = tensor.numpy(copy=True).ravel(order="F")
            coefficients = [foam.compute_coherent_coefficients(
                [SPINS[i] for i in spin_indices], ANGLES[list(angle_rows)])
                for spin_indices, angle_rows in zip(NODE_SPIN_INDICES, NODE_ANGLE_ROWS)]
            # Independent route: CPU tensor contracted explicitly with NumPy.
            cls.expected_coherent = np.einsum(
                "abcde,a,b,c,d,e->", tensor.numpy(), *coefficients)

    def computation(self, **options):
        returncode, stdout, stderr = _run_progress_child(**options)
        self.assertEqual(returncode, 0, (stdout + stderr).decode(errors="replace"))
        result = json.loads(stdout)
        scale = np.max(np.abs(self.expected_tensor))
        np.testing.assert_allclose(result["tensor"], self.expected_tensor,
                                   rtol=5e-10, atol=5e-10 * scale)
        np.testing.assert_allclose(complex(*result["coherent"]),
                                   self.expected_coherent, rtol=5e-10, atol=0)
        if options["callback"]:
            for operation in ("full_tensor", "coherent_amplitude"):
                events = [event for event in result["events"]
                          if event["operation"] == operation]
                self.assertGreaterEqual(len(events), 2)
                self.assertEqual(events[0]["status"], "started")
                self.assertEqual(events[-1]["status"], "completed")
                self.assertGreater(events[-1]["total"], 0)
                self.assertEqual(events[-1]["completed"], events[-1]["total"])
                for event in events:
                    self.assertEqual(event["backend"], _PROGRESS_BACKEND.name)
                    self.assertGreaterEqual(event["attempt"], 1)
                    self.assertLessEqual(event["completed"], event["total"])
                    self.assertTrue(np.isfinite(event["elapsed"]))
        else:
            self.assertEqual(result["events"], [])
        return stderr.decode("utf-8")

    def test_callback_and_display_switch_are_independent(self):
        for display in (False, True):
            for callback in (False, True):
                with self.subTest(display=display, callback=callback):
                    output = self.computation(display=display, callback=callback)
                    if not display:
                        self.assertEqual(output, "")
                        continue
                    self.assertNotIn("\r", output)
                    self.assertNotIn("\x1b", output)
                    records = output.splitlines()
                    self.assertTrue(all(line.startswith("[sl2cfoam++: progress]")
                                        for line in records), output)
                    completions = [line for line in records if "status=complete " in line]
                    self.assertEqual(len(completions), 2, output)
                    self.assertTrue(all("percent=100.0" in line for line in completions))

    def test_persistent_terminal_bar_and_callbacks(self):
        for style in ("ASCII", "UTF8"):
            with self.subTest(style=style):
                output = self.computation(display=True, callback=True, terminal=True,
                                          verbosity="summary", style=style)
                self.assertIn("\r", output)
                self.assertNotIn("\x1b", output)
                self.assertIn("full tensor", output)
                self.assertIn("coherent amplitude", output)
                self.assertIn(_PROGRESS_BACKEND.name, output)
                self.assertGreaterEqual(output.count("100.0%"), 2, output)
                self.assertIn("ETA", output)
                self.assertIn("█" if style == "UTF8" else "#", output)

    def test_silent_terminal_bar_is_erased_after_success(self):
        output = self.computation(display=True, callback=False, terminal=True)
        self.assertIn("full tensor", output)
        self.assertIn("coherent amplitude", output)
        self.assertIn(_PROGRESS_BACKEND.name, output)
        self.assertIn("ETA", output)
        self.assertNotIn("\n", output)
        self.assertNotIn("100.0%", output)
        self.assertNotIn("\x1b", output)
        # The last physical frame contains only the spaces that erase it.
        self.assertTrue(output.endswith("\r"))
        self.assertTrue(output.split("\r")[-2].isspace(), repr(output))


def _progress_main(arguments):
    global _PROGRESS_BACKEND
    parser = argparse.ArgumentParser(
        description="Check native Python progress in pipes and terminals.")
    parser.add_argument("--backend", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--display", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--callback", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verbosity", default="silent", help=argparse.SUPPRESS)
    parser.add_argument("--style", default="ASCII", help=argparse.SUPPRESS)
    arguments = parser.parse_args(arguments)
    _PROGRESS_BACKEND = getattr(foam.Backend, arguments.backend)
    if arguments.worker:
        _progress_worker(arguments)
        return 0
    if (_PROGRESS_BACKEND == foam.Backend.GPU and
            not foam.backend_available(_PROGRESS_BACKEND)):
        reason = ("CUDA was not compiled" if not foam.build_info().cuda_backend_built
                  else "no usable CUDA device is visible")
        print(f"SKIP: Python GPU progress tests: {reason}")
        return 77
    try:
        master, slave = _open_progress_terminal()
        os.close(master)
        os.close(slave)
    except (ImportError, OSError) as error:
        print(f"SKIP: Python terminal progress tests: cannot allocate a PTY: {error}")
        return 77
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProgressTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "progress":
        sys.exit(_progress_main(sys.argv[2:]))
    unittest.main(defaultTest="PythonApiTest", verbosity=2)
