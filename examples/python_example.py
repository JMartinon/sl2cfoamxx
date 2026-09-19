# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later


"""Basic sl2cfoam++ Python usage, with optional NumPy einsum contractions.

From the repository root, using the Python selected by CMake:
    PYTHONPATH=build-python/python python3 examples/python_example.py
    PYTHONPATH=build-python/python python3 examples/python_example.py --einsum
    PYTHONPATH=build-python/python python3 examples/python_example.py --backend GPU --einsum

For a longer visible progress bar, add --two-j 12 --Dl 6. A CUDA build and
compatible device are required for GPU; no extra einsum package is needed.
"""

import argparse
from pathlib import Path
import tempfile

import numpy as np
import sl2cfoamxx as sf


# Equilateral Euclidean boundary from coherent_boundary_example.cpp.
# Rows are (theta, phi) in radians, in canonical oriented-normal order.
ANGLES = np.array([
    [0.000000000000000,  0.000000000000000],  # n12
    [1.910633236249019,  0.000000000000000],  # n13
    [1.910633236249019,  2.094395102393196],  # n14
    [1.910633236249019, -2.094395102393196],  # n15
    [3.141592653589793,  0.000000000000000],  # n21
    [1.230959417340775, -1.823476581936975],  # n23
    [1.230959417340775,  0.270918520456220],  # n24
    [1.230959417340775,  2.365313622849416],  # n25
    [1.230959417340775,  3.141592653589793],  # n31
    [1.910633236249019,  1.318116071652818],  # n32
    [2.411864997362827, -1.435337066566786],  # n34
    [0.729727656226966, -0.388139515370189],  # n35
    [1.230959417340775, -1.047197551196598],  # n41
    [1.910633236249019, -2.870674133133573],  # n42
    [0.729727656226966,  1.706255587023007],  # n43
    [2.411864997362827,  0.659058035826409],  # n45
    [1.230959417340775,  1.047197551196598],  # n51
    [1.910633236249019, -0.776279030740378],  # n52
    [2.411864997362827,  2.753453138219605],  # n53
    [0.729727656226966, -2.482534617763384],  # n54
], dtype=np.float64)

# Each node's public coefficient vector follows its tensor recoupling basis.
NODE_SPIN_INDICES = ((0, 1, 2, 3), (4, 5, 6, 0), (7, 8, 1, 4),
                     (9, 2, 5, 7), (3, 6, 8, 9))
NODE_ANGLE_ROWS = ((0, 1, 2, 3), (5, 6, 7, 4), (10, 11, 8, 9),
                   (15, 12, 13, 14), (16, 17, 18, 19))


def nonnegative_integer(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def show_einsum(tensor, coefficients, coherent_amplitude):
    # FullTensor is accepted directly by NumPy; no .numpy() call is required.
    contracted = np.einsum("abcde,a,b,c,d,e->", tensor, *coefficients,
                           optimize=True)
    np.testing.assert_allclose(contracted, coherent_amplitude, rtol=5e-9, atol=5e-21)

    # Contract two rank-five tensors over four axes, leaving a small matrix.
    # A second computed tensor could be used here; clone() makes this example
    # quick and demonstrates that native FullTensor operands work on both sides.
    other = tensor.clone()
    matrix = np.einsum("abcde,fbcde->af", tensor, other, optimize=True)
    array = np.asarray(tensor)
    rows = array.reshape(array.shape[0], -1)
    reference = rows @ rows.T
    # Some off-diagonal entries nearly cancel to zero. Bound their absolute
    # roundoff relative to the matrix scale, not relative to those tiny entries.
    np.testing.assert_allclose(matrix, reference, rtol=5e-10,
                               atol=5e-14 * np.max(np.abs(reference)))
    print(f"einsum coherent contraction: {contracted:.16g}")
    print(f"einsum tensor/tensor contraction shape: {matrix.shape}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--backend", choices=("CPU", "GPU"), default="CPU",
                        help="backend for coherent and full-tensor computations")
    parser.add_argument("--two-j", type=nonnegative_integer, default=1,
                        help="doubled spin on all ten faces; 1 means physical j=1/2")
    parser.add_argument("--Dl", type=nonnegative_integer, default=2,
                        help="shell cutoff")
    parser.add_argument("--threads", type=nonnegative_integer, default=2,
                        help="host CPU threads; 0 selects automatic")
    parser.add_argument("--einsum", action="store_true",
                        help="also demonstrate coherent and tensor/tensor contractions")
    parser.add_argument("--quiet", action="store_true",
                        help="erase the native progress bar after success")
    args = parser.parse_args()
    backend = getattr(sf.Backend, args.backend)
    if not sf.backend_available(backend):
        reason = ("GPU support was not built; enable ENABLE_CUDA in CMake"
                  if not sf.build_info().cuda_backend_built
                  else "no compatible CUDA device is visible")
        parser.error(reason)

    with tempfile.TemporaryDirectory(prefix="sl2cfoamxx-python-") as directory:
        config = sf.default_config()
        config.backend = backend
        config.cpu_threads = args.threads
        config.progress_bar = True  # Applies to both coherent and full tensor.
        config.verbosity = sf.Verbosity.silent if args.quiet else sf.Verbosity.summary
        config.tensors.directory = Path(directory)
        config.tensors.storage = sf.TensorStorage.in_memory_sym
        config.tensors.reuse_existing = False
        # Explicitly choose rho=gamma*(j+1), matching the reference below.
        # The library default is gamma_j, meaning rho=gamma*j.
        config.simplicity_constraint = sf.SimplicityConstraint.gamma_j_p1
        config.coherent_amplitude.pruning_ratio = 0.0
        sf.configure(config)

        # Doubled spins follow j12,j13,j14,j15,j23,j24,j25,j34,j35,j45.
        spins = [args.two_j] * 10
        intertwiners = [0] * 5
        events = []
        tensor = sf.compute_full_tensor(
            spins, args.Dl, 1.2, backend=backend, progress=events.append)
        # Fixed-intertwiner evaluation always runs on CPU.
        amplitude = sf.compute_amplitude(spins, intertwiners, args.Dl, 1.2)
        indices = sf.indices_from_intertwiners(spins, intertwiners)
        np.testing.assert_allclose(tensor[indices], amplitude, rtol=5e-10, atol=0)

        closure = sf.check_closure(spins, ANGLES)
        if not closure.all_closed:
            raise ValueError("Boundary normals do not close")
        # Generate vectors in each tensor axis's recoupling basis. These can
        # also be replaced with your own finite complex vectors of the same sizes.
        coefficients = [sf.compute_coherent_coefficients(
            [spins[i] for i in spin_indices], ANGLES[list(angle_rows)])
            for spin_indices, angle_rows in zip(NODE_SPIN_INDICES, NODE_ANGLE_ROWS)]
        C1 = coefficients[0]
        coherent = sf.compute_coherent_amplitude(
            spins, args.Dl, 1.2, ANGLES, backend=backend, progress=events.append)
        supplied = sf.compute_coherent_amplitude(
            spins, args.Dl, 1.2, *coefficients, backend=backend, progress=events.append)
        np.testing.assert_allclose(supplied, coherent, rtol=5e-9, atol=5e-21)
        if args.two_j == 1 and args.Dl == 2:
            # Independent Julia reference retained in test_coherent_public_api.cpp.
            expected = complex(-1.6748256353857123e-10, 1.2973143587300214e-10)
            np.testing.assert_allclose(coherent, expected, rtol=5e-9, atol=5e-21)

        # np.asarray produces a native ndarray for slicing and arithmetic.
        # Dense tensors share memory; symmetry-compressed tensors expand.
        array = np.asarray(tensor)
        first_slice = array[0, :, :, :, :]
        scaled = 2.0 * array
        np.testing.assert_allclose(scaled[indices], 2.0 * amplitude, rtol=5e-10, atol=0)
        # Explicit writable zero-copy view of an owned dense tensor:
        owned = tensor.dense_copy()
        view = owned.numpy()
        view[indices] = 0.0
        assert owned[indices] == 0.0

        path = Path(directory) / "vertex.sl2cxx"
        tensor.save(path)
        loaded = sf.FullTensor.load(path)  # independent owned tensor
        mapped = sf.FullTensor.map(path)  # read-only file mapping
        np.testing.assert_array_equal(np.asarray(loaded), array)
        np.testing.assert_array_equal(np.asarray(mapped), array)

        print(f"sl2cfoam++ {sf.version()}; backend: {args.backend}")
        print(f"Fixed amplitude: {amplitude:.16g}; coherent amplitude: {coherent:.16g}")
        print(f"Closure passed: {closure.all_closed}; C1: {C1}")
        print(f"Tensor shape: {tensor.shape}; stored values: {tensor.stored_size}")
        print(f"NumPy dtype: {array.dtype}; first slice shape: {first_slice.shape}")
        print(f"Progress events: {len(events)}; mapped read-only: {mapped.is_read_only}")
        if args.einsum:
            show_einsum(tensor, coefficients, coherent)


if __name__ == "__main__":
    main()
