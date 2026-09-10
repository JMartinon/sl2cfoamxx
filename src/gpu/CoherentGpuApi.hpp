// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/Complex64.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

// CUDA-free boundary between the project adapter and coherent backend. There
// is deliberately no route selector: every invocation runs the production
// device-recurrence -> cuTENSOR X2/X3 -> real-DGEMM X4 -> exact-support tail.
namespace b4gpu {

// Preserve the backend-facing name while sharing the exact scalar type used by
// host-side coherent preparation. This prevents their transfer ABI from
// drifting independently.
using Complex64 = b4utils::Complex64;

template <class T, std::size_t Rank> struct DenseTensorView {
  const T *data;
  std::array<std::uint64_t, Rank> extents;
  std::array<std::uint64_t, Rank> strides;
};

template <class T, std::size_t Rank>
[[nodiscard]] constexpr DenseTensorView<T, Rank>
make_first_axis_contiguous_view(
    const T *data, const std::array<std::uint64_t, Rank> &extents) noexcept {
  DenseTensorView<T, Rank> view{data, extents, {}};
  std::uint64_t stride = 1;
  for (std::size_t axis = 0; axis < Rank; ++axis) {
    view.strides[axis] = stride;
    stride *= extents[axis];
  }
  return view;
}

using DenseComplexVectorView = DenseTensorView<Complex64, 1>;
using DenseIntVectorView = DenseTensorView<int, 1>;
using DenseComplexTensor4View = DenseTensorView<Complex64, 4>;

struct DenseCoherentDimensions {
  std::uint64_t a;
  std::uint64_t k2;
  std::uint64_t k3;
  std::uint64_t k4;
  std::uint64_t k5;
  std::uint64_t l23;
  std::uint64_t l24;
  std::uint64_t l25;
  std::uint64_t l34;
  std::uint64_t l35;
  std::uint64_t l45;
};

struct DenseCoherentRecurrenceProblem {
  DenseCoherentDimensions dimensions;

  int Dj12;
  int Dj13;
  int Dj14;
  int Dj15;
  int Dj23;
  int Dj24;
  int Dj25;
  int Dj34;
  int Dj35;
  int Dj45;

  DenseIntVectorView x;
  DenseIntVectorView i1;
  DenseIntVectorView k2;
  DenseIntVectorView k3;
  DenseIntVectorView k4;
  DenseIntVectorView k5;
};

// Canonical first-axis-contiguous layouts:
// B2 [k2,l25,l23,l24], B3 [k3,l23,l35,l34],
// B4 [k4,l45,l34,l24], B5 [k5,l25,l45,l35].
struct DenseCoherentBoosters {
  DenseCoherentDimensions dimensions;
  DenseComplexTensor4View B2;
  DenseComplexTensor4View B3;
  DenseComplexTensor4View B4;
  DenseComplexTensor4View B5;
};

// Slots are a memory/performance choice, not an algorithm selector: values
// 1..3 all execute exactly the same dependency graph.
using DenseContractionProgressBegin =
    void (*)(void *context, std::uint64_t total_units) noexcept;
using DenseContractionProgressAdvance =
    void (*)(void *context, std::uint64_t completed_units) noexcept;
using DenseContractionProgressReset = void (*)(void *context) noexcept;

// Mask views are valid only during the optional callback. No owning C++
// containers cross this host/device implementation boundary.
struct DenseContractionWorkerDiagnostics {
  int device;
  std::uint64_t thread_id;
  int cpu;
  const char *cpus_allowed_list;
  const char *mems_allowed_list;
  std::uint64_t x_slices;
  std::uint64_t q_tiles;
  std::uint64_t r_tile;
  std::uint64_t peak_device_bytes;
  double upload_seconds;
  double contraction_seconds;
};

using DenseContractionWorkerDiagnosticCallback =
    void (*)(const void *context,
             const DenseContractionWorkerDiagnostics &record) noexcept;

struct DenseContractionOptions {
  const int *device_ids;
  std::size_t device_count;
  double vram_fraction;
  std::uint64_t reserve_device_bytes;
  std::uint64_t max_device_bytes;
  std::uint64_t r_tile;
  std::uint64_t cutensor_workspace_bytes;
  bool cutensor_jit;
  std::uint32_t pipeline_slots;
  DenseContractionProgressBegin progress_begin;
  DenseContractionProgressAdvance progress_advance;
  DenseContractionProgressReset progress_reset;
  void *progress_context;
  DenseContractionWorkerDiagnosticCallback worker_diagnostic;
  const void *diagnostic_context;
};

[[nodiscard]] constexpr DenseContractionOptions
default_dense_contraction_options() noexcept {
  return DenseContractionOptions{nullptr,
                                 0,
                                 0.85,
                                 512ull * 1024ull * 1024ull,
                                 0,
                                 0,
                                 128ull * 1024ull * 1024ull,
                                 false,
                                 3,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr};
}

struct DenseContractionResult {
  Complex64 amplitude;
  std::uint64_t x_slices_completed;
  std::uint64_t q_tiles_completed;
  std::uint64_t peak_device_bytes;
  std::uint64_t r_tile_min;
  std::uint64_t r_tile_max;
  std::uint32_t devices_used;
  std::uint32_t pipeline_slots;
  double upload_seconds;
  double contraction_seconds;
  double total_seconds;
  std::uint32_t worker_records;
};

#if defined(HEAVY_COMPUTE_USE_CUDA)
[[nodiscard]] bool coherent_cuda_backend_available() noexcept;
[[nodiscard]] std::size_t coherent_cuda_device_count() noexcept;

[[nodiscard]] DenseContractionResult
contract_device_generated_coherent_multi_gpu(
    const DenseCoherentBoosters &boosters,
    const DenseComplexVectorView &c1_coeffs,
    const DenseCoherentRecurrenceProblem &problem,
    const DenseContractionOptions &options =
        default_dense_contraction_options());
#else
[[nodiscard]] inline bool coherent_cuda_backend_available() noexcept {
  return false;
}

[[nodiscard]] inline std::size_t coherent_cuda_device_count() noexcept {
  return 0;
}

[[nodiscard]] inline DenseContractionResult
contract_device_generated_coherent_multi_gpu(
    const DenseCoherentBoosters &, const DenseComplexVectorView &,
    const DenseCoherentRecurrenceProblem &,
    const DenseContractionOptions & = default_dense_contraction_options()) {
  throw std::runtime_error(
      "CUDA coherent backend was not built (configure with -DENABLE_CUDA=ON)");
}
#endif

static_assert(std::is_trivial_v<DenseCoherentDimensions> &&
              std::is_standard_layout_v<DenseCoherentDimensions>);
static_assert(std::is_trivial_v<DenseCoherentBoosters> &&
              std::is_standard_layout_v<DenseCoherentBoosters>);
static_assert(std::is_trivial_v<DenseCoherentRecurrenceProblem> &&
              std::is_standard_layout_v<DenseCoherentRecurrenceProblem>);
static_assert(std::is_trivial_v<DenseContractionOptions> &&
              std::is_standard_layout_v<DenseContractionOptions>);
static_assert(std::is_trivial_v<DenseContractionWorkerDiagnostics> &&
              std::is_standard_layout_v<DenseContractionWorkerDiagnostics>);
static_assert(std::is_trivial_v<DenseContractionResult> &&
              std::is_standard_layout_v<DenseContractionResult>);

} // namespace b4gpu
