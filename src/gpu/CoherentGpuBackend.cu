// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "CoherentGpuApi.hpp"

#include "core/Failure.hpp"
#include "CoherentGpuCuTensor.hpp"
#include "CoherentGpuKernels.cuh"
#include "CoherentGpuRecurrence.cuh"
#include "CoherentGpuSupportKernels.cuh"
#include "CudaBuffer.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fstream>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace b4gpu {
namespace cuda_detail {

using Clock = std::chrono::steady_clock;
using coherent_cutensor_detail::BinaryPlan;
using coherent_cutensor_detail::Context;
using coherent_cutensor_detail::TensorSpec;

constexpr int maximum_safe_doubled_spin = std::numeric_limits<int>::max() - 2;
constexpr std::size_t maximum_pipeline_slots = 3;
constexpr std::uint64_t q_slab_fused_b_tile = 128;
constexpr std::uint64_t q_slab_fused_reduce_blocks = 1024;

inline void check_cublas(cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) +
                             " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

class CudaEvent {
public:
  explicit CudaEvent(unsigned flags = cudaEventDisableTiming) {
    check_cuda(cudaEventCreateWithFlags(&event_, flags),
               "cudaEventCreateWithFlags coherent pipeline");
  }
  ~CudaEvent() {
    if (event_ != nullptr)
      (void)cudaEventDestroy(event_);
  }
  CudaEvent(const CudaEvent &) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;
  [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

private:
  cudaEvent_t event_ = nullptr;
};

inline std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b,
                                 const char *what) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
    throw std::overflow_error(std::string("size overflow for ") + what);
  }
  return a * b;
}

inline std::uint64_t checked_add(std::uint64_t a, std::uint64_t b,
                                 const char *what) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) {
    throw std::overflow_error(std::string("size overflow for ") + what);
  }
  return a + b;
}

template <class... Ts>
inline std::uint64_t product(const char *what, Ts... values) {
  std::uint64_t result = 1;
  ((result = checked_mul(result, static_cast<std::uint64_t>(values), what)),
   ...);
  return result;
}

inline std::size_t as_size(std::uint64_t value, const char *what) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error(std::string(what) + " exceeds size_t");
  }
  return static_cast<std::size_t>(value);
}

inline int as_int(std::uint64_t value, const char *what) {
  if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string(what) + " exceeds int");
  }
  return static_cast<int>(value);
}

inline long long as_blas_stride(std::uint64_t value, const char *what) {
  if (value >
      static_cast<std::uint64_t>(std::numeric_limits<long long>::max())) {
    throw std::overflow_error(std::string(what) +
                              " exceeds cuBLAS stride range");
  }
  return static_cast<long long>(value);
}

inline unsigned launch_blocks(std::uint64_t work, unsigned threads = 256) {
  if (work == 0)
    return 1;
  return static_cast<unsigned>(
      std::min<std::uint64_t>(65535, (work + threads - 1) / threads));
}

inline std::size_t pipeline_slots(const DenseContractionOptions &options) {
  if (options.pipeline_slots < 1 ||
      options.pipeline_slots > maximum_pipeline_slots) {
    throw std::invalid_argument("coherent pipeline_slots must be in [1,3]");
  }
  return static_cast<std::size_t>(options.pipeline_slots);
}

template <class T, std::size_t Rank>
inline std::uint64_t view_count(const DenseTensorView<T, Rank> &view,
                                const char *what) {
  std::uint64_t count = 1;
  for (const auto extent : view.extents) {
    count = checked_mul(count, extent, what);
  }
  return count;
}

template <class T, std::size_t Rank>
inline bool is_first_axis_contiguous(const DenseTensorView<T, Rank> &view) {
  std::uint64_t stride = 1;
  for (std::size_t axis = 0; axis < Rank; ++axis) {
    if (view.strides[axis] != stride)
      return false;
    stride = checked_mul(stride, view.extents[axis], "view stride");
  }
  return true;
}

template <class T, std::size_t Rank>
inline void validate_view(const DenseTensorView<T, Rank> &view,
                          const std::array<std::uint64_t, Rank> &expected,
                          const char *name) {
  if (view.extents != expected) {
    throw std::invalid_argument(std::string(name) + " has unexpected extents");
  }
  if (view_count(view, name) != 0 && view.data == nullptr) {
    throw std::invalid_argument(std::string(name) + " has null storage");
  }
  for (std::size_t axis = 0; axis < Rank; ++axis) {
    if (view.strides[axis] == 0) {
      throw std::invalid_argument(std::string(name) + " has a zero stride");
    }
  }
}

inline int vector_value(const DenseIntVectorView &view, std::uint64_t index) {
  return view.data[index * view.strides[0]];
}

inline void validate_nonnegative_vector(const DenseIntVectorView &view,
                                        const char *name) {
  for (std::uint64_t i = 0; i < view.extents[0]; ++i) {
    if (vector_value(view, i) < 0) {
      throw std::invalid_argument(std::string(name) +
                                  " contains a negative doubled spin");
    }
  }
}

inline void validate_regular_step_two_vector(const DenseIntVectorView &view,
                                             const char *name) {
  validate_nonnegative_vector(view, name);
  for (std::uint64_t i = 1; i < view.extents[0]; ++i) {
    if (vector_value(view, i) != vector_value(view, i - 1) + 2) {
      throw std::invalid_argument(std::string(name) +
                                  " must be a contiguous step-two axis");
    }
  }
}

inline int checked_spin_sum(int lhs, int rhs, const char *what) {
  if (lhs < 0 || rhs < 0 || lhs > maximum_safe_doubled_spin - rhs) {
    throw std::overflow_error(std::string("doubled-spin overflow for ") + what);
  }
  return lhs + rhs;
}

inline int lifted_axis_last(int base, std::uint64_t extent, const char *what) {
  if (base < 0 || extent == 0 ||
      extent - 1 >
          static_cast<std::uint64_t>((maximum_safe_doubled_spin - base) / 2)) {
    throw std::overflow_error(std::string("lifted-spin overflow for ") + what);
  }
  return base + 2 * static_cast<int>(extent - 1);
}

template <class T, std::size_t Rank>
std::vector<T> pack_view(const DenseTensorView<T, Rank> &view) {
  const std::size_t count =
      as_size(view_count(view, "packed view"), "packed view");
  std::vector<T> packed(count);
  for (std::size_t linear = 0; linear < count; ++linear) {
    std::uint64_t remaining = linear;
    std::uint64_t source = 0;
    for (std::size_t axis = 0; axis < Rank; ++axis) {
      const std::uint64_t coordinate = remaining % view.extents[axis];
      remaining /= view.extents[axis];
      source += coordinate * view.strides[axis];
    }
    packed[linear] = view.data[source];
  }
  return packed;
}

template <class T, std::size_t Rank>
void copy_view_to_device(const DenseTensorView<T, Rank> &view,
                         DeviceBuffer<T> &destination, cudaStream_t stream) {
  const std::size_t count =
      as_size(view_count(view, "tensor copy"), "tensor copy");
  if (destination.size() != count)
    destination.allocate(count);
  if (count == 0)
    return;
  if (is_first_axis_contiguous(view)) {
    check_cuda(cudaMemcpyAsync(destination.get(), view.data, count * sizeof(T),
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync coherent tensor H2D");
  } else {
    const std::vector<T> packed = pack_view(view);
    check_cuda(cudaMemcpyAsync(destination.get(), packed.data(),
                               count * sizeof(T), cudaMemcpyHostToDevice,
                               stream),
               "cudaMemcpyAsync packed coherent tensor H2D");
    check_cuda(cudaStreamSynchronize(stream),
               "cudaStreamSynchronize packed coherent tensor");
  }
}

struct DeviceStats {
  std::uint64_t peak_bytes = 0;
  std::uint64_t r_tile = 0;
  std::uint64_t x_slices = 0;
  std::uint64_t q_tiles = 0;
  std::uint64_t cutensor_workspace_bytes_max = 0;
  double upload_seconds = 0.0;
  double contraction_seconds = 0.0;
};

// Called only for requested diagnostics, after the CUDA worker finishes.
// Reading its own status distinguishes the launch thread from OpenMP workers.
// A missing procfs or allocation failure must not affect the numerical call.
void report_worker_diagnostics(int device, const DeviceStats &stats,
                               const DenseContractionOptions &options) noexcept {
  DenseContractionWorkerDiagnostics record{};
  record.device = device;
  record.cpu = -1;
  record.x_slices = stats.x_slices;
  record.q_tiles = stats.q_tiles;
  record.r_tile = stats.r_tile;
  record.peak_device_bytes = stats.peak_bytes;
  record.upload_seconds = stats.upload_seconds;
  record.contraction_seconds = stats.contraction_seconds;
  std::string cpus, mems;
#if defined(__linux__)
  const long tid = syscall(SYS_gettid);
  if (tid > 0) record.thread_id = static_cast<std::uint64_t>(tid);
  record.cpu = sched_getcpu();
  try {
    std::ifstream status("/proc/thread-self/status");
    for (std::string line; std::getline(status, line);) {
      const auto copy_mask = [&](const char *key, std::string &destination) {
        const std::string_view prefix(key);
        if (line.compare(0, prefix.size(), prefix) != 0) return;
        const auto begin = line.find_first_not_of(" \t", prefix.size());
        if (begin != std::string::npos) destination = line.substr(begin);
      };
      copy_mask("Cpus_allowed_list:", cpus);
      copy_mask("Mems_allowed_list:", mems);
    }
  } catch (...) {
    // Placement reporting is optional and best-effort.
  }
#endif
  record.cpus_allowed_list = cpus.empty() ? "unavailable" : cpus.c_str();
  record.mems_allowed_list = mems.empty() ? "unavailable" : mems.c_str();
  options.worker_diagnostic(options.diagnostic_context, record);
}

cudaError_t q_slab_tail_occupancy(int *blocks_per_sm) noexcept {
  if (blocks_per_sm == nullptr)
    return cudaErrorInvalidValue;
  constexpr std::size_t shared_bytes =
      2 * 32 * fused_q_slab_t_tile * sizeof(double);
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      blocks_per_sm, support_q_slab_tail_partials_kernel, 32, shared_bytes);
}

struct QSlabSupportTailArguments {
  const double *x4_re = nullptr;
  const double *x4_im = nullptr;
  const double *booster5_re = nullptr;
  const double *booster5_im = nullptr;
  const double *sjB_packed = nullptr;
  const cuDoubleComplex *k4 = nullptr;
  cuDoubleComplex *partials = nullptr;
  const SupportIndexRange *i_ranges_by_z = nullptr;
  const SupportIndexRange *j_ranges_by_z = nullptr;
  const SupportIndexRange *p_ranges_by_v = nullptr;
  const std::uint8_t *v_x_active = nullptr;
  int Dk4_first = 0;
  int Dk5_first = 0;
  int Dj14 = 0;
  int Dj35 = 0;
  int Dj45 = 0;
  std::uint64_t q = 0;
  std::uint64_t r_begin = 0;
  std::uint64_t Rt = 0;
  std::uint64_t V = 0;
  std::uint64_t J = 0;
  std::uint64_t Rdim = 0;
  std::uint64_t P = 0;
  std::uint64_t S = 0;
  std::uint64_t T = 0;
  std::uint64_t x4_z_pitch = 0;
  std::uint64_t x4_v_pitch = 0;
  std::uint64_t x4_p_pitch = 0;
};

cudaError_t launch_q_slab_support_tail(const QSlabSupportTailArguments &a,
                                       cudaStream_t stream) noexcept {
  if (a.x4_re == nullptr || a.x4_im == nullptr || a.booster5_re == nullptr ||
      a.booster5_im == nullptr || a.sjB_packed == nullptr || a.k4 == nullptr ||
      a.partials == nullptr || a.i_ranges_by_z == nullptr ||
      a.j_ranges_by_z == nullptr || a.p_ranges_by_v == nullptr ||
      a.v_x_active == nullptr || a.Rt == 0 || a.V == 0 || a.J == 0 ||
      a.Rdim == 0 || a.P == 0 || a.S == 0 || a.T == 0 || a.x4_z_pitch == 0 ||
      a.x4_v_pitch == 0 || a.x4_p_pitch == 0) {
    return cudaErrorInvalidValue;
  }
  const std::uint64_t JR = a.J * a.Rt;
  const std::uint64_t SV = a.S * a.V;
  const std::uint64_t b_tiles =
      (SV + q_slab_fused_b_tile - 1) / q_slab_fused_b_tile;
  const dim3 grid(static_cast<unsigned>((JR + 31) / 32),
                  static_cast<unsigned>(b_tiles),
                  static_cast<unsigned>((a.T + fused_q_slab_t_tile - 1) /
                                        fused_q_slab_t_tile));
  support_q_slab_tail_partials_kernel<<<
      grid, dim3(32, 1, 1), 2 * 32 * fused_q_slab_t_tile * sizeof(double),
      stream>>>(a.x4_re, a.x4_im, a.booster5_re, a.booster5_im, a.sjB_packed, a.k4,
                a.partials, a.i_ranges_by_z, a.j_ranges_by_z, a.p_ranges_by_v,
                a.v_x_active, a.Dk4_first, a.Dk5_first, a.Dj14, a.Dj35, a.Dj45,
                a.q, a.r_begin, a.Rt, a.V, a.J, a.Rdim, a.P, a.S, a.T,
                a.x4_z_pitch, a.x4_v_pitch, a.x4_p_pitch, q_slab_fused_b_tile);
  return cudaGetLastError();
}

struct QSlabShape {
  std::uint64_t Rt = 0;
  std::uint64_t VP = 0;
  std::uint64_t UI = 0;
  std::uint64_t Z = 0;
  std::uint64_t JR = 0;
  std::uint64_t SV = 0;
  std::uint64_t r_tiles = 0;
  std::uint64_t nB5 = 0;
  std::uint64_t nH = 0;
  std::uint64_t nX2 = 0;
  std::uint64_t nX3 = 0;
  std::uint64_t nX4 = 0;
  std::uint64_t nC = 0;
  std::uint64_t nPartials = 0;
  std::uint64_t nBlockSums = 0;
  std::uint64_t nOutputs = 0;
  std::uint64_t nSjBpacked = 0;
  std::uint64_t nSupportV = 0;
};

inline QSlabShape make_q_slab_shape(const DenseCoherentDimensions &d,
                                    std::uint64_t Rt) {
  if (Rt == 0 || Rt > d.l34) {
    throw std::invalid_argument("q-slab l34 tile is outside the domain");
  }
  QSlabShape s;
  s.Rt = Rt;
  s.VP = product("q-slab V*P", d.k5, d.l25);
  s.UI = product("q-slab U*I", d.l23, d.k3);
  s.Z = product("q-slab Rt*S", Rt, d.l35);
  s.JR = product("q-slab J*Rt", d.k4, Rt);
  s.SV = product("q-slab S*V", d.l35, d.k5);
  s.r_tiles = 1 + (d.l34 - 1) / Rt;
  s.nB5 = product("q-slab B5 pack", d.k5, d.l25, d.l45, d.l35);
  s.nH = product("q-slab H", d.k2, d.l23, d.k3, d.l25);
  s.nX2 = product("q-slab X2", d.k5, d.l25, d.l23, d.k3);
  s.nX3 = product("q-slab X3", s.VP, d.k3, s.Z);
  s.nX4 = product("q-slab X4", d.k4, s.Z, s.VP);
  s.nC = product("q-slab C", d.k4, d.k3, s.Z);
  const std::uint64_t b_tiles = 1 + (s.SV - 1) / q_slab_fused_b_tile;
  const std::uint64_t jr_tiles = 1 + (s.JR - 1) / 32;
  const std::uint64_t t_tiles = 1 + (d.l45 - 1) / fused_q_slab_t_tile;
  s.nPartials =
      product("q-slab fused CTA partials", jr_tiles, b_tiles, t_tiles);
  s.nBlockSums = std::min(q_slab_fused_reduce_blocks, s.nPartials);
  s.nOutputs = product("q-slab outputs", d.l24, s.r_tiles);
  s.nSjBpacked = product("q-slab packed sjB", d.k4, s.SV, d.l45);
  s.nSupportV = d.k5;
  return s;
}

inline bool q_slab_shape_fits_blas(const QSlabShape &s,
                                   const DenseCoherentDimensions &d) {
  constexpr std::uint64_t int_max =
      static_cast<std::uint64_t>(std::numeric_limits<int>::max());
  constexpr std::uint64_t stride_max =
      static_cast<std::uint64_t>(std::numeric_limits<long long>::max());
  const std::uint64_t values[] = {
      s.VP,  s.UI,
      s.Z,   s.JR,
      s.SV,  d.k2,
      d.k3,  d.k4,
      d.k5,  d.l23,
      d.l25, d.l35,
      d.l45, checked_mul(2, s.VP, "hybrid X4 doubled columns")};
  for (const std::uint64_t value : values) {
    if (value > int_max)
      return false;
  }
  const std::uint64_t strides[] = {
      product("X3 A stride", s.VP, d.l23), product("X3 B stride", d.l23, s.Z),
      product("hybrid X4 B stride", 2, s.VP, d.k3),
      product("hybrid X4 C stride", 2, d.k4, s.VP)};
  for (const std::uint64_t value : strides) {
    if (value > stride_max)
      return false;
  }
  const std::uint64_t b_tiles = 1 + (s.SV - 1) / q_slab_fused_b_tile;
  const std::uint64_t t_tiles = 1 + (d.l45 - 1) / fused_q_slab_t_tile;
  return b_tiles <= 65535 && t_tiles <= 65535;
}

inline std::uint64_t q_slab_peak_bytes(std::uint64_t persistent_base_bytes,
                                       const DenseCoherentDimensions &d,
                                       const QSlabShape &s,
                                       const DenseContractionOptions &options) {
  const std::uint64_t slots = pipeline_slots(options);
  const std::uint64_t booster5_bytes =
      checked_mul(s.nB5, sizeof(cuDoubleComplex), "q-slab B5 pack bytes");
  std::uint64_t dynamic = booster5_bytes;
  dynamic =
      checked_add(dynamic,
                  checked_mul(checked_add(checked_add(s.nH, s.nX2, "H+X2"),
                                          s.nX3, "H+X2+X3"),
                              sizeof(cuDoubleComplex), "q-slab middle bytes"),
                  "q-slab dynamic bytes");
  const std::uint64_t per_slot_complex =
      checked_add(checked_add(s.nX4, s.nPartials, "slot X4+partials"),
                  s.nBlockSums, "slot complex values");
  dynamic = checked_add(
      dynamic,
      checked_mul(checked_mul(slots, per_slot_complex, "slot complex values"),
                  sizeof(cuDoubleComplex), "slot complex bytes"),
      "q-slab dynamic bytes");
  dynamic = checked_add(
      dynamic,
      checked_mul(checked_add(checked_mul(slots, s.nC, "C slot values"),
                              s.nSjBpacked, "real q-slab values"),
                  sizeof(double), "real q-slab bytes"),
      "q-slab dynamic bytes");
  const std::uint64_t range_count =
      checked_add(checked_mul(checked_mul(2, slots, "slot range families"), s.Z,
                              "slot range values"),
                  d.k5, "support range values");
  dynamic = checked_add(
      dynamic,
      checked_add(
          checked_mul(range_count, sizeof(SupportIndexRange),
                      "support range bytes"),
          checked_mul(d.k5, sizeof(std::uint8_t), "support active bytes"),
          "support descriptor bytes"),
      "q-slab dynamic bytes");
  dynamic = checked_add(
      dynamic,
      checked_mul(s.nOutputs, sizeof(cuDoubleComplex), "q-slab output bytes"),
      "q-slab dynamic bytes");
  dynamic = checked_add(dynamic, options.cutensor_workspace_bytes,
                        "q-slab cuTENSOR workspace cap");

  const std::uint64_t transition_peak =
      checked_add(persistent_base_bytes, booster5_bytes, "q-slab B5 transition peak");
  if (booster5_bytes > persistent_base_bytes) {
    throw std::logic_error("B5 release exceeds persistent storage");
  }
  const std::uint64_t steady_peak = checked_add(
      persistent_base_bytes - booster5_bytes, dynamic, "q-slab steady peak");
  return std::max(transition_peak, steady_peak);
}

class BlasHandle {
public:
  BlasHandle() { check_cublas(cublasCreate(&handle_), "cublasCreate"); }
  ~BlasHandle() {
    if (handle_ != nullptr)
      (void)cublasDestroy(handle_);
  }
  BlasHandle(const BlasHandle &) = delete;
  BlasHandle &operator=(const BlasHandle &) = delete;
  [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }

private:
  cublasHandle_t handle_ = nullptr;
};

class DeviceContraction {
public:
  DeviceContraction(int device, const DenseCoherentBoosters &boosters,
                    const DenseComplexVectorView &c1,
                    const DenseCoherentRecurrenceProblem &recurrence,
                    const DenseContractionOptions &options);

  ~DeviceContraction() { (void)cudaSetDevice(device_); }
  DeviceContraction(const DeviceContraction &) = delete;
  DeviceContraction &operator=(const DeviceContraction &) = delete;

  [[nodiscard]] Complex64 contract_generated_slice(std::size_t x_index);
  [[nodiscard]] DeviceStats stats();

private:
  struct TilePlans {
    std::uint64_t Rt = 0;
    std::unique_ptr<BinaryPlan> x3;
  };

  struct QSlabWorkspace {
    DeviceBuffer<double> B5packed;
    DeviceBuffer<double> sjBpacked;
    DeviceBuffer<cuDoubleComplex> H;
    DeviceBuffer<cuDoubleComplex> X2;
    DeviceBuffer<cuDoubleComplex> X3;
    std::array<DeviceBuffer<double>, maximum_pipeline_slots> C;
    std::array<DeviceBuffer<cuDoubleComplex>, maximum_pipeline_slots> X4;
    std::array<DeviceBuffer<SupportIndexRange>, maximum_pipeline_slots>
        i_ranges;
    std::array<DeviceBuffer<SupportIndexRange>, maximum_pipeline_slots>
        j_ranges;
    DeviceBuffer<SupportIndexRange> p_ranges;
    DeviceBuffer<std::uint8_t> v_active;
    std::array<DeviceBuffer<cuDoubleComplex>, maximum_pipeline_slots> partials;
    std::array<DeviceBuffer<cuDoubleComplex>, maximum_pipeline_slots>
        block_sums;
    DeviceBuffer<cuDoubleComplex> outputs;
    DeviceBuffer<std::uint8_t> cutensor_workspace;

    explicit QSlabWorkspace(const QSlabShape &shape)
        : B5packed(as_size(checked_mul(2, shape.nB5, "B5 split planes"),
                           "B5 split planes")) {}

    void allocate_steady(const QSlabShape &shape, std::size_t slots,
                         std::uint64_t workspace_bytes) {
      sjBpacked.allocate(as_size(shape.nSjBpacked, "packed sjB"));
      H.allocate(as_size(shape.nH, "q-slab H"));
      X2.allocate(as_size(shape.nX2, "q-slab X2"));
      X3.allocate(as_size(shape.nX3, "q-slab X3"));
      for (std::size_t slot = 0; slot < slots; ++slot) {
        C[slot].allocate(as_size(shape.nC, "q-slab C slot"));
        X4[slot].allocate(as_size(shape.nX4, "q-slab X4 slot"));
        i_ranges[slot].allocate(as_size(shape.Z, "support i ranges"));
        j_ranges[slot].allocate(as_size(shape.Z, "support j ranges"));
        partials[slot].allocate(as_size(shape.nPartials, "tail partials"));
        block_sums[slot].allocate(as_size(shape.nBlockSums, "tail block sums"));
      }
      p_ranges.allocate(as_size(shape.nSupportV, "support p ranges"));
      v_active.allocate(as_size(shape.nSupportV, "support active flags"));
      outputs.allocate(as_size(shape.nOutputs, "q-slab outputs"));
      cutensor_workspace.allocate(
          as_size(workspace_bytes, "cuTENSOR workspace"));
    }
  };

  void validate_dimensions() const;
  void validate_boosters(const DenseCoherentBoosters &boosters) const;
  [[nodiscard]] std::uint64_t
  prepare_recurrence_problem(const DenseCoherentRecurrenceProblem &problem);
  void
  initialize_device_recurrence(const DenseCoherentRecurrenceProblem &problem);
  void initialize_plans();
  void initialize_workspace();

  [[nodiscard]] TensorSpec cutensor_k_spec() const;
  [[nodiscard]] TensorSpec cutensor_h_spec() const;
  [[nodiscard]] TensorSpec cutensor_x2_spec() const;
  [[nodiscard]] TensorSpec cutensor_k3_tile_spec(std::uint64_t Rt) const;
  [[nodiscard]] TensorSpec cutensor_x3_spec(std::uint64_t Rt) const;
  [[nodiscard]] TilePlans &tile_plans(std::uint64_t Rt);
  void add_plan_metrics(const coherent_cutensor_detail::PlanMetrics &metrics);

  template <std::size_t Rank>
  void copy_complex_booster(const DenseTensorView<Complex64, Rank> &source,
                            DeviceBuffer<cuDoubleComplex> &destination);
  void form_k();
  void run_x2(std::uint64_t q);
  void run_hybrid_x4(const double *c, const cuDoubleComplex *x3,
                     cuDoubleComplex *x4, std::uint64_t Rt);
  void enqueue_middle(std::uint64_t r_begin, std::uint64_t Rt, const double *c,
                      cuDoubleComplex *x4);
  void enqueue_tail(std::uint64_t q, std::uint64_t r_begin, std::uint64_t Rt,
                    const cuDoubleComplex *x4,
                    cuDoubleComplex *output, std::size_t slot);
  [[nodiscard]] Complex64 run_pipeline(std::size_t x_index);

  static double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
  }

  int device_ = 0;
  DenseCoherentDimensions dims_{};
  DenseContractionOptions options_{};
  std::size_t slots_ = 0;
  std::uint64_t budget_bytes_ = 0;
  std::uint64_t fixed_bytes_ = 0;
  std::uint64_t persistent_base_bytes_ = 0;
  DeviceStats stats_{};

  std::unique_ptr<Stream> stream_;
  std::unique_ptr<Stream> recurrence_stream_;
  std::unique_ptr<Stream> middle_stream_;
  std::unique_ptr<Stream> tail_stream_;
  std::unique_ptr<BlasHandle> blas_;
  std::unique_ptr<BlasHandle> hybrid_blas_;
  std::unique_ptr<Context> cutensor_;
  std::unique_ptr<BinaryPlan> x2_plan_;
  std::vector<TilePlans> tile_plans_;
  std::array<std::unique_ptr<CudaEvent>, maximum_pipeline_slots> c_ready_;
  std::array<std::unique_ptr<CudaEvent>, maximum_pipeline_slots> x4_ready_;
  std::array<std::unique_ptr<CudaEvent>, maximum_pipeline_slots> slot_free_;
  std::unique_ptr<CudaEvent> x2_ready_;
  std::unique_ptr<CudaEvent> middle_q_done_;

  DeviceBuffer<cuDoubleComplex> B2_, B3_, B4_, B5_;
  DeviceBuffer<Complex64> c1_;
  DeviceBuffer<int> x_values_, i1_values_;
  DeviceBuffer<int> k2_values_, k3_values_, k4_values_, k5_values_;
  DeviceBuffer<std::uint64_t> recurrence_failures_;
  DeviceCoherentRecurrenceProblem device_recurrence_{};
  RecurrenceLaunchConfig wing_recurrence_config_{};
  RecurrenceLaunchConfig c_recurrence_config_{};
  std::vector<int> host_x_values_;
  int host_k3_first_ = 0;
  int host_k4_first_ = 0;
  int host_k5_first_ = 0;

  DeviceBuffer<double> sjA_, sjE_, sjB_, sjD_;
  DeviceBuffer<double> weighted_, K_re_, K_im_;
  DeviceBuffer<cuDoubleComplex> K_;
  std::unique_ptr<QSlabShape> shape_;
  std::unique_ptr<QSlabWorkspace> workspace_;
  std::vector<cuDoubleComplex> host_outputs_;
};

// Device-recurrence validation and launch planning.
DeviceContraction::DeviceContraction(
    int device, const DenseCoherentBoosters &boosters,
    const DenseComplexVectorView &c1,
    const DenseCoherentRecurrenceProblem &recurrence,
    const DenseContractionOptions &options)
    : device_(device), dims_(boosters.dimensions), options_(options),
      slots_(pipeline_slots(options)) {
  check_cuda(cudaSetDevice(device_), "cudaSetDevice coherent constructor");
  stream_ = std::make_unique<Stream>();
  blas_ = std::make_unique<BlasHandle>();
  check_cublas(cublasSetStream(blas_->get(), stream_->get()),
               "cublasSetStream coherent K");
  check_cublas(cublasSetMathMode(blas_->get(), CUBLAS_DEFAULT_MATH),
               "cublasSetMathMode coherent K");

  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  (void)total_bytes;
  if (!(options_.vram_fraction > 0.0 && options_.vram_fraction <= 1.0)) {
    throw std::invalid_argument("vram_fraction must be in (0,1]");
  }
  budget_bytes_ = static_cast<std::uint64_t>(
      static_cast<long double>(free_bytes) * options_.vram_fraction);
  if (options_.max_device_bytes != 0) {
    budget_bytes_ = std::min(budget_bytes_, options_.max_device_bytes);
  }
  if (budget_bytes_ <= options_.reserve_device_bytes) {
    throw b4utils::ResourceLimitFailure(
        "CUDA VRAM budget is smaller than the requested reserve");
  }
  budget_bytes_ -= options_.reserve_device_bytes;

  validate_dimensions();
  validate_boosters(boosters);
  validate_view(c1, std::array<std::uint64_t, 1>{dims_.a}, "C1");
  (void)as_int(product("fused K/P", dims_.k2, dims_.l25),
               "fused K/P cuBLAS dimension");

  const std::uint64_t booster_values =
      checked_add(checked_add(view_count(boosters.B2, "B2 fixed count"),
                              view_count(boosters.B3, "B3 fixed count"),
                              "fixed booster count"),
                  checked_add(view_count(boosters.B4, "B4 fixed count"),
                              view_count(boosters.B5, "B5 fixed count"),
                              "fixed booster count"),
                  "fixed booster count");
  fixed_bytes_ = checked_mul(booster_values, sizeof(cuDoubleComplex),
                             "fixed booster bytes");
  fixed_bytes_ = checked_add(fixed_bytes_,
                             checked_mul(view_count(c1, "C1 fixed count"),
                                         sizeof(Complex64), "C1 fixed bytes"),
                             "fixed coherent bytes");
  const std::uint64_t recurrence_input_bytes =
      prepare_recurrence_problem(recurrence);
  fixed_bytes_ =
      checked_add(fixed_bytes_,
                  checked_add(recurrence_input_bytes, sizeof(std::uint64_t),
                              "device recurrence state bytes"),
                  "fixed coherent bytes");
  if (fixed_bytes_ >= budget_bytes_) {
    throw b4utils::ResourceLimitFailure(
        "coherent fixed inputs consume the CUDA VRAM budget");
  }

  const std::uint64_t nA =
      product("persistent sjA", dims_.a, dims_.k5, dims_.l25);
  const std::uint64_t nE =
      product("persistent sjE", dims_.a, dims_.k2, dims_.l25);
  const std::uint64_t nB =
      product("persistent sjB", dims_.k5, dims_.k4, dims_.l35, dims_.l45);
  const std::uint64_t nD =
      product("persistent sjD", dims_.k3, dims_.k2, dims_.l23, dims_.l24);
  const std::uint64_t nK =
      product("persistent K", dims_.k5, dims_.k2, dims_.l25);
  persistent_base_bytes_ = checked_add(
      fixed_bytes_,
      checked_add(
          checked_mul(checked_add(checked_add(nA, nE, "wing values"),
                                  checked_add(nB, nD, "wing values"),
                                  "wing values"),
                      sizeof(double), "wing bytes"),
          checked_add(checked_mul(nA, sizeof(double), "weighted sjA"),
                      checked_mul(nK,
                                  2 * sizeof(double) + sizeof(cuDoubleComplex),
                                  "K buffers"),
                      "K workspace"),
          "one-X workspace"),
      "persistent coherent bytes");

  const std::uint64_t requested_rt =
      options_.r_tile == 0 ? dims_.l34 : std::min(options_.r_tile, dims_.l34);
  std::uint64_t best_rt = 0;
  std::uint64_t low = 1;
  std::uint64_t high = requested_rt;
  while (low <= high) {
    const std::uint64_t candidate_rt = low + (high - low) / 2;
    bool fits = false;
    try {
      const QSlabShape candidate = make_q_slab_shape(dims_, candidate_rt);
      fits = q_slab_shape_fits_blas(candidate, dims_) &&
             q_slab_peak_bytes(persistent_base_bytes_, dims_, candidate,
                               options_) <= budget_bytes_;
    } catch (const std::overflow_error &) {
      fits = false;
    }
    if (fits) {
      best_rt = candidate_rt;
      low = candidate_rt + 1;
    } else {
      high = candidate_rt - 1;
    }
  }
  if (best_rt == 0) {
    throw b4utils::ResourceLimitFailure(
        "the production coherent route cannot fit its minimum l34 tile");
  }
  shape_ = std::make_unique<QSlabShape>(make_q_slab_shape(dims_, best_rt));
  stats_.r_tile = best_rt;
  stats_.peak_bytes =
      q_slab_peak_bytes(persistent_base_bytes_, dims_, *shape_, options_);

  const auto upload_start = Clock::now();
  copy_complex_booster(boosters.B2, B2_);
  copy_complex_booster(boosters.B3, B3_);
  copy_complex_booster(boosters.B4, B4_);
  copy_complex_booster(boosters.B5, B5_);
  copy_view_to_device(c1, c1_, stream_->get());
  initialize_device_recurrence(recurrence);
  check_cuda(cudaStreamSynchronize(stream_->get()),
             "cudaStreamSynchronize coherent fixed inputs");
  stats_.upload_seconds += seconds_since(upload_start);

  initialize_plans();
  initialize_workspace();
}

void DeviceContraction::validate_dimensions() const {
  const std::uint64_t values[] = {dims_.a,   dims_.k2,  dims_.k3,  dims_.k4,
                                  dims_.k5,  dims_.l23, dims_.l24, dims_.l25,
                                  dims_.l34, dims_.l35, dims_.l45};
  for (const std::uint64_t value : values) {
    if (value == 0) {
      throw std::invalid_argument("coherent dimensions must all be nonzero");
    }
  }
}

void DeviceContraction::validate_boosters(
    const DenseCoherentBoosters &b) const {
  validate_view(
      b.B2,
      std::array<std::uint64_t, 4>{dims_.k2, dims_.l25, dims_.l23, dims_.l24},
      "B2");
  validate_view(
      b.B3,
      std::array<std::uint64_t, 4>{dims_.k3, dims_.l23, dims_.l35, dims_.l34},
      "B3");
  validate_view(
      b.B4,
      std::array<std::uint64_t, 4>{dims_.k4, dims_.l45, dims_.l34, dims_.l24},
      "B4");
  validate_view(
      b.B5,
      std::array<std::uint64_t, 4>{dims_.k5, dims_.l25, dims_.l45, dims_.l35},
      "B5");
}

[[nodiscard]] std::uint64_t DeviceContraction::prepare_recurrence_problem(
    const DenseCoherentRecurrenceProblem &problem) {
  const DenseCoherentDimensions &d = problem.dimensions;
  if (d.a != dims_.a || d.k2 != dims_.k2 || d.k3 != dims_.k3 ||
      d.k4 != dims_.k4 || d.k5 != dims_.k5 || d.l23 != dims_.l23 ||
      d.l24 != dims_.l24 || d.l25 != dims_.l25 || d.l34 != dims_.l34 ||
      d.l35 != dims_.l35 || d.l45 != dims_.l45) {
    throw std::invalid_argument(
        "device recurrence dimensions do not match dense boosters");
  }

  if (problem.x.extents[0] == 0) {
    throw std::invalid_argument("device recurrence X axis must be nonempty");
  }
  validate_view(problem.x, std::array<std::uint64_t, 1>{problem.x.extents[0]},
                "X");
  validate_view(problem.i1, std::array<std::uint64_t, 1>{dims_.a}, "i1");
  validate_view(problem.k2, std::array<std::uint64_t, 1>{dims_.k2}, "k2");
  validate_view(problem.k3, std::array<std::uint64_t, 1>{dims_.k3}, "k3");
  validate_view(problem.k4, std::array<std::uint64_t, 1>{dims_.k4}, "k4");
  validate_view(problem.k5, std::array<std::uint64_t, 1>{dims_.k5}, "k5");
  validate_nonnegative_vector(problem.x, "X");
  validate_nonnegative_vector(problem.i1, "i1");
  validate_regular_step_two_vector(problem.k2, "k2");
  validate_regular_step_two_vector(problem.k3, "k3");
  validate_regular_step_two_vector(problem.k4, "k4");
  validate_regular_step_two_vector(problem.k5, "k5");

  host_x_values_.resize(as_size(problem.x.extents[0], "host X axis"));
  for (std::uint64_t i = 0; i < problem.x.extents[0]; ++i) {
    host_x_values_[static_cast<std::size_t>(i)] = vector_value(problem.x, i);
  }
  host_k3_first_ = vector_value(problem.k3, 0);
  host_k4_first_ = vector_value(problem.k4, 0);
  host_k5_first_ = vector_value(problem.k5, 0);

  const int outer_spins[] = {
      problem.Dj12, problem.Dj13, problem.Dj14, problem.Dj15, problem.Dj23,
      problem.Dj24, problem.Dj25, problem.Dj34, problem.Dj35, problem.Dj45};
  for (const int spin : outer_spins) {
    if (spin < 0 || spin > maximum_safe_doubled_spin) {
      throw std::invalid_argument(
          "device recurrence outer doubled spin is outside the "
          "supported integer range");
    }
  }

  const int max_l23 = lifted_axis_last(problem.Dj23, dims_.l23, "l23");
  const int max_l24 = lifted_axis_last(problem.Dj24, dims_.l24, "l24");
  const int max_l25 = lifted_axis_last(problem.Dj25, dims_.l25, "l25");
  const int max_l34 = lifted_axis_last(problem.Dj34, dims_.l34, "l34");
  const int max_l35 = lifted_axis_last(problem.Dj35, dims_.l35, "l35");
  const int max_l45 = lifted_axis_last(problem.Dj45, dims_.l45, "l45");

  int max_x = 0;
  for (std::uint64_t i = 0; i < problem.x.extents[0]; ++i) {
    max_x = std::max(max_x, vector_value(problem.x, i));
  }
  // Prove that the integer expressions used by the device recurrence
  // cannot overflow, then size shared memory from the true longest
  // (Dmax-Dmin)/2+1 string rather than the much looser Dmax/2+1 bound.
  (void)checked_spin_sum(problem.Dj15, max_l25, "sjA recurrence");
  (void)checked_spin_sum(max_x, problem.Dj14, "sjA/sjB recurrence");
  (void)checked_spin_sum(max_l45, max_l35, "sjB recurrence");
  (void)checked_spin_sum(max_l34, max_l35, "sjC recurrence");
  (void)checked_spin_sum(max_x, max_l24, "sjC recurrence");
  (void)checked_spin_sum(max_l23, max_l24, "sjD recurrence");
  (void)checked_spin_sum(max_x, problem.Dj13, "sjD/sjE recurrence");
  (void)checked_spin_sum(problem.Dj12, max_l25, "sjE recurrence");

  const auto recurrence_length = [](int j2, int j3, int j5, int j6,
                                    const DenseIntVectorView &virtual_index) {
    if (((j2 + j3) & 1) != ((j5 + j6) & 1)) {
      return std::uint64_t{0};
    }
    const int Dmin = std::max(std::abs(j2 - j3), std::abs(j5 - j6));
    const int Dmax = std::min(j2 + j3, j5 + j6);
    if (Dmin > Dmax || (vector_value(virtual_index, 0) & 1) != (Dmin & 1)) {
      return std::uint64_t{0};
    }
    const int virtual_first = vector_value(virtual_index, 0);
    const int virtual_last = vector_value(virtual_index, virtual_index.extents[0] - 1);
    if (std::max(Dmin, virtual_first) > std::min(Dmax, virtual_last)) {
      return std::uint64_t{0};
    }
    return static_cast<std::uint64_t>((Dmax - Dmin) / 2) + 1;
  };
  const auto lifted = [](int base, std::uint64_t index) {
    return base + 2 * static_cast<int>(index);
  };

  const auto recurrence_length_bound = [](std::initializer_list<int> legs) {
    return static_cast<std::uint64_t>(
               *std::min_element(legs.begin(), legs.end())) +
           1;
  };
  std::uint64_t maximum_wing_length = std::max(
      {recurrence_length_bound({problem.Dj15, max_l25, max_x, problem.Dj14}),
       recurrence_length_bound({max_x, problem.Dj14, max_l45, max_l35}),
       recurrence_length_bound({max_l23, max_l24, max_x, problem.Dj13}),
       recurrence_length_bound({problem.Dj12, max_l25, max_x, problem.Dj13})});
  std::uint64_t maximum_c_length =
      recurrence_length_bound({max_l34, max_l35, max_x, max_l24});

  // For ordinary domains, refine the bound to the exact longest string.
  // For extremely large q/r/s domains, keep the O(1) four-leg bound so
  // recurrence planning itself never becomes a cubic host bottleneck.
  const std::uint64_t qrs_count =
      product("recurrence planning q/r/s", dims_.l24, dims_.l34, dims_.l35);
  const std::uint64_t planning_points =
      problem.x.extents[0] != 0 &&
              qrs_count > std::numeric_limits<std::uint64_t>::max() /
                              problem.x.extents[0]
          ? std::numeric_limits<std::uint64_t>::max()
          : qrs_count * problem.x.extents[0];
  constexpr std::uint64_t exact_planning_limit = 8'000'000;
  if (planning_points <= exact_planning_limit) {
    maximum_wing_length = 1;
    maximum_c_length = 1;
    for (std::uint64_t xi = 0; xi < problem.x.extents[0]; ++xi) {
      const int x = vector_value(problem.x, xi);
      for (std::uint64_t p = 0; p < dims_.l25; ++p) {
        const int l25 = lifted(problem.Dj25, p);
        maximum_wing_length = std::max(
            maximum_wing_length,
            recurrence_length(problem.Dj15, l25, x, problem.Dj14, problem.k5));
        maximum_wing_length = std::max(
            maximum_wing_length,
            recurrence_length(problem.Dj12, l25, x, problem.Dj13, problem.k2));
      }
      for (std::uint64_t s = 0; s < dims_.l35; ++s) {
        const int l35 = lifted(problem.Dj35, s);
        for (std::uint64_t t = 0; t < dims_.l45; ++t) {
          maximum_wing_length = std::max(
              maximum_wing_length,
              recurrence_length(x, problem.Dj14, lifted(problem.Dj45, t), l35,
                                problem.k5));
        }
      }
      for (std::uint64_t q = 0; q < dims_.l24; ++q) {
        const int l24 = lifted(problem.Dj24, q);
        for (std::uint64_t u = 0; u < dims_.l23; ++u) {
          maximum_wing_length =
              std::max(maximum_wing_length,
                       recurrence_length(lifted(problem.Dj23, u), l24, x,
                                         problem.Dj13, problem.k2));
        }
        for (std::uint64_t r = 0; r < dims_.l34; ++r) {
          const int l34 = lifted(problem.Dj34, r);
          for (std::uint64_t s = 0; s < dims_.l35; ++s) {
            maximum_c_length =
                std::max(maximum_c_length,
                         recurrence_length(l34, lifted(problem.Dj35, s), x, l24,
                                           problem.k3));
          }
        }
      }
    }
  }

  if (maximum_wing_length > std::numeric_limits<std::uint32_t>::max() ||
      maximum_c_length > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error(
        "device recurrence string exceeds the CUDA index range");
  }
  const std::uint32_t maximum_wing_lanes = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(32, std::max({dims_.a, dims_.k3, dims_.k4})));
  const std::uint32_t maximum_c_lanes =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(32, dims_.k4));
  int wing_blocks_per_sm = 0;
  int c_blocks_per_sm = 0;
  check_cuda(select_coherent_recurrence_launch_config(
                 static_cast<std::uint32_t>(maximum_wing_length),
                 maximum_wing_lanes, RecurrenceKernelFamily::wings,
                 &wing_recurrence_config_, &wing_blocks_per_sm),
             "select coherent wing recurrence occupancy");
  check_cuda(select_coherent_recurrence_launch_config(
                 static_cast<std::uint32_t>(maximum_c_length), maximum_c_lanes,
                 RecurrenceKernelFamily::c_tile, &c_recurrence_config_,
                 &c_blocks_per_sm),
             "select coherent sjC recurrence occupancy");
  std::uint64_t input_count = view_count(problem.x, "X channel count");
  input_count =
      checked_add(input_count, view_count(problem.i1, "i1 channel count"),
                  "recurrence channel count");
  input_count =
      checked_add(input_count, view_count(problem.k2, "k2 channel count"),
                  "recurrence channel count");
  input_count =
      checked_add(input_count, view_count(problem.k3, "k3 channel count"),
                  "recurrence channel count");
  input_count =
      checked_add(input_count, view_count(problem.k4, "k4 channel count"),
                  "recurrence channel count");
  input_count =
      checked_add(input_count, view_count(problem.k5, "k5 channel count"),
                  "recurrence channel count");
  return checked_mul(input_count, sizeof(int), "recurrence channel bytes");
}

void DeviceContraction::initialize_device_recurrence(
    const DenseCoherentRecurrenceProblem &problem) {
  copy_view_to_device(problem.x, x_values_, stream_->get());
  copy_view_to_device(problem.i1, i1_values_, stream_->get());
  copy_view_to_device(problem.k2, k2_values_, stream_->get());
  copy_view_to_device(problem.k3, k3_values_, stream_->get());
  copy_view_to_device(problem.k4, k4_values_, stream_->get());
  copy_view_to_device(problem.k5, k5_values_, stream_->get());
  recurrence_failures_.allocate(1);
  check_cuda(cudaMemsetAsync(recurrence_failures_.get(), 0,
                             sizeof(std::uint64_t), stream_->get()),
             "cudaMemsetAsync recurrence failure count");

  device_recurrence_.dimensions = dims_;
  device_recurrence_.x = x_values_.get();
  device_recurrence_.x_count = problem.x.extents[0];
  device_recurrence_.i1 = i1_values_.get();
  device_recurrence_.k2 = k2_values_.get();
  device_recurrence_.k3 = k3_values_.get();
  device_recurrence_.k4 = k4_values_.get();
  device_recurrence_.k5 = k5_values_.get();
  device_recurrence_.Dj12 = problem.Dj12;
  device_recurrence_.Dj13 = problem.Dj13;
  device_recurrence_.Dj14 = problem.Dj14;
  device_recurrence_.Dj15 = problem.Dj15;
  device_recurrence_.Dj23 = problem.Dj23;
  device_recurrence_.Dj24 = problem.Dj24;
  device_recurrence_.Dj25 = problem.Dj25;
  device_recurrence_.Dj34 = problem.Dj34;
  device_recurrence_.Dj35 = problem.Dj35;
  device_recurrence_.Dj45 = problem.Dj45;
}

TensorSpec DeviceContraction::cutensor_k_spec() const {
  return TensorSpec{{static_cast<std::int64_t>(dims_.k5),
                     static_cast<std::int64_t>(dims_.k2),
                     static_cast<std::int64_t>(dims_.l25)},
                    {1, static_cast<std::int64_t>(dims_.k5),
                     static_cast<std::int64_t>(dims_.k5 * dims_.k2)},
                    {'v', 'k', 'p'},
                    cutensor_compat::kComplex64};
}

TensorSpec DeviceContraction::cutensor_h_spec() const {
  return TensorSpec{
      {static_cast<std::int64_t>(dims_.k2),
       static_cast<std::int64_t>(dims_.l23),
       static_cast<std::int64_t>(dims_.k3),
       static_cast<std::int64_t>(dims_.l25)},
      {1, static_cast<std::int64_t>(dims_.k2),
       static_cast<std::int64_t>(dims_.k2 * dims_.l23),
       static_cast<std::int64_t>(dims_.k2 * dims_.l23 * dims_.k3)},
      {'k', 'u', 'i', 'p'},
      cutensor_compat::kComplex64};
}

TensorSpec DeviceContraction::cutensor_x2_spec() const {
  const std::uint64_t VP = dims_.k5 * dims_.l25;
  return TensorSpec{{static_cast<std::int64_t>(dims_.k5),
                     static_cast<std::int64_t>(dims_.l25),
                     static_cast<std::int64_t>(dims_.l23),
                     static_cast<std::int64_t>(dims_.k3)},
                    {1, static_cast<std::int64_t>(dims_.k5),
                     static_cast<std::int64_t>(VP),
                     static_cast<std::int64_t>(VP * dims_.l23)},
                    {'v', 'p', 'u', 'i'},
                    cutensor_compat::kComplex64};
}

TensorSpec DeviceContraction::cutensor_k3_tile_spec(std::uint64_t Rt) const {
  return TensorSpec{
      {static_cast<std::int64_t>(dims_.k3),
       static_cast<std::int64_t>(dims_.l23), static_cast<std::int64_t>(Rt),
       static_cast<std::int64_t>(dims_.l35)},
      {1, static_cast<std::int64_t>(dims_.k3),
       static_cast<std::int64_t>(dims_.k3 * dims_.l23 * dims_.l35),
       static_cast<std::int64_t>(dims_.k3 * dims_.l23)},
      {'i', 'u', 'r', 's'},
      cutensor_compat::kComplex64};
}

TensorSpec DeviceContraction::cutensor_x3_spec(std::uint64_t Rt) const {
  const std::uint64_t VP = dims_.k5 * dims_.l25;
  return TensorSpec{
      {static_cast<std::int64_t>(dims_.k5),
       static_cast<std::int64_t>(dims_.l25),
       static_cast<std::int64_t>(dims_.k3), static_cast<std::int64_t>(Rt),
       static_cast<std::int64_t>(dims_.l35)},
      {1, static_cast<std::int64_t>(dims_.k5), static_cast<std::int64_t>(VP),
       static_cast<std::int64_t>(VP * dims_.k3),
       static_cast<std::int64_t>(VP * dims_.k3 * Rt)},
      {'v', 'p', 'i', 'r', 's'},
      cutensor_compat::kComplex64};
}

void DeviceContraction::add_plan_metrics(
    const coherent_cutensor_detail::PlanMetrics &metrics) {
  stats_.cutensor_workspace_bytes_max =
      std::max(stats_.cutensor_workspace_bytes_max, metrics.workspace_actual);
}

void DeviceContraction::initialize_plans() {
  cutensor_ = std::make_unique<Context>(options_.cutensor_jit);
  x2_plan_ = std::make_unique<BinaryPlan>(BinaryPlan::make(
      *cutensor_, cutensor_k_spec(), cutensor_h_spec(), cutensor_x2_spec(),
      options_.cutensor_workspace_bytes, "coherent-X2"));
  add_plan_metrics(x2_plan_->metrics());

  std::vector<std::uint64_t> tile_sizes{shape_->Rt};
  const std::uint64_t tail = dims_.l34 % shape_->Rt;
  if (tail != 0 && tail != shape_->Rt)
    tile_sizes.push_back(tail);
  for (const std::uint64_t Rt : tile_sizes) {
    TilePlans plans;
    plans.Rt = Rt;
    plans.x3 = std::make_unique<BinaryPlan>(BinaryPlan::make(
        *cutensor_, cutensor_x2_spec(), cutensor_k3_tile_spec(Rt),
        cutensor_x3_spec(Rt), options_.cutensor_workspace_bytes,
        "coherent-X3-rt" + std::to_string(Rt)));
    add_plan_metrics(plans.x3->metrics());
    tile_plans_.push_back(std::move(plans));
  }

  recurrence_stream_ = std::make_unique<Stream>();
  middle_stream_ = std::make_unique<Stream>();
  tail_stream_ = std::make_unique<Stream>();
  hybrid_blas_ = std::make_unique<BlasHandle>();
  check_cublas(cublasSetStream(hybrid_blas_->get(), middle_stream_->get()),
               "cublasSetStream coherent hybrid pipeline");
  check_cublas(cublasSetMathMode(hybrid_blas_->get(), CUBLAS_DEFAULT_MATH),
               "cublasSetMathMode coherent hybrid pipeline");
  for (std::size_t slot = 0; slot < slots_; ++slot) {
    c_ready_[slot] = std::make_unique<CudaEvent>();
    x4_ready_[slot] = std::make_unique<CudaEvent>();
    slot_free_[slot] = std::make_unique<CudaEvent>();
    check_cuda(cudaEventRecord(slot_free_[slot]->get(), tail_stream_->get()),
               "record initial coherent slot-free event");
  }
  x2_ready_ = std::make_unique<CudaEvent>();
  middle_q_done_ = std::make_unique<CudaEvent>();
}

DeviceContraction::TilePlans &DeviceContraction::tile_plans(std::uint64_t Rt) {
  const auto found =
      std::find_if(tile_plans_.begin(), tile_plans_.end(),
                   [Rt](const TilePlans &plans) { return plans.Rt == Rt; });
  if (found == tile_plans_.end() || !found->x3) {
    throw std::logic_error("missing coherent cuTENSOR X3 tile plan");
  }
  return *found;
}

void DeviceContraction::initialize_workspace() {
  const std::uint64_t nA =
      product("persistent sjA", dims_.a, dims_.k5, dims_.l25);
  const std::uint64_t nE =
      product("persistent sjE", dims_.a, dims_.k2, dims_.l25);
  const std::uint64_t nB =
      product("persistent sjB", dims_.k5, dims_.k4, dims_.l35, dims_.l45);
  const std::uint64_t nD =
      product("persistent sjD", dims_.k3, dims_.k2, dims_.l23, dims_.l24);
  const std::uint64_t nK =
      product("persistent K", dims_.k5, dims_.k2, dims_.l25);
  sjA_.allocate(as_size(nA, "persistent sjA"));
  sjE_.allocate(as_size(nE, "persistent sjE"));
  sjB_.allocate(as_size(nB, "persistent sjB"));
  sjD_.allocate(as_size(nD, "persistent sjD"));
  weighted_.allocate(as_size(nA, "weighted sjA"));
  K_re_.allocate(as_size(nK, "K real"));
  K_im_.allocate(as_size(nK, "K imaginary"));
  K_.allocate(as_size(nK, "K complex"));

  int tail_blocks_per_sm = 0;
  check_cuda(q_slab_tail_occupancy(&tail_blocks_per_sm),
             "select coherent exact-support tail kernel");
  if (tail_blocks_per_sm <= 0) {
    throw std::runtime_error(
        "coherent exact-support tail cannot reside on this GPU");
  }

  workspace_ = std::make_unique<QSlabWorkspace>(*shape_);
  constexpr unsigned threads = 256;
  double *const booster5_re = workspace_->B5packed.get();
  double *const booster5_im = booster5_re + as_size(shape_->nB5, "B5 split plane");
  pack_q_slab_booster5_fused_split_kernel<<<launch_blocks(shape_->nB5), threads, 0,
                                      stream_->get()>>>(
      B5_.get(), booster5_re, booster5_im, dims_.k5, dims_.l25, dims_.l45, dims_.l35);
  check_cuda(cudaGetLastError(), "pack coherent B5 exact-tail layout");
  check_cuda(cudaStreamSynchronize(stream_->get()),
             "synchronize coherent B5 packing");
  B5_.reset();

  workspace_->allocate_steady(*shape_, slots_,
                              stats_.cutensor_workspace_bytes_max);
  host_outputs_.resize(as_size(shape_->nOutputs, "host coherent outputs"));
}

template <std::size_t Rank>
void DeviceContraction::copy_complex_booster(
    const DenseTensorView<Complex64, Rank> &source,
    DeviceBuffer<cuDoubleComplex> &destination) {
  static_assert(sizeof(Complex64) == sizeof(cuDoubleComplex));
  const std::size_t count = as_size(view_count(source, "coherent booster copy"),
                                    "coherent booster copy");
  if (destination.size() != count)
    destination.allocate(count);
  if (is_first_axis_contiguous(source)) {
    check_cuda(cudaMemcpyAsync(destination.get(), source.data,
                               count * sizeof(Complex64),
                               cudaMemcpyHostToDevice, stream_->get()),
               "cudaMemcpyAsync coherent booster H2D");
  } else {
    const std::vector<Complex64> packed = pack_view(source);
    check_cuda(cudaMemcpyAsync(destination.get(), packed.data(),
                               count * sizeof(Complex64),
                               cudaMemcpyHostToDevice, stream_->get()),
               "cudaMemcpyAsync packed coherent booster H2D");
    check_cuda(cudaStreamSynchronize(stream_->get()),
               "synchronize packed coherent booster");
  }
}

void DeviceContraction::form_k() {
  const std::uint64_t nA = product("sjA", dims_.a, dims_.k5, dims_.l25);
  const std::uint64_t nK = product("K", dims_.k5, dims_.k2, dims_.l25);
  constexpr unsigned threads = 256;
  const double one = 1.0;
  const double zero = 0.0;
  const auto gemm = [&](const double *weighted, double *output) {
    check_cublas(cublasDgemmStridedBatched(
                     blas_->get(), CUBLAS_OP_N, CUBLAS_OP_N,
                     as_int(dims_.k5, "K m"), as_int(dims_.k2, "K n"),
                     as_int(dims_.a, "K k"), &one, weighted,
                     as_int(dims_.k5, "K lda"),
                     as_blas_stride(product("K A stride", dims_.k5, dims_.a),
                                    "K A stride"),
                     sjE_.get(), as_int(dims_.a, "K ldb"),
                     as_blas_stride(product("K B stride", dims_.a, dims_.k2),
                                    "K B stride"),
                     &zero, output, as_int(dims_.k5, "K ldc"),
                     as_blas_stride(product("K C stride", dims_.k5, dims_.k2),
                                    "K C stride"),
                     as_int(dims_.l25, "K batches")),
                 "cublasDgemmStridedBatched coherent K");
  };

  weight_and_transpose_sjA_kernel<<<launch_blocks(nA), threads, 0,
                                    stream_->get()>>>(
      sjA_.get(), c1_.get(), weighted_.get(), dims_.a, dims_.k5, dims_.l25,
      false);
  check_cuda(cudaGetLastError(), "weight coherent sjA real");
  gemm(weighted_.get(), K_re_.get());
  weight_and_transpose_sjA_kernel<<<launch_blocks(nA), threads, 0,
                                    stream_->get()>>>(
      sjA_.get(), c1_.get(), weighted_.get(), dims_.a, dims_.k5, dims_.l25,
      true);
  check_cuda(cudaGetLastError(), "weight coherent sjA imaginary");
  gemm(weighted_.get(), K_im_.get());
  interleave_complex_kernel<<<launch_blocks(nK), threads, 0, stream_->get()>>>(
      K_re_.get(), K_im_.get(), K_.get(), nK);
  check_cuda(cudaGetLastError(), "interleave coherent K");
}

void DeviceContraction::run_x2(std::uint64_t q) {
  constexpr unsigned threads = 256;
  build_q_slab_h_complex_kernel<<<launch_blocks(shape_->nH), threads, 0,
                                  stream_->get()>>>(
      sjD_.get(), B2_.get(), workspace_->H.get(), q, dims_.k3, dims_.k2,
      dims_.l25, dims_.l23);
  check_cuda(cudaGetLastError(), "build coherent direct-complex H");
  x2_plan_->execute(*cutensor_, K_.get(), workspace_->H.get(),
                    workspace_->X2.get(), workspace_->cutensor_workspace.get(),
                    stream_->get());
}

void DeviceContraction::run_hybrid_x4(const double *c,
                                      const cuDoubleComplex *x3,
                                      cuDoubleComplex *x4, std::uint64_t Rt) {
  if (c == nullptr || x3 == nullptr || x4 == nullptr || Rt == 0) {
    throw std::invalid_argument("invalid coherent hybrid X4 arguments");
  }
  const std::uint64_t VP = dims_.k5 * dims_.l25;
  const std::uint64_t Z = Rt * dims_.l35;
  const double one = 1.0;
  const double zero = 0.0;

  // C is [j,i,z] real and X3 is [vp,i,z] complex.  Viewing X3 as
  // [component+2*vp,i,z] gives one real DGEMM per z and writes the
  // exact-tail-native layout [j,component+2*vp,z] without a repack.
  check_cublas(
      cublasDgemmStridedBatched(
          hybrid_blas_->get(), CUBLAS_OP_N, CUBLAS_OP_T,
          as_int(dims_.k4, "hybrid X4 m"), as_int(2 * VP, "hybrid X4 n"),
          as_int(dims_.k3, "hybrid X4 k"), &one, c,
          as_int(dims_.k4, "hybrid X4 lda"),
          as_blas_stride(product("hybrid X4 A stride", dims_.k4, dims_.k3),
                         "hybrid X4 A stride"),
          reinterpret_cast<const double *>(x3), as_int(2 * VP, "hybrid X4 ldb"),
          as_blas_stride(product("hybrid X4 B stride", 2, VP, dims_.k3),
                         "hybrid X4 B stride"),
          &zero, reinterpret_cast<double *>(x4),
          as_int(dims_.k4, "hybrid X4 ldc"),
          as_blas_stride(product("hybrid X4 C stride", 2, dims_.k4, VP),
                         "hybrid X4 C stride"),
          as_int(Z, "hybrid X4 batches")),
      "cublasDgemmStridedBatched coherent hybrid X4");
}

void DeviceContraction::enqueue_middle(std::uint64_t r_begin, std::uint64_t Rt,
                                       const double *c, cuDoubleComplex *x4) {
  const std::uint64_t k3_offset = product("coherent B3 tile offset", dims_.l23,
                                          dims_.l35, dims_.k3, r_begin);
  const cuDoubleComplex *const k3 =
      B3_.get() + as_size(k3_offset, "coherent B3 tile offset");
  TilePlans &plans = tile_plans(Rt);
  plans.x3->execute(*cutensor_, workspace_->X2.get(), k3, workspace_->X3.get(),
                    workspace_->cutensor_workspace.get(),
                    middle_stream_->get());
  run_hybrid_x4(c, workspace_->X3.get(), x4, Rt);
}

void DeviceContraction::enqueue_tail(std::uint64_t q, std::uint64_t r_begin,
                                     std::uint64_t Rt,
                                     const cuDoubleComplex *x4,
                                     cuDoubleComplex *output,
                                     std::size_t slot) {
  if (slot >= slots_) {
    throw std::logic_error("invalid coherent exact-tail slot");
  }
  const std::uint64_t b_tiles = 1 + (shape_->SV - 1) / q_slab_fused_b_tile;
  const std::uint64_t partial_count = product(
      "coherent exact-tail partial count", (dims_.k4 * Rt + 31) / 32, b_tiles,
      (dims_.l45 + fused_q_slab_t_tile - 1) / fused_q_slab_t_tile);
  if (partial_count > workspace_->partials[slot].size()) {
    throw std::logic_error(
        "coherent exact tail exceeds its planned partial workspace");
  }
  const double *const booster5_re = workspace_->B5packed.get();
  const double *const booster5_im =
      booster5_re + as_size(shape_->nB5, "coherent B5 split plane");
  const double *const x4_re = reinterpret_cast<const double *>(x4);

  QSlabSupportTailArguments args;
  args.x4_re = x4_re;
  args.x4_im = x4_re + dims_.k4;
  args.booster5_re = booster5_re;
  args.booster5_im = booster5_im;
  args.sjB_packed = workspace_->sjBpacked.get();
  args.k4 = B4_.get();
  args.partials = workspace_->partials[slot].get();
  args.i_ranges_by_z = workspace_->i_ranges[slot].get();
  args.j_ranges_by_z = workspace_->j_ranges[slot].get();
  args.p_ranges_by_v = workspace_->p_ranges.get();
  args.v_x_active = workspace_->v_active.get();
  args.Dk4_first = host_k4_first_;
  args.Dk5_first = host_k5_first_;
  args.Dj14 = device_recurrence_.Dj14;
  args.Dj35 = device_recurrence_.Dj35;
  args.Dj45 = device_recurrence_.Dj45;
  args.q = q;
  args.r_begin = r_begin;
  args.Rt = Rt;
  args.V = dims_.k5;
  args.J = dims_.k4;
  args.Rdim = dims_.l34;
  args.P = dims_.l25;
  args.S = dims_.l35;
  args.T = dims_.l45;
  args.x4_z_pitch = product("hybrid X4 z pitch", 2, dims_.k4, shape_->VP);
  args.x4_v_pitch = product("hybrid X4 v pitch", 2, dims_.k4);
  args.x4_p_pitch = product("hybrid X4 p pitch", 2, dims_.k4, dims_.k5);
  check_cuda(launch_q_slab_support_tail(args, tail_stream_->get()),
             "launch coherent exact-support tail");

  constexpr unsigned threads = 256;
  const std::uint64_t reduce_target =
      (partial_count + threads * 8 - 1) / (threads * 8);
  const std::uint64_t reduce_blocks = std::max<std::uint64_t>(
      1, std::min<std::uint64_t>(workspace_->block_sums[slot].size(),
                                 reduce_target));
  reduce_q_slab_fused_block_sums_kernel<<<static_cast<unsigned>(reduce_blocks),
                                          threads, 0, tail_stream_->get()>>>(
      workspace_->partials[slot].get(), partial_count,
      workspace_->block_sums[slot].get());
  check_cuda(cudaGetLastError(), "reduce coherent exact-tail block partials");
  reduce_q_slab_partials_kernel<<<1, threads, 2 * threads * sizeof(double),
                                  tail_stream_->get()>>>(
      workspace_->block_sums[slot].get(), reduce_blocks, output);
  check_cuda(cudaGetLastError(), "reduce coherent exact-tail scalar");
}

Complex64 DeviceContraction::run_pipeline(std::size_t x_index) {
  constexpr unsigned threads = 256;
  const std::uint64_t nB =
      product("coherent sjB pack", dims_.k4, shape_->SV, dims_.l45);
  pack_sjB_for_fused_q_slab_kernel<<<launch_blocks(nB), threads, 0,
                                     stream_->get()>>>(
      sjB_.get(), workspace_->sjBpacked.get(), dims_.k5, dims_.k4, dims_.l35,
      dims_.l45);
  check_cuda(cudaGetLastError(), "pack coherent per-X sjB");
  support_q_slab_build_x_descriptors_kernel<<<launch_blocks(dims_.k5), threads,
                                              0, stream_->get()>>>(
      workspace_->p_ranges.get(), workspace_->v_active.get(),
      host_x_values_.at(x_index), host_k5_first_, device_recurrence_.Dj14,
      device_recurrence_.Dj15, device_recurrence_.Dj25, dims_.k5, dims_.l25);
  check_cuda(cudaGetLastError(), "build coherent per-X support descriptors");
  check_cuda(cudaStreamSynchronize(stream_->get()),
             "synchronize coherent per-X preparation");

  std::uint64_t output_index = 0;
  std::uint64_t tile_ordinal = 0;
  for (std::uint64_t q = 0; q < dims_.l24; ++q) {
    if (q != 0) {
      check_cuda(cudaStreamWaitEvent(stream_->get(), middle_q_done_->get(), 0),
                 "wait previous coherent middle q");
    }
    run_x2(q);
    check_cuda(cudaEventRecord(x2_ready_->get(), stream_->get()),
               "record coherent X2 ready");

    for (std::uint64_t r_begin = 0; r_begin < dims_.l34;
         r_begin += shape_->Rt, ++tile_ordinal) {
      const std::uint64_t Rt = std::min(shape_->Rt, dims_.l34 - r_begin);
      const std::size_t slot = static_cast<std::size_t>(tile_ordinal % slots_);
      ++stats_.q_tiles;
      if (output_index >= shape_->nOutputs) {
        throw std::logic_error("coherent pipeline produced too many outputs");
      }

      check_cuda(cudaStreamWaitEvent(recurrence_stream_->get(),
                                     slot_free_[slot]->get(), 0),
                 "wait coherent pipeline slot free");
      check_cuda(launch_generate_coherent_c_qr_slab(
                     device_recurrence_, static_cast<std::uint64_t>(x_index), q,
                     r_begin, Rt, workspace_->C[slot].get(),
                     recurrence_failures_.get(), c_recurrence_config_,
                     recurrence_stream_->get()),
                 "generate coherent sjC q/r slab");
      const std::uint64_t Z = product("coherent support Z", Rt, dims_.l35);
      support_q_slab_build_z_descriptors_kernel<<<launch_blocks(Z), threads, 0,
                                                  recurrence_stream_->get()>>>(
          workspace_->i_ranges[slot].get(), workspace_->j_ranges[slot].get(),
          host_x_values_.at(x_index), host_k3_first_, host_k4_first_,
          device_recurrence_.Dj24, device_recurrence_.Dj34,
          device_recurrence_.Dj35, q, r_begin, Rt, dims_.k3, dims_.k4,
          dims_.l35);
      check_cuda(cudaGetLastError(), "build coherent q/r support descriptors");
      check_cuda(
          cudaEventRecord(c_ready_[slot]->get(), recurrence_stream_->get()),
          "record coherent C ready");

      check_cuda(
          cudaStreamWaitEvent(middle_stream_->get(), x2_ready_->get(), 0),
          "wait coherent X2 ready");
      check_cuda(
          cudaStreamWaitEvent(middle_stream_->get(), c_ready_[slot]->get(), 0),
          "wait coherent C ready");
      enqueue_middle(r_begin, Rt, workspace_->C[slot].get(),
                     workspace_->X4[slot].get());
      check_cuda(cudaEventRecord(x4_ready_[slot]->get(), middle_stream_->get()),
                 "record coherent X4 ready");

      check_cuda(
          cudaStreamWaitEvent(tail_stream_->get(), x4_ready_[slot]->get(), 0),
          "wait coherent X4 in exact tail");
      enqueue_tail(q, r_begin, Rt, workspace_->X4[slot].get(),
                   workspace_->outputs.get() + output_index, slot);
      check_cuda(cudaEventRecord(slot_free_[slot]->get(), tail_stream_->get()),
                 "record coherent slot free");
      ++output_index;
    }
    check_cuda(cudaEventRecord(middle_q_done_->get(), middle_stream_->get()),
               "record coherent middle q complete");
  }

  check_cuda(cudaStreamSynchronize(tail_stream_->get()),
             "synchronize coherent production pipeline");
  if (output_index != shape_->nOutputs) {
    throw std::logic_error(
        "coherent pipeline produced fewer outputs than planned");
  }
  check_cuda(cudaMemcpy(host_outputs_.data(), workspace_->outputs.get(),
                        as_size(output_index, "coherent output count") *
                            sizeof(cuDoubleComplex),
                        cudaMemcpyDeviceToHost),
             "copy coherent tile outputs D2H");

  long double total_re = 0.0L;
  long double total_im = 0.0L;
  for (std::uint64_t i = 0; i < output_index; ++i) {
    total_re +=
        static_cast<long double>(host_outputs_[static_cast<std::size_t>(i)].x);
    total_im +=
        static_cast<long double>(host_outputs_[static_cast<std::size_t>(i)].y);
  }
  return Complex64{static_cast<double>(total_re),
                   static_cast<double>(total_im)};
}

Complex64 DeviceContraction::contract_generated_slice(std::size_t x_index) {
  if (x_index >= device_recurrence_.x_count) {
    throw std::out_of_range("coherent X index is out of range");
  }
  check_cuda(cudaSetDevice(device_), "cudaSetDevice coherent slice");
  ++stats_.x_slices;
  const auto start = Clock::now();

  check_cuda(launch_generate_coherent_wings(
                 device_recurrence_, static_cast<std::uint64_t>(x_index),
                 sjA_.get(), sjB_.get(), sjD_.get(), sjE_.get(),
                 recurrence_failures_.get(), wing_recurrence_config_,
                 stream_->get()),
             "generate coherent 6j wings");
  form_k();
  const Complex64 value = run_pipeline(x_index);
  stats_.contraction_seconds += seconds_since(start);
  return value;
}

DeviceStats DeviceContraction::stats() {
  check_cuda(cudaSetDevice(device_), "cudaSetDevice coherent stats");
  std::uint64_t failures = 0;
  check_cuda(cudaMemcpy(&failures, recurrence_failures_.get(), sizeof(failures),
                        cudaMemcpyDeviceToHost),
             "copy coherent recurrence failure count D2H");
  if (failures != 0) {
    throw b4utils::NumericalFailure(
        "device-side coherent 6j generation reported " +
        std::to_string(failures) + " failed series");
  }
  return stats_;
}

inline std::vector<int> select_devices(const DenseContractionOptions &options) {
  int count = 0;
  check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount coherent");
  if (count <= 0) {
    throw std::runtime_error("no CUDA device is visible");
  }
  std::vector<int> devices;
  if (options.device_count == 0) {
    devices.reserve(static_cast<std::size_t>(count));
    for (int device = 0; device < count; ++device) {
      devices.push_back(device);
    }
  } else {
    if (options.device_ids == nullptr) {
      throw std::invalid_argument(
          "device_count is nonzero but device_ids is null");
    }
    devices.assign(options.device_ids,
                   options.device_ids + options.device_count);
    for (const int device : devices) {
      if (device < 0 || device >= count) {
        throw std::invalid_argument(
            "coherent device id is outside the visible range");
      }
    }
    std::vector<int> unique = devices;
    std::sort(unique.begin(), unique.end());
    if (std::adjacent_find(unique.begin(), unique.end()) != unique.end()) {
      throw std::invalid_argument("coherent device_ids contains a duplicate");
    }
  }
  return devices;
}

} // namespace cuda_detail

bool coherent_cuda_backend_available() noexcept {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::size_t coherent_cuda_device_count() noexcept {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 0)
    return 0;
  return static_cast<std::size_t>(count);
}

DenseContractionResult contract_device_generated_coherent_multi_gpu(
    const DenseCoherentBoosters &boosters,
    const DenseComplexVectorView &c1_coeffs,
    const DenseCoherentRecurrenceProblem &problem,
    const DenseContractionOptions &options) {
  const auto total_start = cuda_detail::Clock::now();
  DenseContractionResult result{};
  const std::size_t x_count = static_cast<std::size_t>(problem.x.extents[0]);
  if (options.progress_begin != nullptr) {
    options.progress_begin(options.progress_context,
                           static_cast<std::uint64_t>(x_count));
  }
  if (x_count == 0) {
    result.pipeline_slots = options.pipeline_slots;
    result.total_seconds =
        std::chrono::duration<double>(cuda_detail::Clock::now() - total_start)
            .count();
    return result;
  }

  std::vector<int> devices = cuda_detail::select_devices(options);
  if (devices.size() > x_count)
    devices.resize(x_count);
  std::vector<Complex64> per_x(x_count, Complex64{0.0, 0.0});
  std::vector<cuda_detail::DeviceStats> stats(devices.size());
  std::atomic<std::size_t> next_x{0};
  std::atomic<bool> cancelled{false};
  std::atomic<std::uint32_t> active_devices{0};
  std::mutex barrier_mutex;
  std::condition_variable barrier_cv;
  std::size_t initialized = 0;
  std::exception_ptr failure;
  std::vector<std::thread> workers;
  workers.reserve(devices.size());

  for (std::size_t worker_index = 0; worker_index < devices.size();
       ++worker_index) {
    workers.emplace_back([&, worker_index] {
      std::unique_ptr<cuda_detail::DeviceContraction> contraction;
      try {
        contraction = std::make_unique<cuda_detail::DeviceContraction>(
            devices[worker_index], boosters, c1_coeffs, problem, options);
      } catch (...) {
        std::lock_guard<std::mutex> lock(barrier_mutex);
        if (!failure)
          failure = std::current_exception();
        cancelled.store(true, std::memory_order_release);
      }

      {
        std::unique_lock<std::mutex> lock(barrier_mutex);
        ++initialized;
        barrier_cv.notify_all();
        barrier_cv.wait(lock, [&] {
          return initialized == devices.size() ||
                 cancelled.load(std::memory_order_acquire);
        });
      }
      if (!contraction || cancelled.load(std::memory_order_acquire)) {
        return;
      }

      active_devices.fetch_add(1, std::memory_order_relaxed);
      try {
        while (!cancelled.load(std::memory_order_acquire)) {
          const std::size_t x = next_x.fetch_add(1, std::memory_order_relaxed);
          if (x >= x_count)
            break;
          per_x[x] = contraction->contract_generated_slice(x);
          // contract_generated_slice returns only after the exact-tail stream
          // has synchronized and its output has been copied to the host.  A
          // progress unit therefore represents completed work, not merely
          // kernels enqueued on an asynchronous CUDA stream.
          if (options.progress_advance != nullptr) {
            options.progress_advance(options.progress_context, 1);
          }
        }
        stats[worker_index] = contraction->stats();
        if (options.worker_diagnostic != nullptr) {
          cuda_detail::report_worker_diagnostics(
              devices[worker_index], stats[worker_index], options);
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(barrier_mutex);
        if (!failure)
          failure = std::current_exception();
        cancelled.store(true, std::memory_order_release);
      }
    });
  }

  {
    std::unique_lock<std::mutex> lock(barrier_mutex);
    barrier_cv.wait(lock, [&] {
      return initialized == devices.size() ||
             cancelled.load(std::memory_order_acquire);
    });
    barrier_cv.notify_all();
  }
  for (std::thread &worker : workers)
    worker.join();
  if (failure)
    std::rethrow_exception(failure);
  if (next_x.load(std::memory_order_relaxed) < x_count) {
    throw b4utils::InternalFailure(
        "coherent multi-GPU scheduling stopped before all X slices");
  }

  // Device assignment is dynamic, but reduction order is fixed by X.
  long double amplitude_re = 0.0L;
  long double amplitude_im = 0.0L;
  for (const Complex64 value : per_x) {
    amplitude_re += static_cast<long double>(value.re);
    amplitude_im += static_cast<long double>(value.im);
  }
  result.amplitude = {static_cast<double>(amplitude_re),
                      static_cast<double>(amplitude_im)};
  result.x_slices_completed = x_count;
  result.devices_used = active_devices.load(std::memory_order_relaxed);
  result.pipeline_slots = options.pipeline_slots;
  result.worker_records = options.worker_diagnostic != nullptr
      ? static_cast<std::uint32_t>(devices.size()) : 0;
  result.r_tile_min = std::numeric_limits<std::uint64_t>::max();

  for (const cuda_detail::DeviceStats &device : stats) {
    if (device.x_slices == 0)
      continue;
    result.peak_device_bytes =
        std::max(result.peak_device_bytes, device.peak_bytes);
    result.r_tile_min = std::min(result.r_tile_min, device.r_tile);
    result.r_tile_max = std::max(result.r_tile_max, device.r_tile);
    result.upload_seconds =
        std::max(result.upload_seconds, device.upload_seconds);
    result.contraction_seconds =
        std::max(result.contraction_seconds, device.contraction_seconds);

    result.q_tiles_completed += device.q_tiles;
  }

  if (result.r_tile_min == std::numeric_limits<std::uint64_t>::max()) {
    result.r_tile_min = 0;
  }
  result.total_seconds =
      std::chrono::duration<double>(cuda_detail::Clock::now() - total_start)
          .count();
  return result;
}

} // namespace b4gpu
