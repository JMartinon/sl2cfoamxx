// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "gpu/FullTensorSemiFusedGpuApi.hpp"

#include "boosters/B4_function.hpp"
#include "core/Failure.hpp"
#include "full_tensor/FullTensorInputSupport_HalfSjC.hpp"
#include "full_tensor/FullTensorPackedContainers.hpp"
#include "full_tensor/FullTensorQReflectionOrbitPlan.hpp"
#include "gpu/CudaBuffer.cuh"
#include "gpu/FullTensorGpuDeviceSixJ.cuh"
#include "gpu/FullTensorSemiFusedGpuKernels.cuh"
#include "gpu/FullTensorSemiFusedNvtx.hpp"
#include "gpu/FullTensorSemiFusedCuTensor.hpp"
#include "gpu/ProductionGpuFallback.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace b4gpu {
namespace {

using Clock = std::chrono::steady_clock;
using cuda_detail::DeviceBuffer;
using cuda_detail::DeviceCoherentRecurrenceProblem;
using cuda_detail::FullTensorDeviceSixJLaunchPlan;
using cuda_detail::check_cuda;
using semifused_detail::SemiFusedDims;
using semifused_detail::SemiFusedQDescriptor;
using semifused_cutensor_detail::Context;
using semifused_cutensor_detail::BinaryPlan;

[[nodiscard]] FullTensorSemiFusedFailureKind classify_failure(
    const std::exception& failure) noexcept
{
    if (dynamic_cast<const b4utils::ResourceLimitFailure*>(&failure) !=
            nullptr ||
        sl2cfoam::gpu::production::detail::is_memory_fallback_failure(
            failure)) {
        return FullTensorSemiFusedFailureKind::MemoryPressure;
    }
    if (dynamic_cast<const b4utils::NumericalFailure*>(&failure) != nullptr) {
        return FullTensorSemiFusedFailureKind::Numerical;
    }
    if (dynamic_cast<const b4utils::InternalFailure*>(&failure) != nullptr ||
        dynamic_cast<const std::logic_error*>(&failure) != nullptr) {
        return FullTensorSemiFusedFailureKind::Internal;
    }
    if (dynamic_cast<const b4utils::UnsupportedOperationFailure*>(&failure) !=
        nullptr) {
        return FullTensorSemiFusedFailureKind::UnsupportedOperation;
    }
    return FullTensorSemiFusedFailureKind::Fatal;
}

[[nodiscard]] std::uint64_t checked_add(
    std::uint64_t a, std::uint64_t b, const char* what)
{
    if (a > std::numeric_limits<std::uint64_t>::max() - b) {
        throw std::overflow_error(std::string(what) + " addition overflow");
    }
    return a + b;
}

[[nodiscard]] std::uint64_t checked_mul(
    std::uint64_t a, std::uint64_t b, const char* what)
{
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error(
            std::string(what) + " multiplication overflow");
    }
    return a * b;
}

template <class... Rest>
[[nodiscard]] std::uint64_t product(
    const char* what, std::uint64_t first, Rest... rest)
{
    std::uint64_t result = first;
    ((result = checked_mul(result, static_cast<std::uint64_t>(rest), what)), ...);
    return result;
}

[[nodiscard]] std::size_t as_size(std::uint64_t value, const char* what)
{
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(std::string(what) + " exceeds size_t");
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] int as_blas_int(std::uint64_t value, const char* what)
{
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string(what) + " exceeds cuBLAS int");
    }
    return static_cast<int>(value);
}

[[nodiscard]] long long as_blas_stride(std::uint64_t value, const char* what)
{
    if (value > static_cast<std::uint64_t>(
                    std::numeric_limits<long long>::max())) {
        throw std::overflow_error(std::string(what) + " exceeds cuBLAS stride");
    }
    return static_cast<long long>(value);
}

[[nodiscard]] std::uint64_t bytes_for(
    std::uint64_t values, std::uint64_t element_bytes, const char* what)
{
    return checked_mul(values, element_bytes, what);
}

[[nodiscard]] unsigned grid_cap(
    std::uint64_t tasks, int sms, std::uint32_t waves)
{
    if (tasks == 0) return 1;
    const std::uint64_t cap = checked_mul(
        static_cast<std::uint64_t>(std::max(1, sms)),
        static_cast<std::uint64_t>(std::max<std::uint32_t>(1, waves)),
        "semi-fused producer grid cap");
    return static_cast<unsigned>(std::min<std::uint64_t>(
        tasks, std::min<std::uint64_t>(cap, 0x7fffffffull)));
}

void check_cublas(cublasStatus_t status, const char* operation)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuBLAS status " +
            std::to_string(static_cast<int>(status)));
    }
}

class Stream {
public:
    explicit Stream(int priority = 0)
    {
        check_cuda(cudaStreamCreateWithPriority(
                       &stream_, cudaStreamNonBlocking, priority),
                   "cudaStreamCreateWithPriority");
    }

    ~Stream()
    {
        if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class BlasHandle {
public:
    explicit BlasHandle(cudaStream_t stream)
    {
        check_cublas(cublasCreate(&handle_), "cublasCreate");
        try {
            check_cublas(cublasSetStream(handle_, stream), "cublasSetStream");
            check_cublas(
                cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH),
                "cublasSetMathMode");
        } catch (...) {
            (void)cublasDestroy(handle_);
            handle_ = nullptr;
            throw;
        }
    }
    ~BlasHandle()
    {
        if (handle_ != nullptr) (void)cublasDestroy(handle_);
    }
    BlasHandle(const BlasHandle&) = delete;
    BlasHandle& operator=(const BlasHandle&) = delete;
    [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

class Event {
public:
    explicit Event(unsigned flags = cudaEventDisableTiming)
    {
        check_cuda(cudaEventCreateWithFlags(&event_, flags),
                   "cudaEventCreateWithFlags");
    }
    ~Event()
    {
        if (event_ != nullptr) (void)cudaEventDestroy(event_);
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

enum class ProfileStage : std::uint8_t {
    T1,
    T3,
    LeftProducer,
    RightProducer,
    FinalGemm,
    CompactFold
};

class DeferredProfiler {
public:
    explicit DeferredProfiler(bool enabled) : enabled_(enabled) {}

    ~DeferredProfiler()
    {
        for (Pair& pair : pairs_) {
            if (pair.begin != nullptr) (void)cudaEventDestroy(pair.begin);
            if (pair.end != nullptr) (void)cudaEventDestroy(pair.end);
        }
    }

    [[nodiscard]] std::size_t begin(ProfileStage stage, cudaStream_t stream)
    {
        if (!enabled_) return invalid_index();
        Pair pair;
        pair.stage = stage;
        check_cuda(cudaEventCreate(&pair.begin), "cudaEventCreate profile begin");
        try {
            check_cuda(cudaEventCreate(&pair.end),
                       "cudaEventCreate profile end");
            check_cuda(cudaEventRecord(pair.begin, stream),
                       "cudaEventRecord profile begin");
        } catch (...) {
            if (pair.begin != nullptr) (void)cudaEventDestroy(pair.begin);
            if (pair.end != nullptr) (void)cudaEventDestroy(pair.end);
            throw;
        }
        pairs_.push_back(pair);
        return pairs_.size() - 1u;
    }

    void end(std::size_t index, cudaStream_t stream)
    {
        if (!enabled_) return;
        if (index >= pairs_.size()) {
            throw std::logic_error("invalid deferred profile event index");
        }
        check_cuda(cudaEventRecord(pairs_[index].end, stream),
                   "cudaEventRecord profile end");
    }

    void resolve(FullTensorSemiFusedStageProfile& profile)
    {
        if (!enabled_) return;
        for (const Pair& pair : pairs_) {
            check_cuda(cudaEventSynchronize(pair.end),
                       "cudaEventSynchronize profile end");
            float milliseconds = 0.0f;
            check_cuda(cudaEventElapsedTime(
                           &milliseconds, pair.begin, pair.end),
                       "cudaEventElapsedTime profile");
            add(profile, pair.stage,
                static_cast<double>(milliseconds) * 1.0e-3);
        }
    }

private:
    struct Pair {
        ProfileStage stage{};
        cudaEvent_t begin = nullptr;
        cudaEvent_t end = nullptr;
    };

    [[nodiscard]] static constexpr std::size_t invalid_index() noexcept
    {
        return std::numeric_limits<std::size_t>::max();
    }

    static void add(
        FullTensorSemiFusedStageProfile& p,
        ProfileStage stage,
        double seconds)
    {
        switch (stage) {
        case ProfileStage::T1:
            p.t1_seconds += seconds; break;
        case ProfileStage::T3:
            p.t3_seconds += seconds; break;
        case ProfileStage::LeftProducer:
            p.left_producer_seconds += seconds;
            break;
        case ProfileStage::RightProducer:
            p.right_producer_seconds += seconds;
            break;
        case ProfileStage::FinalGemm:
            p.final_gemm_seconds += seconds; break;
        case ProfileStage::CompactFold:
            p.compact_fold_seconds += seconds;
            break;
        }
    }

    bool enabled_ = false;
    std::vector<Pair> pairs_;
};


[[nodiscard]] std::uint64_t visible_mask(int count) noexcept
{
    if (count <= 0) return 0;
    if (count >= 64) return std::numeric_limits<std::uint64_t>::max();
    return (std::uint64_t{1} << static_cast<unsigned>(count)) - 1u;
}

[[nodiscard]] SemiFusedDims make_dims(
    const b4support::FullTensorInputSupportPlan& support)
{
    SemiFusedDims d;
    d.I1 = support.channels.i1.size();
    d.I2 = support.channels.i2.size();
    d.I3 = support.channels.i3.size();
    d.I4 = support.channels.i4.size();
    d.I5 = support.channels.i5.size();
    d.X = support.channels.xvals.size();
    d.K2 = support.channels.k2.size();
    d.K3 = support.channels.k3.size();
    d.K4 = support.channels.k4.size();
    d.K5 = support.channels.k5.size();
    d.U = support.l23.size();
    d.Q = support.l24.size();
    d.P = support.l25.size();
    d.R = support.l34.size();
    d.S = support.l35.size();
    d.T = support.l45.size();

    const std::array<std::uint64_t, 16> axes{
        d.I1, d.I2, d.I3, d.I4, d.I5, d.X,
        d.K2, d.K3, d.K4, d.K5, d.U, d.Q, d.P, d.R, d.S, d.T};
    if (std::find(axes.begin(), axes.end(), 0) != axes.end()) {
        throw std::invalid_argument(
            "semi-fused full-tensor route received an empty tensor axis");
    }

    d.M = checked_mul(d.I2, d.I3, "semi-fused M");
    d.N = checked_mul(d.I4, d.I5, "semi-fused N");
    d.Upad = semifused_detail::round_up_u64(
        d.U, semifused_detail::kMmaK);
    d.Tpad = semifused_detail::round_up_u64(
        d.T, semifused_detail::kMmaK);
    d.tile2 = (d.I2 + semifused_detail::kBoundaryTile - 1u) /
        semifused_detail::kBoundaryTile;
    d.tile3 = (d.I3 + semifused_detail::kBoundaryTile - 1u) /
        semifused_detail::kBoundaryTile;
    d.tile4 = (d.I4 + semifused_detail::kBoundaryTile - 1u) /
        semifused_detail::kBoundaryTile;
    d.tile5 = (d.I5 + semifused_detail::kBoundaryTile - 1u) /
        semifused_detail::kBoundaryTile;
    d.T1Logical = product(
        "semi-fused logical T1", d.T, d.K4, d.I5, d.P, d.S);
    d.T3Logical = product(
        "semi-fused logical T3", d.U, d.K3, d.I2, d.P, d.Q);
    d.T1 = product(
        "semi-fused MMA T1", d.Tpad, semifused_detail::kBoundaryTile,
        d.tile5, d.K4, d.P, d.S);
    d.T3 = product(
        "semi-fused MMA T3", d.Upad, semifused_detail::kBoundaryTile,
        d.tile2, d.K3, d.P, d.Q);
    d.T1Operand = product(
        "semi-fused cuTENSOR T1 operand",
        d.K5, d.T, d.I5, d.K4, d.P, d.S);
    d.T3Operand = product(
        "semi-fused cuTENSOR T3 operand",
        d.K2, d.U, d.I2, d.K3, d.P, d.Q);
    d.B3Mma = product(
        "semi-fused MMA B3", d.Upad, semifused_detail::kBoundaryTile,
        d.tile3, d.K3, d.R, d.S);
    d.B4Mma = product(
        "semi-fused MMA B4", d.Tpad, semifused_detail::kBoundaryTile,
        d.tile4, d.K4, d.Q, d.R);
    d.Cq = product("semi-fused C q", d.K4, d.K3, d.R, d.S);
    d.Cx = checked_mul(d.Q, d.Cq, "semi-fused C x");
    return d;
}

[[nodiscard]] DenseCoherentDimensions dense_dimensions(
    const SemiFusedDims& d) noexcept
{
    return DenseCoherentDimensions{
        d.I1, d.K2, d.K3, d.K4, d.K5,
        d.U, d.Q, d.P, d.R, d.S, d.T};
}

[[nodiscard]] std::uint64_t live_budget(
    std::size_t free_bytes,
    const FullTensorSemiFusedOptions& options) noexcept
{
    const double fraction = std::clamp(options.vram_fraction, 0.05, 1.0);
    std::uint64_t budget = static_cast<std::uint64_t>(
        std::floor(static_cast<long double>(free_bytes) * fraction));
    budget = budget > options.reserve_device_bytes
        ? budget - options.reserve_device_bytes : 0;
    if (options.max_device_bytes != 0) {
        budget = std::min(budget, options.max_device_bytes);
    }
    return budget;
}

struct DeviceInfo {
    int id = -1;
    int major = 0;
    int minor = 0;
    int sms = 0;
    int clock_rate_khz = 0;
    int max_threads_per_sm = 0;
    std::uint64_t budget = 0;
};

[[nodiscard]] std::vector<DeviceInfo> discover_devices(
    const FullTensorSemiFusedOptions& options)
{
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    const std::uint64_t selected = options.device_mask == 0
        ? visible_mask(count) : options.device_mask;
    if ((selected & ~visible_mask(count)) != 0) {
        throw std::invalid_argument(
            "semi-fused device mask contains a non-visible CUDA ordinal");
    }

    std::vector<DeviceInfo> devices;
    for (int id = 0; id < count && id < 64; ++id) {
        if ((selected & (std::uint64_t{1} << static_cast<unsigned>(id))) == 0) {
            continue;
        }
        cudaDeviceProp property{};
        check_cuda(cudaGetDeviceProperties(&property, id),
                   "cudaGetDeviceProperties");
        check_cuda(cudaSetDevice(id), "cudaSetDevice during admission");
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
        (void)total_bytes;
        devices.push_back(DeviceInfo{
            id, property.major, property.minor,
            property.multiProcessorCount, property.clockRate,
            property.maxThreadsPerMultiProcessor,
            live_budget(free_bytes, options)});
    }
    if (devices.empty()) {
        throw std::runtime_error(
            "no selected CUDA device is available");
    }
    return devices;
}

template <class T>
void upload(
    DeviceBuffer<T>& destination,
    const T* source,
    std::uint64_t count,
    cudaStream_t stream)
{
    destination.allocate(as_size(count, "CUDA upload count"));
    if (count == 0) return;
    if (source == nullptr) {
        throw std::invalid_argument("null source for nonempty CUDA upload");
    }
    const std::uint64_t bytes = bytes_for(count, sizeof(T), "CUDA upload");
    check_cuda(cudaMemcpyAsync(
                   destination.get(), source, as_size(bytes, "CUDA upload bytes"),
                   cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync H2D");
}

template <class T>
void upload(
    DeviceBuffer<T>& destination,
    const std::vector<T>& source,
    cudaStream_t stream)
{
    upload(destination, source.data(), source.size(), stream);
}

[[nodiscard]] std::uint8_t choose_association(
    std::uint64_t nk3,
    std::uint64_t nk4,
    const SemiFusedDims& d) noexcept
{
    const long double left =
        2.0L * d.P * nk3 * nk4 * d.M +
        2.0L * d.P * nk4 * d.M * d.N;
    const long double right =
        2.0L * d.P * nk3 * nk4 * d.N +
        2.0L * d.P * nk3 * d.M * d.N;
    return right < left ? 1u : 0u;
}

[[nodiscard]] std::uint64_t descriptor_rows(
    const SemiFusedQDescriptor& q,
    const SemiFusedDims& d)
{
    return checked_mul(
        d.P, q.association == 0u ? q.k4_count : q.k3_count,
        "semi-fused descriptor rows");
}

struct HostDescriptor {
    SemiFusedQDescriptor device{};
};

struct BaseSchedule {
    FullTensorGpuNativeContractionTree tree =
        FullTensorGpuNativeContractionTree::LogicalQ;
    std::vector<std::vector<HostDescriptor>> by_x;
    std::uint64_t maximum_descriptor_rows = 0;
    std::uint64_t total_rows_per_i1 = 0;
};

[[nodiscard]] BaseSchedule build_base_schedule(
    const b4support::FullTensorInputSupportPlan& support,
    const FullTensorGpuNativeOutputLayout& output_layout,
    FullTensorGpuNativeContractionTree requested,
    const SemiFusedDims& d)
{
    if (output_layout.M != d.M || output_layout.N != d.N) {
        throw std::invalid_argument(
            "semi-fused output layout does not match boundary intertwiner dimensions");
    }

    const auto orbit_plan =
        b4support::build_full_tensor_q_reflection_orbit_plan(support);
    const bool canonical_available =
        orbit_plan.canonical_orbits_enabled &&
        output_layout.has_valid_reflection_orbit_metadata();
    if (requested ==
            FullTensorGpuNativeContractionTree::CanonicalReflectionOrbits &&
        !canonical_available) {
        std::string reason = orbit_plan.fallback_reason;
        if (reason.empty()) {
            reason = "compact reflection output metadata is unavailable";
        }
        throw std::invalid_argument(
            "canonical reflection-orbit semi-fused tree is unavailable: " +
            reason);
    }
    const bool use_canonical = canonical_available &&
        requested != FullTensorGpuNativeContractionTree::LogicalQ;

    BaseSchedule result;
    result.tree = use_canonical
        ? FullTensorGpuNativeContractionTree::CanonicalReflectionOrbits
        : FullTensorGpuNativeContractionTree::LogicalQ;
    result.by_x.resize(as_size(d.X, "semi-fused X schedule"));

    const auto append_member = [&] (
        const b4support::FullTensorQOrbitMember& member,
        b4support::FullTensorQOrbitKind kind,
        std::uint8_t association) {
        if (member.logical.empty()) return;
        SemiFusedQDescriptor q;
        q.il24 = member.il24;
        q.il34 = member.il34;
        q.il35 = member.il35;
        q.k3_begin = member.logical.i_begin;
        q.k3_count = member.logical.i_end - member.logical.i_begin;
        q.k4_begin = member.logical.j_begin;
        q.k4_count = member.logical.j_end - member.logical.j_begin;
        q.association = association;
        q.orbit_kind = static_cast<std::uint8_t>(kind);
        HostDescriptor entry;
        entry.device = q;
        result.by_x.at(member.ix).push_back(entry);
        const std::uint64_t rows = descriptor_rows(q, d);
        result.maximum_descriptor_rows = std::max(
            result.maximum_descriptor_rows, rows);
        result.total_rows_per_i1 = checked_add(
            result.total_rows_per_i1, rows, "semi-fused total rows");
    };

    if (use_canonical) {
        for (const auto& orbit : orbit_plan.orbits) {
            const std::uint8_t association =
                orbit.tail_association ==
                    b4support::FullTensorQTailAssociation::CTimesRight
                ? 1u : 0u;
            append_member(orbit.representative, orbit.kind, association);
        }
    } else {
        for (std::uint64_t ix = 0; ix < d.X; ++ix) {
            for (std::uint64_t il35 = 0; il35 < d.S; ++il35) {
                for (std::uint64_t il24 = 0; il24 < d.Q; ++il24) {
                    for (std::uint64_t il34 = 0; il34 < d.R; ++il34) {
                        const std::size_t block = b4support::idx4(
                            ix, il24, il34, il35, d.Q, d.R, d.S);
                        const auto& map = support.sjC_map.at(block);
                        if (map.logical.empty()) continue;
                        b4support::FullTensorQOrbitMember member;
                        member.logical_block = block;
                        member.ix = static_cast<std::uint32_t>(ix);
                        member.il24 = static_cast<std::uint32_t>(il24);
                        member.il34 = static_cast<std::uint32_t>(il34);
                        member.il35 = static_cast<std::uint32_t>(il35);
                        member.logical = map.logical;
                        append_member(
                            member,
                            b4support::FullTensorQOrbitKind::Unpaired,
                            choose_association(
                                member.nk3(), member.nk4(), d));
                    }
                }
            }
        }
    }
    return result;
}

struct WorkGroup {
    std::uint64_t x = 0;
    std::uint64_t descriptor_begin = 0;
    std::uint64_t descriptor_count = 0;
    std::uint64_t rows = 0;
    std::uint32_t il24 = 0;
    std::uint32_t l34_begin = 0;
    std::uint32_t l34_count = 0;
    bool resident_c = true;
};

struct PackedSchedule {
    std::vector<SemiFusedQDescriptor> descriptors;
    std::vector<WorkGroup> groups;
    std::vector<std::uint64_t> x_group_offsets;
};

void append_group(
    PackedSchedule& result,
    std::uint64_t x,
    const std::vector<const HostDescriptor*>& selected,
    bool resident_c,
    std::uint32_t il24,
    std::uint32_t l34_begin,
    std::uint32_t l34_count,
    const SemiFusedDims& d)
{
    if (selected.empty()) return;
    WorkGroup group;
    group.x = x;
    group.descriptor_begin = result.descriptors.size();
    group.descriptor_count = selected.size();
    group.il24 = il24;
    group.l34_begin = l34_begin;
    group.l34_count = l34_count;
    group.resident_c = resident_c;

    std::vector<const HostDescriptor*> ordered = selected;
    std::stable_sort(ordered.begin(), ordered.end(),
        [&](const HostDescriptor* a, const HostDescriptor* b) {
            if (a->device.association != b->device.association) {
                return a->device.association < b->device.association;
            }
            const std::uint64_t ar = descriptor_rows(a->device, d);
            const std::uint64_t br = descriptor_rows(b->device, d);
            return ar > br;
        });

    std::uint64_t prefix = 0;
    for (const HostDescriptor* source : ordered) {
        SemiFusedQDescriptor q = source->device;
        q.row_prefix = prefix;
        const std::uint64_t rows = descriptor_rows(q, d);
        prefix = checked_add(prefix, rows, "semi-fused group row prefix");
        result.descriptors.push_back(q);
    }
    group.rows = prefix;
    result.groups.push_back(group);
}

[[nodiscard]] PackedSchedule pack_schedule(
    const BaseSchedule& base,
    const SemiFusedDims& d,
    FullTensorSemiFusedMemoryPolicy policy,
    std::uint32_t l34_tile)
{
    PackedSchedule result;
    result.x_group_offsets.resize(as_size(d.X + 1u, "x group offsets"), 0);

    for (std::uint64_t x = 0; x < d.X; ++x) {
        result.x_group_offsets[x] = result.groups.size();
        if (policy == FullTensorSemiFusedMemoryPolicy::ResidentC) {
            std::vector<const HostDescriptor*> selected;
            selected.reserve(base.by_x[x].size());
            for (const HostDescriptor& descriptor : base.by_x[x]) {
                selected.push_back(&descriptor);
            }
            append_group(
                result, x, selected, true, 0, 0,
                static_cast<std::uint32_t>(d.R), d);
        } else {
            for (std::uint32_t q = 0; q < d.Q; ++q) {
                for (std::uint32_t r0 = 0; r0 < d.R; r0 += l34_tile) {
                    const std::uint32_t rc = std::min<std::uint32_t>(
                        l34_tile, static_cast<std::uint32_t>(d.R) - r0);
                    std::vector<const HostDescriptor*> selected;
                    for (const HostDescriptor& descriptor : base.by_x[x]) {
                        const auto& v = descriptor.device;
                        if (v.il24 == q && v.il34 >= r0 && v.il34 < r0 + rc) {
                            selected.push_back(&descriptor);
                        }
                    }
                    append_group(result, x, selected, false, q, r0, rc, d);
                }
            }
        }
    }
    result.x_group_offsets[d.X] = result.groups.size();
    return result;
}

[[nodiscard]] std::vector<std::uint32_t> i1_candidates(
    std::uint64_t maximum_assignment,
    std::uint32_t requested)
{
    if (requested != 0) {
        if (requested > maximum_assignment) {
            throw std::invalid_argument(
                "explicit semi-fused i1 tile exceeds device assignment");
        }
        return {requested};
    }
    std::set<std::uint32_t, std::greater<>> values;
    if (maximum_assignment >
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error(
            "semi-fused i1 assignment exceeds uint32_t");
    }
    const std::uint32_t full =
        static_cast<std::uint32_t>(maximum_assignment);
    values.insert(full);
    for (const std::uint32_t v :
         {256u, 192u, 160u, 128u, 96u, 80u, 64u, 48u, 32u,
          24u, 16u, 12u, 8u, 6u, 4u, 3u, 2u, 1u}) {
        if (v <= maximum_assignment) values.insert(v);
    }
    return {values.begin(), values.end()};
}

[[nodiscard]] std::vector<std::uint32_t> x_candidates(
    std::uint64_t extent,
    std::uint32_t requested)
{
    if (requested != 0) {
        if (requested > extent) {
            throw std::invalid_argument(
                "explicit semi-fused x tile exceeds extent");
        }
        return {requested};
    }
    std::set<std::uint32_t, std::greater<>> values;
    if (extent > static_cast<std::uint64_t>(
                     std::numeric_limits<std::uint32_t>::max())) {
        throw std::overflow_error("semi-fused X extent exceeds uint32_t");
    }
    const std::uint32_t full = static_cast<std::uint32_t>(extent);
    values.insert(full);
    for (std::uint32_t value = full; value > 1u;) {
        value = (value + 1u) / 2u;
        values.insert(value);
    }
    for (const std::uint32_t v :
         {256u, 192u, 128u, 96u, 64u, 48u, 32u, 24u, 16u, 12u,
          8u, 6u, 4u, 3u, 2u, 1u}) {
        if (v <= full) values.insert(v);
    }
    return {values.begin(), values.end()};
}

[[nodiscard]] std::vector<std::uint32_t> r_candidates(
    std::uint64_t extent,
    std::uint32_t requested)
{
    if (requested != 0) {
        if (requested > extent) {
            throw std::invalid_argument(
                "explicit semi-fused l34 tile exceeds extent");
        }
        return {requested};
    }
    std::set<std::uint32_t, std::greater<>> values;
    const std::uint32_t full = static_cast<std::uint32_t>(extent);
    values.insert(full);
    for (std::uint32_t value = full; value > 1u;) {
        value = (value + 1u) / 2u;
        values.insert(value);
    }
    for (const std::uint32_t v : {16u, 12u, 8u, 6u, 4u, 3u, 2u, 1u}) {
        if (v <= full) values.insert(v);
    }
    return {values.begin(), values.end()};
}

struct MemoryEstimate {
    std::uint64_t fixed_bytes = 0;
    std::uint64_t expansion_peak_bytes = 0;
    std::uint64_t peak_bytes = 0;
};

[[nodiscard]] MemoryEstimate estimate_fixed_memory(
    const SemiFusedDims& d,
    const b4support::FullTensorInputSupportPlan& support,
    const FullTensorGpuNativeOutputLayout& output_layout,
    FullTensorSemiFusedMemoryPolicy policy,
    std::uint32_t i1_tile,
    std::uint32_t x_tile,
    std::uint32_t l34_tile,
    std::uint64_t descriptor_count,
    const FullTensorSemiFusedOptions& options)
{
    const std::uint64_t dense_B2 = product(
        "semi-fused dense B2", d.I2, d.U, d.Q, d.P, d.K2);
    const std::uint64_t dense_B3_source = product(
        "semi-fused source B3", d.I3, d.K3, d.R, d.S, d.U);
    const std::uint64_t dense_B4_source = product(
        "semi-fused source B4", d.I4, d.K4, d.Q, d.R, d.T);
    const std::uint64_t dense_B5 = product(
        "semi-fused dense B5", d.I5, d.P, d.S, d.T, d.K5);
    const std::uint64_t resident_boosters = checked_add(
        checked_add(dense_B2, dense_B5, "resident boosters"),
        checked_add(d.B3Mma, d.B4Mma, "resident MMA boosters"),
        "resident boosters");

    std::uint64_t values = resident_boosters;
    auto add_values = [&](std::uint64_t amount) {
        values = checked_add(values, amount, "semi-fused fixed values");
    };
    if (policy == FullTensorSemiFusedMemoryPolicy::ResidentC) {
        add_values(checked_mul(x_tile, d.Cx, "resident C"));
    } else {
        add_values(product("streamed C", d.K4, d.K3, l34_tile, d.S));
    }
    const std::uint64_t boundary_a_values = product(
        "boundary intertwiner A", i1_tile, d.K5, d.P);
    const std::uint64_t boundary_e_values = product(
        "boundary intertwiner E", i1_tile, d.K2, d.P);
    add_values(boundary_a_values);
    add_values(boundary_e_values);
    add_values(boundary_a_values); // beta-fast cuTENSOR A pack
    add_values(boundary_e_values); // beta-fast cuTENSOR E pack
    add_values(d.T1Operand);    // recurrence-fused sjB * B5 operand
    add_values(d.T3Operand);    // recurrence-fused sjD * B2 operand
    add_values(checked_mul(i1_tile, d.T1Logical,
                           "packed logical T1 output"));
    add_values(checked_mul(i1_tile, d.T3Logical,
                           "packed logical T3 output"));
    add_values(checked_mul(i1_tile, d.T1, "T1 cache"));
    add_values(checked_mul(i1_tile, d.T3, "T3 cache"));
    // Compact reflection output needs one dense GEMM scratch panel plus its
    // persistent folded payload. Rectangular output is itself the persistent
    // dense GEMM destination and needs no duplicate panel.
    if (output_layout.kind == FullTensorGpuNativeOutputKind::CompactByI1) {
        add_values(product("dense output", i1_tile, d.M, d.N));
    }
    add_values(checked_mul(
        i1_tile, output_layout.values_per_i1(), "persistent output"));

    std::uint64_t fixed = bytes_for(values, sizeof(double), "fixed doubles");
    fixed = checked_add(fixed, bytes_for(
        descriptor_count, sizeof(SemiFusedQDescriptor), "descriptors"),
        "fixed bytes");
    const std::uint64_t integer_values =
        d.X + d.I1 + d.K2 + d.K3 + d.K4 + d.K5;
    fixed = checked_add(fixed, bytes_for(
        integer_values, sizeof(int), "recurrence axes"), "fixed bytes");
    fixed = checked_add(fixed, sizeof(std::uint64_t), "fixed bytes");
    fixed = checked_add(
        fixed,
        checked_mul(2u, options.cutensor_workspace_bytes,
                    "two concurrent cuTENSOR workspaces"),
        "fixed bytes");
    if (output_layout.kind == FullTensorGpuNativeOutputKind::CompactByI1) {
        fixed = checked_add(fixed, bytes_for(product(
            "compact coordinates", output_layout.values_per_i1(), 4u),
            sizeof(std::uint32_t), "compact coordinates"), "fixed bytes");
    }

    const std::uint64_t packed_values =
        support.B2.packed_values + support.B3.packed_values +
        support.B4.packed_values + support.B5.packed_values;
    const std::uint64_t range_count =
        support.B2.blocks.size() + support.B3.blocks.size() +
        support.B4.blocks.size() + support.B5.blocks.size();
    const std::uint64_t expansion_values = checked_add(
        resident_boosters,
        checked_add(dense_B3_source, dense_B4_source,
                    "temporary canonical B3/B4"),
        "booster expansion values");
    std::uint64_t expansion = bytes_for(
        expansion_values, sizeof(double), "dense booster expansion");
    expansion = checked_add(expansion, bytes_for(
        packed_values, sizeof(double), "packed boosters"), "expansion peak");
    expansion = checked_add(expansion, bytes_for(
        range_count, sizeof(b4support::Range1D), "booster ranges"),
        "expansion peak");
    return MemoryEstimate{fixed, expansion, std::max(fixed, expansion)};
}

[[nodiscard]] std::uint64_t maximum_rows_per_x_slab(
    const PackedSchedule& schedule,
    const SemiFusedDims& d,
    std::uint32_t x_tile)
{
    std::uint64_t maximum = 0;
    for (std::uint64_t x0 = 0; x0 < d.X; x0 += x_tile) {
        const std::uint64_t x1 = std::min<std::uint64_t>(d.X, x0 + x_tile);
        std::uint64_t rows = 0;
        for (std::uint64_t x = x0; x < x1; ++x) {
            const std::uint64_t group_begin = schedule.x_group_offsets[x];
            const std::uint64_t group_end = schedule.x_group_offsets[x + 1u];
            for (std::uint64_t gi = group_begin; gi < group_end; ++gi) {
                rows = checked_add(
                    rows, schedule.groups[gi].rows,
                    "semi-fused x-slab aggregate rows");
            }
        }
        maximum = std::max(maximum, rows);
    }
    return maximum;
}

[[nodiscard]] std::uint64_t predicted_flushes_per_i1_tile(
    const PackedSchedule& schedule,
    const SemiFusedDims& d,
    std::uint64_t capacity,
    std::uint32_t x_tile)
{
    std::uint64_t calls = 0;
    for (std::uint64_t x0 = 0; x0 < d.X; x0 += x_tile) {
        const std::uint64_t x1 = std::min<std::uint64_t>(d.X, x0 + x_tile);
        std::uint64_t used = 0;
        for (std::uint64_t x = x0; x < x1; ++x) {
            const std::uint64_t group_begin = schedule.x_group_offsets[x];
            const std::uint64_t group_end = schedule.x_group_offsets[x + 1u];
            for (std::uint64_t gi = group_begin; gi < group_end; ++gi) {
                const WorkGroup& group = schedule.groups[gi];
                const std::uint64_t end =
                    group.descriptor_begin + group.descriptor_count;
                for (std::uint64_t i = group.descriptor_begin; i < end; ++i) {
                    const std::uint64_t rows =
                        descriptor_rows(schedule.descriptors[i], d);
                    if (rows > capacity) {
                        return std::numeric_limits<std::uint64_t>::max();
                    }
                    if (used != 0 && rows > capacity - used) {
                        ++calls;
                        used = 0;
                    }
                    used += rows;
                    if (used == capacity) {
                        ++calls;
                        used = 0;
                    }
                }
            }
        }
        // Device output is downloaded and accumulated once per x slab, so a
        // partially filled K buffer cannot straddle this boundary.
        if (used != 0) ++calls;
    }
    return calls;
}

[[nodiscard]] std::uint64_t predicted_launch_pairs_per_i1_tile(
    const PackedSchedule& schedule,
    const SemiFusedDims& d,
    std::uint64_t capacity,
    std::uint32_t x_tile)
{
    // Mirror process_group(): one producer pair cannot cross an sjC group,
    // an association boundary, the remaining K-slab capacity, or an x-slab
    // boundary.  The latter matters because each x slab is downloaded and
    // accumulated independently when common 6j storage must be bounded.
    std::uint64_t pairs = 0;
    for (std::uint64_t x0 = 0; x0 < d.X; x0 += x_tile) {
        const std::uint64_t x1 = std::min<std::uint64_t>(d.X, x0 + x_tile);
        std::uint64_t used = 0;
        for (std::uint64_t x = x0; x < x1; ++x) {
            const std::uint64_t group_begin = schedule.x_group_offsets[x];
            const std::uint64_t group_end = schedule.x_group_offsets[x + 1u];
            for (std::uint64_t gi = group_begin; gi < group_end; ++gi) {
                const WorkGroup& group = schedule.groups[gi];
                const std::uint64_t end =
                    group.descriptor_begin + group.descriptor_count;
                std::uint64_t begin = group.descriptor_begin;
                while (begin < end) {
                    const std::uint64_t remaining = capacity - used;
                    const std::uint8_t association =
                        schedule.descriptors[begin].association;
                    std::uint64_t segment_end = begin;
                    std::uint64_t segment_rows = 0;
                    while (segment_end < end) {
                        const SemiFusedQDescriptor& q =
                            schedule.descriptors[segment_end];
                        if (q.association != association) break;
                        const std::uint64_t rows = descriptor_rows(q, d);
                        if (rows > remaining - segment_rows) break;
                        segment_rows += rows;
                        ++segment_end;
                    }
                    if (segment_end == begin) {
                        if (used != 0) {
                            used = 0;
                            continue;
                        }
                        return std::numeric_limits<std::uint64_t>::max();
                    }
                    ++pairs;
                    used += segment_rows;
                    begin = segment_end;
                    if (used == capacity) used = 0;
                }
            }
        }
        used = 0;
    }
    return pairs;
}

struct SelectedPlan {
    FullTensorSemiFusedPlan public_plan{};
    PackedSchedule schedule{};
};

struct ProgressCompletion {
    FullTensorSemiFusedProgressAdvance advance = nullptr;
    void* context = nullptr;
    std::uint64_t completed_rows = 0;
};

void CUDART_CB publish_completed_rows(void* opaque) noexcept
{
    std::unique_ptr<ProgressCompletion> completion(
        static_cast<ProgressCompletion*>(opaque));
    if (completion != nullptr && completion->advance != nullptr) {
        completion->advance(
            completion->context, completion->completed_rows);
    }
}

[[nodiscard]] SelectedPlan choose_plan(
    const SemiFusedDims& d,
    const b4support::FullTensorInputSupportPlan& support,
    const FullTensorGpuNativeOutputLayout& output_layout,
    const FullTensorSemiFusedOptions& options,
    const BaseSchedule& base,
    const std::vector<DeviceInfo>& devices)
{
    const std::uint64_t minimum_budget = std::min_element(
        devices.begin(), devices.end(),
        [](const DeviceInfo& a, const DeviceInfo& b) {
            return a.budget < b.budget;
        })->budget;
    const std::uint64_t maximum_assignment =
        (d.I1 + devices.size() - 1u) / devices.size();

    struct Candidate {
        bool valid = false;
        FullTensorSemiFusedMemoryPolicy policy{};
        std::uint32_t i1 = 0;
        std::uint32_t x = 0;
        std::uint32_t r = 0;
        std::uint64_t capacity = 0;
        std::uint64_t peak = 0;
        std::uint64_t gemm_calls = 0;
        std::uint64_t launch_pairs = 0;
        std::uint64_t c_generation_calls = 0;
        std::uint64_t output_passes = 0;
        double average_k = 0.0;
        PackedSchedule schedule{};
    } best;

    std::vector<FullTensorSemiFusedMemoryPolicy> policies;
    if (options.memory_policy == FullTensorSemiFusedMemoryPolicy::Automatic) {
        policies = {FullTensorSemiFusedMemoryPolicy::ResidentC,
                    FullTensorSemiFusedMemoryPolicy::StreamedC};
    } else {
        policies = {options.memory_policy};
    }

    for (const auto policy : policies) {
        const auto rs = policy == FullTensorSemiFusedMemoryPolicy::ResidentC
            ? std::vector<std::uint32_t>{static_cast<std::uint32_t>(d.R)}
            : r_candidates(d.R, options.l34_tile);
        for (const std::uint32_t r : rs) {
            PackedSchedule packed = pack_schedule(base, d, policy, r);
            for (const std::uint32_t x : x_candidates(d.X, options.x_tile)) {
                for (const std::uint32_t i1 :
                     i1_candidates(maximum_assignment, options.i1_tile)) {
                    const MemoryEstimate memory = estimate_fixed_memory(
                        d, support, output_layout, policy, i1, x, r,
                        packed.descriptors.size(), options);
                    if (memory.peak_bytes > minimum_budget) continue;

                    const std::uint64_t row_values = product(
                        "two semi-fused slabs per row", 2u, i1,
                        checked_add(d.M, d.N, "M+N"));
                    const std::uint64_t row_bytes = bytes_for(
                        row_values, sizeof(double), "semi-fused row bytes");
                    if (row_bytes == 0) continue;
                    const std::uint64_t available =
                        minimum_budget - memory.fixed_bytes;
                    std::uint64_t capacity = available / row_bytes;
                    capacity = std::min(
                        capacity, maximum_rows_per_x_slab(packed, d, x));
                    capacity = std::min<std::uint64_t>(
                        capacity, std::numeric_limits<int>::max());
                    if (options.row_capacity != 0) {
                        capacity = std::min(capacity, options.row_capacity);
                    }
                    if (capacity < base.maximum_descriptor_rows) continue;

                    const std::uint64_t aggregate_bytes = checked_mul(
                        capacity, row_bytes, "semi-fused aggregate buffers");
                    const std::uint64_t runtime_peak = checked_add(
                        memory.fixed_bytes, aggregate_bytes,
                        "semi-fused runtime peak");
                    const std::uint64_t peak = std::max(
                        memory.expansion_peak_bytes, runtime_peak);
                    const std::uint64_t flushes =
                        predicted_flushes_per_i1_tile(packed, d, capacity, x);
                    if (flushes == std::numeric_limits<std::uint64_t>::max()) {
                        continue;
                    }
                    const std::uint64_t tiles =
                        (maximum_assignment + i1 - 1u) / i1;
                    const std::uint64_t x_slabs = (d.X + x - 1u) / x;
                    const std::uint64_t launch_pairs_per_tile =
                        predicted_launch_pairs_per_i1_tile(
                            packed, d, capacity, x);
                    if (launch_pairs_per_tile ==
                        std::numeric_limits<std::uint64_t>::max()) {
                        continue;
                    }
                    const std::uint64_t launch_pairs = checked_mul(
                        launch_pairs_per_tile, tiles,
                        "predicted semi-fused producer launch pairs");
                    const std::uint64_t gemm_calls = checked_mul(
                        flushes, tiles, "predicted semi-fused GEMM calls");
                    const std::uint64_t c_generation_calls = policy ==
                            FullTensorSemiFusedMemoryPolicy::ResidentC
                        ? checked_mul(d.X, d.Q,
                                      "resident C generation calls")
                        : product("streamed C generation calls", d.X, d.Q,
                                  (d.R + r - 1u) / r, tiles);
                    const std::uint64_t output_passes = checked_mul(
                        x_slabs, tiles, "semi-fused output passes");
                    const std::uint64_t final_calls_per_tile = flushes;
                    const double average_k = final_calls_per_tile == 0 ? 0.0 :
                        static_cast<double>(base.total_rows_per_i1) /
                        static_cast<double>(final_calls_per_tile);

                    const bool better = !best.valid ||
                        gemm_calls < best.gemm_calls ||
                        (gemm_calls == best.gemm_calls &&
                         c_generation_calls < best.c_generation_calls) ||
                        (gemm_calls == best.gemm_calls &&
                         c_generation_calls == best.c_generation_calls &&
                         output_passes < best.output_passes) ||
                        (gemm_calls == best.gemm_calls &&
                         c_generation_calls == best.c_generation_calls &&
                         output_passes == best.output_passes &&
                         launch_pairs < best.launch_pairs) ||
                        (gemm_calls == best.gemm_calls &&
                         c_generation_calls == best.c_generation_calls &&
                         output_passes == best.output_passes &&
                         launch_pairs == best.launch_pairs && i1 > best.i1) ||
                        (gemm_calls == best.gemm_calls &&
                         c_generation_calls == best.c_generation_calls &&
                         output_passes == best.output_passes &&
                         launch_pairs == best.launch_pairs && i1 == best.i1 &&
                         x > best.x) ||
                        (gemm_calls == best.gemm_calls &&
                         c_generation_calls == best.c_generation_calls &&
                         output_passes == best.output_passes &&
                         launch_pairs == best.launch_pairs && i1 == best.i1 &&
                         x == best.x && average_k > best.average_k) ||
                        (gemm_calls == best.gemm_calls &&
                         c_generation_calls == best.c_generation_calls &&
                         output_passes == best.output_passes &&
                         launch_pairs == best.launch_pairs && i1 == best.i1 &&
                         x == best.x && average_k == best.average_k &&
                         policy == FullTensorSemiFusedMemoryPolicy::ResidentC &&
                         best.policy != FullTensorSemiFusedMemoryPolicy::ResidentC) ||
                        (gemm_calls == best.gemm_calls &&
                         c_generation_calls == best.c_generation_calls &&
                         output_passes == best.output_passes &&
                         launch_pairs == best.launch_pairs && i1 == best.i1 &&
                         x == best.x && average_k == best.average_k &&
                         policy == best.policy && r > best.r);
                    if (better) {
                        best.valid = true;
                        best.policy = policy;
                        best.i1 = i1;
                        best.x = x;
                        best.r = r;
                        best.capacity = capacity;
                        best.peak = peak;
                        best.gemm_calls = gemm_calls;
                        best.launch_pairs = launch_pairs;
                        best.c_generation_calls = c_generation_calls;
                        best.output_passes = output_passes;
                        best.average_k = average_k;
                        best.schedule = packed;
                    }
                }
            }
        }
    }

    if (!best.valid) {
        throw b4utils::ResourceLimitFailure(
            "no selected CUDA device can hold the minimum semi-fused "
            "workspace (reduce i1/x/row/l34 tiles or use a larger VRAM cap)");
    }

    SelectedPlan selected;
    selected.public_plan.memory_policy = best.policy;
    selected.public_plan.i1_tile = best.i1;
    selected.public_plan.x_tile = best.x;
    selected.public_plan.l34_tile = best.r;
    selected.public_plan.row_capacity = best.capacity;
    selected.public_plan.tree = base.tree;
    const std::uint32_t common_waves =
        options.producer_waves_per_sm == 0 ? 8u : options.producer_waves_per_sm;
    selected.public_plan.producer_waves_per_sm = common_waves;
    selected.public_plan.estimated_peak_device_bytes = best.peak;
    selected.schedule = std::move(best.schedule);
    return selected;
}

class Worker {
public:
    Worker(
        const DeviceInfo& device,
        const B4Spins& spins,
        const b4support::FullTensorInputSupportPlan& support,
        const b4support::PackedFullTensorBoosters& boosters,
        const FullTensorGpuNativeOutputLayout& output_layout,
        const FullTensorSemiFusedOptions& options,
        const SemiFusedDims& d,
        const SelectedPlan& selected,
        std::uint64_t i1_begin,
        std::uint64_t i1_end)
        : device_(device), spins_(spins), support_(support), boosters_(boosters),
          output_layout_(output_layout), options_(options), d_(d),
          selected_(selected), i1_begin_(i1_begin), i1_end_(i1_end),
          profiler_(options.enable_profiling)
    {
        stats_.peak_device_bytes = selected.public_plan.estimated_peak_device_bytes;
    }

    ~Worker()
    {
        // cudaStreamDestroy may return before queued host functions run.  Keep
        // the borrowed progress context alive through this worker's lifetime,
        // including exceptional exits that bypass run_i1_tile's normal sync.
        if ((progress_callbacks_pending_ || unreported_progress_rows_ != 0) &&
            gemm_stream_ != nullptr) {
            if (cudaStreamSynchronize(gemm_stream_->get()) == cudaSuccess) {
                publish_unreported_rows_now();
            }
        }
    }

    void run()
    {
        check_cuda(cudaSetDevice(device_.id), "cudaSetDevice semi-fused worker");
        semifused_nvtx::ScopedRange worker_range(
            options_.enable_nvtx, "b4/semifused/worker", UINT32_C(0xff4f81bd));

        prep_stream_ = std::make_unique<Stream>();
        left_stream_ = std::make_unique<Stream>();
        right_stream_ = std::make_unique<Stream>();
        gemm_stream_ = std::make_unique<Stream>();
        if (options_.enable_nvtx) {
            const std::string prefix = "k4.sf.gpu" + std::to_string(device_.id);
            semifused_nvtx::name_stream(
                prep_stream_->get(), (prefix + ".prep").c_str());
            semifused_nvtx::name_stream(
                left_stream_->get(), (prefix + ".left").c_str());
            semifused_nvtx::name_stream(
                right_stream_->get(), (prefix + ".right").c_str());
            semifused_nvtx::name_stream(
                gemm_stream_->get(), (prefix + ".gemm").c_str());
        }
        blas_ = std::make_unique<BlasHandle>(gemm_stream_->get());
        for (unsigned b = 0; b < 2; ++b) {
            left_ready_[b] = std::make_unique<Event>();
            right_ready_[b] = std::make_unique<Event>();
            free_[b] = std::make_unique<Event>();
        }
        boundary_ready_ = std::make_unique<Event>();
        t1_built_ = std::make_unique<Event>();
        t3_built_ = std::make_unique<Event>();
        c_ready_ = std::make_unique<Event>();
        left_c_done_ = std::make_unique<Event>();
        right_c_done_ = std::make_unique<Event>();

        const auto init_begin = Clock::now();
        upload_and_expand_boosters();
        initialize_recurrence();
        upload_schedule_and_layout();
        initialize_cutensor_plans();
        allocate_workspace();
        check_cuda(cudaStreamSynchronize(prep_stream_->get()),
                   "semi-fused initialization synchronize");
        stats_.initialization_seconds =
            std::chrono::duration<double>(Clock::now() - init_begin).count();

        host_result_.assign(as_size(checked_mul(
            i1_end_ - i1_begin_, output_layout_.values_per_i1(),
            "semi-fused host result"), "semi-fused host result"), 0.0);

        const auto contraction_begin = Clock::now();
        const std::uint64_t x_tile = selected_.public_plan.x_tile;
        for (std::uint64_t x0 = 0; x0 < d_.X; x0 += x_tile) {
            const std::uint64_t xc = std::min<std::uint64_t>(x_tile, d_.X - x0);
            check_cuda(cudaMemsetAsync(
                           failures_.get(), 0, sizeof(std::uint64_t),
                           prep_stream_->get()),
                       "clear semi-fused x-slab recurrence failures");
            if (selected_.public_plan.memory_policy ==
                FullTensorSemiFusedMemoryPolicy::ResidentC) {
                generate_resident_c(x0, xc);
            }
            check_cuda(cudaStreamSynchronize(prep_stream_->get()),
                       "semi-fused x-slab recurrence synchronize");
            check_failures("semi-fused common x-slab generation");

            const bool first_x_slab = x0 == 0;
            for (std::uint64_t a0 = i1_begin_; a0 < i1_end_;
                 a0 += selected_.public_plan.i1_tile) {
                const std::uint64_t ac = std::min<std::uint64_t>(
                    selected_.public_plan.i1_tile, i1_end_ - a0);
                run_i1_tile(a0, ac, x0, xc, first_x_slab);
            }
        }
        stats_.contraction_seconds =
            std::chrono::duration<double>(Clock::now() - contraction_begin).count();
        profiler_.resolve(stats_.profile);
    }

    [[nodiscard]] std::vector<double> take_host_result() && noexcept
    {
        return std::move(host_result_);
    }

    [[nodiscard]] const FullTensorSemiFusedDeviceStats& stats() const noexcept
    {
        return stats_;
    }

private:
    void launch_expand(
        const DeviceBuffer<b4support::Range1D>& ranges,
        const DeviceBuffer<double>& packed,
        DeviceBuffer<double>& dense,
        std::uint64_t outer,
        std::uint64_t inner,
        const char* name)
    {
        semifused_detail::expand_packed_ranges_kernel<<<
            semifused_detail::launch_blocks(outer), 256, 0,
            prep_stream_->get()>>>(
                ranges.get(), packed.get(), dense.get(), outer, inner);
        const cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("expand semi-fused ") + name + ": " +
                cudaGetErrorString(status));
        }
    }

    void upload_and_expand_boosters()
    {
        DeviceBuffer<double> packed_B2, packed_B3, packed_B4, packed_B5;
        DeviceBuffer<b4support::Range1D> ranges_B2, ranges_B3, ranges_B4,
            ranges_B5;
        DeviceBuffer<double> dense_B3_source, dense_B4_source;

        upload(packed_B2, boosters_.B2_data(), support_.B2.packed_values,
               prep_stream_->get());
        upload(packed_B3, boosters_.B3_data(), support_.B3.packed_values,
               prep_stream_->get());
        upload(packed_B4, boosters_.B4_data(), support_.B4.packed_values,
               prep_stream_->get());
        upload(packed_B5, boosters_.B5_data(), support_.B5.packed_values,
               prep_stream_->get());
        upload(ranges_B2, support_.B2.blocks, prep_stream_->get());
        upload(ranges_B3, support_.B3.blocks, prep_stream_->get());
        upload(ranges_B4, support_.B4.blocks, prep_stream_->get());
        upload(ranges_B5, support_.B5.blocks, prep_stream_->get());

        dense_B2_.allocate(as_size(product(
            "semi-fused dense B2", d_.I2, d_.U, d_.Q, d_.P, d_.K2),
            "semi-fused dense B2"));
        dense_B5_.allocate(as_size(product(
            "semi-fused dense B5", d_.I5, d_.P, d_.S, d_.T, d_.K5),
            "semi-fused dense B5"));
        dense_B3_source.allocate(as_size(product(
            "semi-fused source B3", d_.I3, d_.K3, d_.R, d_.S, d_.U),
            "semi-fused source B3"));
        dense_B4_source.allocate(as_size(product(
            "semi-fused source B4", d_.I4, d_.K4, d_.Q, d_.R, d_.T),
            "semi-fused source B4"));
        mma_B3_.allocate(as_size(d_.B3Mma, "semi-fused MMA B3"));
        mma_B4_.allocate(as_size(d_.B4Mma, "semi-fused MMA B4"));

        launch_expand(ranges_B2, packed_B2, dense_B2_,
                      support_.B2.blocks.size(), d_.K2, "B2");
        launch_expand(ranges_B3, packed_B3, dense_B3_source,
                      support_.B3.blocks.size(), d_.U, "B3");
        launch_expand(ranges_B4, packed_B4, dense_B4_source,
                      support_.B4.blocks.size(), d_.T, "B4");
        launch_expand(ranges_B5, packed_B5, dense_B5_,
                      support_.B5.blocks.size(), d_.K5, "B5");
        semifused_detail::pack_booster3_mma_kernel<<<
            semifused_detail::launch_blocks(d_.B3Mma), 256, 0,
            prep_stream_->get()>>>(
                dense_B3_source.get(), mma_B3_.get(), d_);
        check_cuda(cudaGetLastError(), "launch semi-fused B3 MMA pack");
        semifused_detail::pack_booster4_mma_kernel<<<
            semifused_detail::launch_blocks(d_.B4Mma), 256, 0,
            prep_stream_->get()>>>(
                dense_B4_source.get(), mma_B4_.get(), d_);
        check_cuda(cudaGetLastError(), "launch semi-fused B4 MMA pack");
        check_cuda(cudaStreamSynchronize(prep_stream_->get()),
                   "semi-fused booster expansion/pack synchronize");
    }

    void initialize_recurrence()
    {
        const auto require_regular = [](const std::vector<int>& axis,
                                        const char* name) {
            if (axis.empty()) {
                throw std::invalid_argument(
                    std::string("semi-fused recurrence ") + name +
                    " axis is empty");
            }
            for (std::size_t i = 1; i < axis.size(); ++i) {
                if (axis[i] != axis[i - 1] + 2) {
                    throw std::invalid_argument(
                        std::string("semi-fused recurrence ") + name +
                        " axis must be a regular step-two range");
                }
            }
        };
        require_regular(support_.channels.k2, "k2");
        require_regular(support_.channels.k3, "k3");
        require_regular(support_.channels.k4, "k4");
        require_regular(support_.channels.k5, "k5");

        upload(x_values_, support_.channels.xvals, prep_stream_->get());
        upload(i1_values_, support_.channels.i1, prep_stream_->get());
        upload(k2_values_, support_.channels.k2, prep_stream_->get());
        upload(k3_values_, support_.channels.k3, prep_stream_->get());
        upload(k4_values_, support_.channels.k4, prep_stream_->get());
        upload(k5_values_, support_.channels.k5, prep_stream_->get());

        recurrence_.dimensions = dense_dimensions(d_);
        recurrence_.x = x_values_.get();
        recurrence_.x_count = d_.X;
        recurrence_.i1 = i1_values_.get();
        recurrence_.k2 = k2_values_.get();
        recurrence_.k3 = k3_values_.get();
        recurrence_.k4 = k4_values_.get();
        recurrence_.k5 = k5_values_.get();
        recurrence_.Dj12 = spins_.Dj12;
        recurrence_.Dj13 = spins_.Dj13;
        recurrence_.Dj14 = spins_.Dj14;
        recurrence_.Dj15 = spins_.Dj15;
        recurrence_.Dj23 = spins_.Dj23;
        recurrence_.Dj24 = spins_.Dj24;
        recurrence_.Dj25 = spins_.Dj25;
        recurrence_.Dj34 = spins_.Dj34;
        recurrence_.Dj35 = spins_.Dj35;
        recurrence_.Dj45 = spins_.Dj45;

        const int maximum_x = *std::max_element(
            support_.channels.xvals.begin(), support_.channels.xvals.end());
        check_cuda(cuda_detail::select_full_tensor_device_6j_launch_plan(
                       recurrence_, maximum_x, &recurrence_plan_),
                   "select semi-fused device-6j launch plan");
    }

    void upload_schedule_and_layout()
    {
        upload(descriptors_, selected_.schedule.descriptors,
               prep_stream_->get());
        if (output_layout_.kind ==
                FullTensorGpuNativeOutputKind::CompactByI1) {
            upload(rep_m_, output_layout_.representative_m,
                   prep_stream_->get());
            upload(rep_n_, output_layout_.representative_n,
                   prep_stream_->get());
            upload(partner_m_, output_layout_.representative_partner_m,
                   prep_stream_->get());
            upload(partner_n_, output_layout_.representative_partner_n,
                   prep_stream_->get());
        }
    }

    struct CuTensorPlanPair {
        std::uint64_t i1_count = 0;
        BinaryPlan t1;
        BinaryPlan t3;
    };

    void initialize_cutensor_plans()
    {
        // cuTENSOR documents its calls as non-reentrant. T1 and T3 execute
        // concurrently, so give each stream an independent handle, plan cache,
        // preference, and workspace rather than relying on handle reentrancy.
        cutensor_t1_ = std::make_unique<Context>(options_.cutensor_jit);
        cutensor_t3_ = std::make_unique<Context>(options_.cutensor_jit);

        std::set<std::uint64_t> counts;
        for (std::uint64_t a0 = i1_begin_; a0 < i1_end_;
             a0 += selected_.public_plan.i1_tile) {
            counts.insert(std::min<std::uint64_t>(
                selected_.public_plan.i1_tile, i1_end_ - a0));
        }
        for (const std::uint64_t count : counts) {
            CuTensorPlanPair pair;
            pair.i1_count = count;
            pair.t1 = BinaryPlan::make_t1(
                *cutensor_t1_, d_, count, options_.cutensor_workspace_bytes);
            pair.t3 = BinaryPlan::make_t3(
                *cutensor_t3_, d_, count, options_.cutensor_workspace_bytes);

            cutensor_plans_.push_back(std::move(pair));
        }

    }

    [[nodiscard]] const CuTensorPlanPair& cutensor_plan(
        std::uint64_t i1_count) const
    {
        for (const CuTensorPlanPair& pair : cutensor_plans_) {
            if (pair.i1_count == i1_count) return pair;
        }
        throw std::logic_error("missing cuTENSOR T1/T3 plan for i1 tile");
    }

    void allocate_workspace()
    {
        const std::uint64_t at = selected_.public_plan.i1_tile;
        const std::uint64_t xt = selected_.public_plan.x_tile;
        const std::uint64_t capacity = selected_.public_plan.row_capacity;
        boundary_A_.allocate(as_size(product(
            "semi-fused boundary intertwiner A", at, d_.K5, d_.P),
            "semi-fused boundary intertwiner A"));
        boundary_E_.allocate(as_size(product(
            "semi-fused boundary intertwiner E", at, d_.K2, d_.P),
            "semi-fused boundary intertwiner E"));
        boundary_A_virtual_fast_.allocate(boundary_A_.size());
        boundary_E_virtual_fast_.allocate(boundary_E_.size());
        cutensor_t1_operand_.allocate(as_size(
            d_.T1Operand, "cuTENSOR T1 pointwise operand"));
        cutensor_t3_operand_.allocate(as_size(
            d_.T3Operand, "cuTENSOR T3 pointwise operand"));
        cutensor_t1_logical_.allocate(as_size(checked_mul(
            at, d_.T1Logical, "cuTENSOR packed logical T1"),
            "cuTENSOR packed logical T1"));
        cutensor_t3_logical_.allocate(as_size(checked_mul(
            at, d_.T3Logical, "cuTENSOR packed logical T3"),
            "cuTENSOR packed logical T3"));
        std::uint64_t t1_workspace_bytes = 0;
        std::uint64_t t3_workspace_bytes = 0;
        for (const CuTensorPlanPair& pair : cutensor_plans_) {
            t1_workspace_bytes = std::max(
                t1_workspace_bytes, pair.t1.metrics().workspace_actual);
            t3_workspace_bytes = std::max(
                t3_workspace_bytes, pair.t3.metrics().workspace_actual);
        }
        cutensor_t1_workspace_.allocate(as_size(
            t1_workspace_bytes,
            "cuTENSOR T1 workspace bytes"));
        cutensor_t3_workspace_.allocate(as_size(
            t3_workspace_bytes,
            "cuTENSOR T3 workspace bytes"));
        T1_.allocate(as_size(checked_mul(at, d_.T1, "semi-fused T1"),
                             "semi-fused T1"));
        T3_.allocate(as_size(checked_mul(at, d_.T3, "semi-fused T3"),
                             "semi-fused T3"));
        for (unsigned b = 0; b < 2; ++b) {
            L_[b].allocate(as_size(product(
                "semi-fused L buffer", at, capacity, d_.M),
                "semi-fused L buffer"));
            R_[b].allocate(as_size(product(
                "semi-fused R buffer", at, capacity, d_.N),
                "semi-fused R buffer"));
        }
        if (output_layout_.kind ==
                FullTensorGpuNativeOutputKind::CompactByI1) {
            dense_output_.allocate(as_size(product(
                "semi-fused dense output", at, d_.M, d_.N),
                "semi-fused dense output"));
        }
        compact_output_.allocate(as_size(checked_mul(
            at, output_layout_.values_per_i1(),
            "semi-fused persistent output"),
            "semi-fused persistent output"));
        failures_.allocate(1);
        check_cuda(cudaMemsetAsync(
                       failures_.get(), 0, sizeof(std::uint64_t),
                       prep_stream_->get()),
                   "clear semi-fused recurrence failures");

        if (selected_.public_plan.memory_policy ==
            FullTensorSemiFusedMemoryPolicy::ResidentC) {
            C_.allocate(as_size(checked_mul(xt, d_.Cx, "resident C"),
                                "resident C"));
        } else {
            C_.allocate(as_size(product(
                "streamed C", d_.K4, d_.K3,
                selected_.public_plan.l34_tile, d_.S), "streamed C"));
        }

        check_cuda(cudaFuncSetCacheConfig(
                       semifused_detail::produce_left_u_dmma_kernel,
                       cudaFuncCachePreferL1),
                   "cudaFuncSetCacheConfig semi-fused plain U");
        check_cuda(cudaFuncSetCacheConfig(
                       semifused_detail::produce_right_v_dmma_kernel,
                       cudaFuncCachePreferL1),
                   "cudaFuncSetCacheConfig semi-fused plain V");
        check_cuda(cudaFuncSetCacheConfig(
                       semifused_detail::produce_left_cu_dmma_kernel,
                       cudaFuncCachePreferShared),
                   "cudaFuncSetCacheConfig semi-fused C*U");
        check_cuda(cudaFuncSetCacheConfig(
                       semifused_detail::produce_right_cv_dmma_kernel,
                       cudaFuncCachePreferShared),
                   "cudaFuncSetCacheConfig semi-fused C^T*V");
    }

    void generate_resident_c(std::uint64_t x0, std::uint64_t x_count)
    {
        for (std::uint64_t slot = 0; slot < x_count; ++slot) {
            const std::uint64_t x = x0 + slot;
            for (std::uint64_t q = 0; q < d_.Q; ++q) {
                check_cuda(cuda_detail::launch_generate_full_tensor_6j_c_slab(
                               recurrence_, x, q, 0, d_.R,
                               C_.get() + slot * d_.Cx + q * d_.Cq,
                               failures_.get(), recurrence_plan_,
                               prep_stream_->get()),
                           "generate resident semi-fused C");
            }
        }
    }

    void generate_boundary_and_t(
        std::uint64_t a0,
        std::uint64_t ac,
        std::uint64_t x)
    {
        semifused_nvtx::ScopedRange range(
            options_.enable_nvtx, "b4/prepare/boundary intertwiner-and-T",
            UINT32_C(0xff9467bd));
        if (boundary_buffers_in_use_) {
            check_cuda(cudaStreamWaitEvent(
                           prep_stream_->get(), t1_built_->get(), 0),
                       "boundary intertwiner producer waits for T1 build");
            check_cuda(cudaStreamWaitEvent(
                           prep_stream_->get(), t3_built_->get(), 0),
                       "boundary intertwiner producer waits for T3 build");
            boundary_buffers_in_use_ = false;
        }

        // Boundary recurrence and the i1-independent pointwise operands are
        // independent. Queue all three immediately on separate streams.
        check_cuda(cuda_detail::launch_generate_full_tensor_6j_boundary_wings(
                       recurrence_, x, a0, ac,
                       boundary_A_.get(), boundary_E_.get(), failures_.get(),
                       recurrence_plan_, prep_stream_->get()),
                   "generate semi-fused boundary intertwiner 6j wings");

        {
                semifused_nvtx::ScopedRange operand_range(
                    options_.enable_nvtx, "b4/prepare/W3-operand",
                    UINT32_C(0xff8c6bb1));
                check_cuda(
                    cuda_detail::launch_generate_full_tensor_6j_t3_operand(
                        recurrence_, x, dense_B2_.get(), d_.I2,
                        cutensor_t3_operand_.get(), failures_.get(),
                        recurrence_plan_, left_stream_->get()),
                    "generate recurrence-fused cuTENSOR T3 operand");
            }

            {
                semifused_nvtx::ScopedRange operand_range(
                    options_.enable_nvtx, "b4/prepare/W1-operand",
                    UINT32_C(0xff8c6bb1));
                check_cuda(
                    cuda_detail::launch_generate_full_tensor_6j_t1_operand(
                        recurrence_, x, dense_B5_.get(), d_.I5,
                        cutensor_t1_operand_.get(), failures_.get(),
                        recurrence_plan_, right_stream_->get()),
                    "generate recurrence-fused cuTENSOR T1 operand");
        }

        {
            const std::uint64_t a_values = product(
                "virtual intertwiner-fast boundary intertwiner A pack", ac, d_.K5, d_.P);
            semifused_detail::pack_boundary_virtual_fast_kernel<<<
                semifused_detail::launch_blocks(a_values), 256, 0,
                prep_stream_->get()>>>(
                    boundary_A_.get(), boundary_A_virtual_fast_.get(), ac, d_.K5, d_.P);
            check_cuda(cudaGetLastError(),
                       "launch semi-fused virtual intertwiner-fast boundary intertwiner A pack");
            const std::uint64_t e_values = product(
                "virtual intertwiner-fast boundary intertwiner E pack", ac, d_.K2, d_.P);
            semifused_detail::pack_boundary_virtual_fast_kernel<<<
                semifused_detail::launch_blocks(e_values), 256, 0,
                prep_stream_->get()>>>(
                    boundary_E_.get(), boundary_E_virtual_fast_.get(), ac, d_.K2, d_.P);
            check_cuda(cudaGetLastError(),
                       "launch semi-fused virtual intertwiner-fast boundary intertwiner E pack");
        }

        check_cuda(cudaEventRecord(boundary_ready_->get(), prep_stream_->get()),
                   "record semi-fused boundary intertwiner ready");

        // The operand build is already ahead of this wait on each stream, so
        // it overlaps the boundary intertwiner recurrence/pack instead of extending the
        // critical path unless it is itself slower.
        check_cuda(cudaStreamWaitEvent(
                       left_stream_->get(), boundary_ready_->get(), 0),
                   "T3 waits for boundary intertwiner recurrence/pack");
        const std::uint64_t t3_total = checked_mul(ac, d_.T3, "T3 launch");
        {
            semifused_nvtx::ScopedRange t3_range(
                options_.enable_nvtx, "b4/prepare/T3",
                UINT32_C(0xff8064a2));
            const std::size_t event = profiler_.begin(
                ProfileStage::T3, left_stream_->get());
            const CuTensorPlanPair& plans = cutensor_plan(ac);
            (void)plans.t3.execute(
                *cutensor_t3_, boundary_E_virtual_fast_.get(),
                cutensor_t3_operand_.get(), cutensor_t3_logical_.get(),
                cutensor_t3_workspace_.get(), left_stream_->get());
            profiler_.end(event, left_stream_->get());
            semifused_detail::pack_t3_cutensor_output_kernel<<<
                semifused_detail::launch_blocks(t3_total), 256, 0,
                left_stream_->get()>>>(
                    cutensor_t3_logical_.get(), T3_.get(), d_, ac);
            check_cuda(cudaGetLastError(),
                       "launch semi-fused cuTENSOR T3 output pack");
            check_cuda(cudaEventRecord(t3_built_->get(), left_stream_->get()),
                       "record semi-fused T3 built");
        }

        check_cuda(cudaStreamWaitEvent(
                       right_stream_->get(), boundary_ready_->get(), 0),
                   "T1 waits for boundary intertwiner recurrence/pack");
        const std::uint64_t t1_total = checked_mul(ac, d_.T1, "T1 launch");
        {
            semifused_nvtx::ScopedRange t1_range(
                options_.enable_nvtx, "b4/prepare/T1",
                UINT32_C(0xff8064a2));
            const std::size_t event = profiler_.begin(
                ProfileStage::T1, right_stream_->get());
            const CuTensorPlanPair& plans = cutensor_plan(ac);
            (void)plans.t1.execute(
                *cutensor_t1_, boundary_A_virtual_fast_.get(),
                cutensor_t1_operand_.get(), cutensor_t1_logical_.get(),
                cutensor_t1_workspace_.get(), right_stream_->get());
            profiler_.end(event, right_stream_->get());
            semifused_detail::pack_t1_cutensor_output_kernel<<<
                semifused_detail::launch_blocks(t1_total), 256, 0,
                right_stream_->get()>>>(
                    cutensor_t1_logical_.get(), T1_.get(), d_, ac);
            check_cuda(cudaGetLastError(),
                       "launch semi-fused cuTENSOR T1 output pack");
            check_cuda(cudaEventRecord(t1_built_->get(), right_stream_->get()),
                       "record semi-fused T1 built");
        }

        boundary_buffers_in_use_ = true;
    }

    void ensure_buffer_available(unsigned buffer)
    {
        if (!in_flight_[buffer]) return;
        check_cuda(cudaStreamWaitEvent(
                       left_stream_->get(), free_[buffer]->get(), 0),
                   "left producer waits for semi-fused buffer");
        check_cuda(cudaStreamWaitEvent(
                       right_stream_->get(), free_[buffer]->get(), 0),
                   "right producer waits for semi-fused buffer");
        in_flight_[buffer] = false;
    }

    void ensure_streamed_c_available()
    {
        if (!c_in_use_) return;
        check_cuda(cudaStreamWaitEvent(
                       prep_stream_->get(), left_c_done_->get(), 0),
                   "streamed C waits for left consumer");
        check_cuda(cudaStreamWaitEvent(
                       prep_stream_->get(), right_c_done_->get(), 0),
                   "streamed C waits for right consumer");
        c_in_use_ = false;
    }

    void mark_streamed_c_consumed()
    {
        check_cuda(cudaEventRecord(left_c_done_->get(), left_stream_->get()),
                   "record streamed C left consumer done");
        check_cuda(cudaEventRecord(right_c_done_->get(), right_stream_->get()),
                   "record streamed C right consumer done");
        c_in_use_ = true;
    }

    void launch_left_cu(
        const WorkGroup& group,
        const SemiFusedQDescriptor* q_device,
        std::uint64_t descriptor_count,
        std::uint64_t left_tasks,
        std::uint64_t ac,
        std::uint64_t prefix_base,
        std::uint64_t row_base,
        const double* c_pointer)
    {
        semifused_detail::produce_left_cu_dmma_kernel<<<
            grid_cap(left_tasks, device_.sms,
                     selected_.public_plan.producer_waves_per_sm),
            semifused_detail::kThreadsPerBlock,
            semifused_detail::kTransformProducerSharedBytes,
            left_stream_->get()>>>(
                mma_B3_.get(), T3_.get(), c_pointer, q_device,
                descriptor_count, L_[active_buffer_].get(), d_, ac,
                selected_.public_plan.row_capacity, row_base, prefix_base,
                group.resident_c, group.l34_begin, group.l34_count);
        check_cuda(cudaGetLastError(),
                   "launch custom semi-fused C*U producer");
    }

    void launch_segment(
        const WorkGroup& group,
        std::uint64_t descriptor_begin,
        std::uint64_t descriptor_count,
        std::uint64_t segment_rows,
        std::uint64_t ac,
        const double* c_pointer)
    {
        if (descriptor_count == 0 || segment_rows == 0) return;
        ensure_buffer_available(active_buffer_);
        const std::uint64_t prefix_base =
            selected_.schedule.descriptors[descriptor_begin].row_prefix;
        const SemiFusedQDescriptor* const q_device =
            descriptors_.get() + descriptor_begin;
        const std::uint8_t association =
            selected_.schedule.descriptors[descriptor_begin].association;

        const std::uint64_t left_tasks = product(
            "semi-fused left tasks", ac, descriptor_count, d_.P,
            d_.tile2, d_.tile3);
        const std::uint64_t right_tasks = product(
            "semi-fused right tasks", ac, descriptor_count, d_.P,
            d_.tile4, d_.tile5);

        const auto launch_left = [&] {
            const std::size_t event = profiler_.begin(
                ProfileStage::LeftProducer, left_stream_->get());
            semifused_nvtx::ScopedRange range(
                options_.enable_nvtx,
                association == 0u ? "b4/producer/left/CU"
                                  : "b4/producer/left/U",
                UINT32_C(0xffd9534f));
            if (association == 0u) {
                launch_left_cu(
                    group, q_device, descriptor_count, left_tasks, ac,
                    prefix_base, used_rows_, c_pointer);
            } else {
                semifused_detail::produce_left_u_dmma_kernel<<<
                    grid_cap(
                        left_tasks, device_.sms,
                        selected_.public_plan.producer_waves_per_sm),
                    semifused_detail::kThreadsPerBlock,
                    semifused_detail::kPlainProducerSharedBytes,
                    left_stream_->get()>>>(
                        mma_B3_.get(), T3_.get(), q_device, descriptor_count,
                        L_[active_buffer_].get(), d_, ac,
                        selected_.public_plan.row_capacity, used_rows_,
                        prefix_base);
                check_cuda(cudaGetLastError(),
                           "launch specialized semi-fused plain-U producer");
            }
            profiler_.end(event, left_stream_->get());
        };

        const auto launch_right = [&] {
            semifused_nvtx::ScopedRange range(
                options_.enable_nvtx,
                association == 0u ? "b4/producer/right/V"
                                  : "b4/producer/right/CtV",
                UINT32_C(0xff5bc0de));
            const std::size_t event = profiler_.begin(
                ProfileStage::RightProducer, right_stream_->get());
            if (association == 0u) {
                semifused_detail::produce_right_v_dmma_kernel<<<
                    grid_cap(
                        right_tasks, device_.sms,
                        selected_.public_plan.producer_waves_per_sm),
                    semifused_detail::kThreadsPerBlock,
                    semifused_detail::kPlainProducerSharedBytes,
                    right_stream_->get()>>>(
                        mma_B4_.get(), T1_.get(), q_device, descriptor_count,
                        R_[active_buffer_].get(), d_, ac,
                        selected_.public_plan.row_capacity, used_rows_,
                        prefix_base);
            } else {
                semifused_detail::produce_right_cv_dmma_kernel<<<
                    grid_cap(
                        right_tasks, device_.sms,
                        selected_.public_plan.producer_waves_per_sm),
                    semifused_detail::kThreadsPerBlock,
                    semifused_detail::kTransformProducerSharedBytes,
                    right_stream_->get()>>>(
                        mma_B4_.get(), T1_.get(), c_pointer, q_device,
                        descriptor_count, R_[active_buffer_].get(), d_, ac,
                        selected_.public_plan.row_capacity, used_rows_,
                        prefix_base, group.resident_c, group.l34_begin,
                        group.l34_count);
            }
            check_cuda(cudaGetLastError(),
                       "launch specialized semi-fused right producer");
            profiler_.end(event, right_stream_->get());
        };

        // The measured production schedule submits the right producer first
        // and lets the independent streams overlap at equal priority.
        launch_right();
        launch_left();

        used_rows_ = checked_add(used_rows_, segment_rows,
                                 "active aggregate rows");
    }

    void execute_cublas_materialized_gemm(
        std::uint64_t ac,
        double beta,
        double* output)
    {
        const double alpha = 1.0;
        check_cublas(cublasDgemmStridedBatched(
            blas_->get(), CUBLAS_OP_T, CUBLAS_OP_N,
            as_blas_int(d_.M, "semi-fused GEMM M"),
            as_blas_int(d_.N, "semi-fused GEMM N"),
            as_blas_int(used_rows_, "semi-fused GEMM K"),
            &alpha,
            L_[active_buffer_].get(),
            as_blas_int(selected_.public_plan.row_capacity,
                        "semi-fused GEMM ldL"),
            as_blas_stride(checked_mul(
                selected_.public_plan.row_capacity, d_.M, "stride L"),
                "stride L"),
            R_[active_buffer_].get(),
            as_blas_int(selected_.public_plan.row_capacity,
                        "semi-fused GEMM ldR"),
            as_blas_stride(checked_mul(
                selected_.public_plan.row_capacity, d_.N, "stride R"),
                "stride R"),
            &beta,
            output, as_blas_int(d_.M, "semi-fused GEMM ldC"),
            as_blas_stride(checked_mul(d_.M, d_.N, "stride C"), "stride C"),
            as_blas_int(ac, "semi-fused GEMM batch")),
            "semi-fused cublasDgemmStridedBatched");
    }

    void flush(std::uint64_t ac)
    {
        if (used_rows_ == 0) return;
        const std::uint64_t completed_rows = checked_mul(
            ac, used_rows_, "semi-fused completed progress rows");
        check_cuda(cudaEventRecord(
                       left_ready_[active_buffer_]->get(), left_stream_->get()),
                   "record semi-fused left buffer ready");
        check_cuda(cudaEventRecord(
                       right_ready_[active_buffer_]->get(), right_stream_->get()),
                   "record semi-fused right buffer ready");
        check_cuda(cudaStreamWaitEvent(
                       gemm_stream_->get(), left_ready_[active_buffer_]->get(), 0),
                   "GEMM waits for semi-fused left buffer");
        check_cuda(cudaStreamWaitEvent(
                       gemm_stream_->get(), right_ready_[active_buffer_]->get(), 0),
                   "GEMM waits for semi-fused right buffer");
        const bool compact_output = output_layout_.kind ==
            FullTensorGpuNativeOutputKind::CompactByI1;
        double* const gemm_output = compact_output
            ? dense_output_.get()
            : compact_output_.get();
        const double beta = !compact_output && output_accumulation_started_
            ? 1.0 : 0.0;
        {
            semifused_nvtx::ScopedRange gemm_range(
                options_.enable_nvtx, "b4/final/materialized-GEMM",
                UINT32_C(0xff5cb85c));
            const std::size_t event = profiler_.begin(
                ProfileStage::FinalGemm, gemm_stream_->get());
            execute_cublas_materialized_gemm(ac, beta, gemm_output);
            profiler_.end(event, gemm_stream_->get());
        }

        ++stats_.final_gemm_calls;
        stats_.final_gemm_matrices = checked_add(
            stats_.final_gemm_matrices, ac, "semi-fused GEMM matrices");

        check_cuda(cudaEventRecord(
                       free_[active_buffer_]->get(), gemm_stream_->get()),
                   "record semi-fused buffer free");
        in_flight_[active_buffer_] = true;

        if (compact_output) {
            const std::uint64_t compact = output_layout_.values_per_i1();
            semifused_nvtx::ScopedRange fold_range(
                options_.enable_nvtx, "b4/final/compact-fold",
                UINT32_C(0xff59a14f));
            const std::size_t event = profiler_.begin(
                ProfileStage::CompactFold, gemm_stream_->get());
            semifused_detail::accumulate_reflection_compact_batched_kernel<<<
                semifused_detail::launch_blocks(checked_mul(
                    ac, compact, "compact accumulation values")), 256, 0,
                gemm_stream_->get()>>>(
                    dense_output_.get(), rep_m_.get(), rep_n_.get(),
                    partner_m_.get(), partner_n_.get(), compact_output_.get(),
                    compact, d_.M, d_.N, ac);
            check_cuda(cudaGetLastError(),
                       "launch semi-fused compact accumulation");
            profiler_.end(event, gemm_stream_->get());
        } else {
            output_accumulation_started_ = true;
        }

        enqueue_completed_rows(completed_rows);

        active_buffer_ ^= 1u;
        used_rows_ = 0;
    }

    void enqueue_completed_rows(std::uint64_t completed_rows) noexcept
    {
        if (completed_rows == 0 || options_.progress_advance == nullptr) {
            return;
        }

        if (completed_rows >
            std::numeric_limits<std::uint64_t>::max() -
                unreported_progress_rows_) {
            // The advertised total is itself a uint64_t, so this can only be
            // reached after a broken accounting invariant.  Keep the fallback
            // monotonic without invoking undefined wraparound in a noexcept
            // reporting path.
            unreported_progress_rows_ =
                std::numeric_limits<std::uint64_t>::max();
            return;
        }
        completed_rows += unreported_progress_rows_;

        ProgressCompletion* const completion =
            new (std::nothrow) ProgressCompletion{
                options_.progress_advance,
                options_.progress_context,
                completed_rows};
        if (completion == nullptr) {
            unreported_progress_rows_ = completed_rows;
            return;
        }

        const cudaError_t status = cudaLaunchHostFunc(
            gemm_stream_->get(), publish_completed_rows, completion);
        if (status != cudaSuccess) {
            delete completion;
            unreported_progress_rows_ = completed_rows;
            return;
        }
        unreported_progress_rows_ = 0;
        progress_callbacks_pending_ = true;
    }

    void publish_unreported_rows_now() noexcept
    {
        if (unreported_progress_rows_ == 0 ||
            options_.progress_advance == nullptr) {
            return;
        }
        const std::uint64_t completed_rows = unreported_progress_rows_;
        unreported_progress_rows_ = 0;
        options_.progress_advance(options_.progress_context, completed_rows);
    }

    void process_group(
        const WorkGroup& group,
        std::uint64_t ac,
        const double* c_pointer)
    {
        const std::uint64_t end =
            group.descriptor_begin + group.descriptor_count;
        std::uint64_t begin = group.descriptor_begin;
        while (begin < end) {
            const std::uint64_t remaining =
                selected_.public_plan.row_capacity - used_rows_;
            std::uint64_t segment_end = begin;
            std::uint64_t segment_rows = 0;
            const std::uint8_t association =
                selected_.schedule.descriptors[begin].association;
            while (segment_end < end) {
                const SemiFusedQDescriptor& q =
                    selected_.schedule.descriptors[segment_end];
                if (q.association != association) break;
                const std::uint64_t rows = descriptor_rows(q, d_);
                if (rows > remaining - segment_rows) break;
                segment_rows += rows;
                ++segment_end;
            }
            if (segment_end == begin) {
                if (used_rows_ != 0) {
                    flush(ac);
                    continue;
                }
                throw std::logic_error(
                    "semi-fused descriptor exceeds planned row capacity");
            }
            launch_segment(
                group, begin, segment_end - begin, segment_rows, ac, c_pointer);
            begin = segment_end;
            if (used_rows_ == selected_.public_plan.row_capacity) {
                flush(ac);
            }
        }
    }

    void run_x(
        std::uint64_t a0,
        std::uint64_t ac,
        std::uint64_t x,
        std::uint64_t x_slot)
    {
        semifused_nvtx::ScopedRange range(
            options_.enable_nvtx, "b4/semifused/x-slice",
            UINT32_C(0xff7f7f7f));
        generate_boundary_and_t(a0, ac, x);
        const std::uint64_t group_begin =
            selected_.schedule.x_group_offsets[x];
        const std::uint64_t group_end =
            selected_.schedule.x_group_offsets[x + 1u];
        for (std::uint64_t gi = group_begin; gi < group_end; ++gi) {
            const WorkGroup& group = selected_.schedule.groups[gi];
            const double* c_pointer = nullptr;
            if (group.resident_c) {
                c_pointer = C_.get() + x_slot * d_.Cx;
            } else {
                ensure_streamed_c_available();
                check_cuda(cuda_detail::launch_generate_full_tensor_6j_c_slab(
                               recurrence_, x, group.il24,
                               group.l34_begin, group.l34_count, C_.get(),
                               failures_.get(), recurrence_plan_,
                               prep_stream_->get()),
                           "generate streamed semi-fused C");
                check_cuda(cudaEventRecord(c_ready_->get(), prep_stream_->get()),
                           "record streamed semi-fused C ready");
                check_cuda(cudaStreamWaitEvent(
                               left_stream_->get(), c_ready_->get(), 0),
                           "left producer waits for streamed C");
                check_cuda(cudaStreamWaitEvent(
                               right_stream_->get(), c_ready_->get(), 0),
                           "right producer waits for streamed C");
                c_pointer = C_.get();
            }
            process_group(group, ac, c_pointer);
            if (!group.resident_c) mark_streamed_c_consumed();
        }
    }

    void run_i1_tile(
        std::uint64_t a0,
        std::uint64_t ac,
        std::uint64_t x0,
        std::uint64_t x_count,
        bool first_x_slab)
    {
        semifused_nvtx::ScopedRange range(
            options_.enable_nvtx, "b4/semifused/i1-tile",
            UINT32_C(0xff8c564b));
        active_buffer_ = 0;
        used_rows_ = 0;
        in_flight_ = {{false, false}};
        boundary_buffers_in_use_ = false;
        c_in_use_ = false;
        output_accumulation_started_ = false;

        const std::uint64_t compact_values = checked_mul(
            ac, output_layout_.values_per_i1(), "compact memset values");
        check_cuda(cudaMemsetAsync(
                       compact_output_.get(), 0,
                       as_size(bytes_for(
                           compact_values, sizeof(double),
                           "compact output memset bytes"),
                           "compact output memset bytes"),
                       gemm_stream_->get()),
                   "semi-fused compact output memset");

        for (std::uint64_t slot = 0; slot < x_count; ++slot) {
            run_x(a0, ac, x0 + slot, slot);
        }

        flush(ac);

        check_cuda(cudaStreamSynchronize(prep_stream_->get()),
                   "semi-fused preparation synchronize");
        check_cuda(cudaStreamSynchronize(left_stream_->get()),
                   "semi-fused left producer synchronize");
        check_cuda(cudaStreamSynchronize(right_stream_->get()),
                   "semi-fused right producer synchronize");
        check_cuda(cudaStreamSynchronize(gemm_stream_->get()),
                   "semi-fused final synchronize");
        progress_callbacks_pending_ = false;
        publish_unreported_rows_now();
        in_flight_ = {{false, false}};
        boundary_buffers_in_use_ = false;
        c_in_use_ = false;
        check_failures("semi-fused contraction");

        const double* result = compact_output_.get();
        const std::uint64_t result_values = checked_mul(
            ac, output_layout_.values_per_i1(), "compact result values");

        const std::uint64_t host_offset = checked_mul(
            a0 - i1_begin_, output_layout_.values_per_i1(),
            "semi-fused host result offset");
        const std::uint64_t bytes = bytes_for(
            result_values, sizeof(double), "semi-fused D2H");
        double* destination = nullptr;
        if (first_x_slab) {
            destination = host_result_.data() + host_offset;
        } else {
            host_partial_.resize(as_size(result_values, "semi-fused host partial"));
            destination = host_partial_.data();
        }
        const auto download_begin = Clock::now();
        check_cuda(cudaMemcpyAsync(
                       destination, result,
                       as_size(bytes, "semi-fused D2H bytes"),
                       cudaMemcpyDeviceToHost, gemm_stream_->get()),
                   "semi-fused output D2H");
        check_cuda(cudaStreamSynchronize(gemm_stream_->get()),
                   "semi-fused D2H synchronize");
        if (!first_x_slab) {
            double* const accumulated = host_result_.data() + host_offset;
            const std::size_t count = as_size(
                result_values, "semi-fused host accumulation values");
            for (std::size_t i = 0; i < count; ++i) {
                accumulated[i] += host_partial_[i];
            }
        }
        stats_.download_seconds += std::chrono::duration<double>(
            Clock::now() - download_begin).count();
    }

    void check_failures(const char* stage)
    {
        std::uint64_t failures = 0;
        check_cuda(cudaMemcpy(
                       &failures, failures_.get(), sizeof(failures),
                       cudaMemcpyDeviceToHost),
                   "download semi-fused recurrence failures");
        if (failures != 0) {
            throw b4utils::NumericalFailure(
                std::string(stage) + " reported " +
                std::to_string(failures) + " failed 6j series");
        }
    }

    DeviceInfo device_;
    const B4Spins& spins_;
    const b4support::FullTensorInputSupportPlan& support_;
    const b4support::PackedFullTensorBoosters& boosters_;
    const FullTensorGpuNativeOutputLayout& output_layout_;
    FullTensorSemiFusedOptions options_{};
    SemiFusedDims d_{};
    const SelectedPlan& selected_;
    std::uint64_t i1_begin_ = 0;
    std::uint64_t i1_end_ = 0;

    std::unique_ptr<Stream> prep_stream_;
    std::unique_ptr<Stream> left_stream_;
    std::unique_ptr<Stream> right_stream_;
    std::unique_ptr<Stream> gemm_stream_;
    std::unique_ptr<BlasHandle> blas_;
    std::unique_ptr<Context> cutensor_t1_;
    std::unique_ptr<Context> cutensor_t3_;
    std::vector<CuTensorPlanPair> cutensor_plans_;
    std::array<std::unique_ptr<Event>, 2> left_ready_;
    std::array<std::unique_ptr<Event>, 2> right_ready_;
    std::array<std::unique_ptr<Event>, 2> free_;
    std::unique_ptr<Event> boundary_ready_;
    std::unique_ptr<Event> t1_built_;
    std::unique_ptr<Event> t3_built_;
    std::unique_ptr<Event> c_ready_;
    std::unique_ptr<Event> left_c_done_;
    std::unique_ptr<Event> right_c_done_;
    std::array<bool, 2> in_flight_{{false, false}};
    bool boundary_buffers_in_use_ = false;
    bool c_in_use_ = false;
    bool output_accumulation_started_ = false;
    bool progress_callbacks_pending_ = false;
    std::uint64_t unreported_progress_rows_ = 0;
    unsigned active_buffer_ = 0;
    std::uint64_t used_rows_ = 0;

    DeviceBuffer<double> dense_B2_, dense_B5_, mma_B3_, mma_B4_;
    DeviceBuffer<int> x_values_, i1_values_, k2_values_, k3_values_,
        k4_values_, k5_values_;
    DeviceCoherentRecurrenceProblem recurrence_{};
    FullTensorDeviceSixJLaunchPlan recurrence_plan_{};
    DeviceBuffer<double> boundary_A_, boundary_E_, boundary_A_virtual_fast_,
        boundary_E_virtual_fast_, C_;
    DeviceBuffer<double> T1_, T3_;
    DeviceBuffer<double> cutensor_t1_operand_, cutensor_t3_operand_;
    DeviceBuffer<double> cutensor_t1_logical_, cutensor_t3_logical_;
    DeviceBuffer<std::uint8_t> cutensor_t1_workspace_, cutensor_t3_workspace_;
    std::array<DeviceBuffer<double>, 2> L_, R_;
    DeviceBuffer<double> dense_output_, compact_output_;
    DeviceBuffer<std::uint64_t> failures_;
    DeviceBuffer<SemiFusedQDescriptor> descriptors_;
    DeviceBuffer<std::uint32_t> rep_m_, rep_n_, partner_m_, partner_n_;
    std::vector<double> host_result_;
    std::vector<double> host_partial_;
    FullTensorSemiFusedDeviceStats stats_{};
    DeferredProfiler profiler_;
};

} // namespace

bool full_tensor_semifused_cuda_available() noexcept
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool contract_full_tensor_gpu_semifused_generated(
    const B4Spins& spins,
    const b4support::FullTensorInputSupportPlan& support,
    const b4support::PackedFullTensorBoosters& boosters,
    const FullTensorGpuNativeOutputLayout& output_layout,
    const FullTensorSemiFusedOptions& options,
    bool zero_output,
    double* host_output,
    std::size_t host_output_values,
    FullTensorSemiFusedStats& stats,
    std::string& error) noexcept
{
    const auto total_begin = Clock::now();
    stats = {};
    error.clear();
    try {
        if (host_output == nullptr && host_output_values != 0) {
            throw std::invalid_argument("null semi-fused output pointer");
        }
        if (!(options.vram_fraction > 0.0) ||
            !std::isfinite(options.vram_fraction)) {
            throw std::invalid_argument(
                "semi-fused VRAM fraction must be finite and positive");
        }
        if (options.producer_waves_per_sm > 64u) {
            throw std::invalid_argument(
                "semi-fused producer waves per SM must not exceed 64");
        }

        const SemiFusedDims d = make_dims(support);
        const std::size_t values_per_i1 = output_layout.values_per_i1();
        if (values_per_i1 == 0) {
            throw std::invalid_argument("invalid semi-fused output layout");
        }
        const std::uint64_t expected = checked_mul(
            d.I1, values_per_i1, "semi-fused host output values");
        if (host_output_values != expected) {
            throw std::invalid_argument(
                "semi-fused host output size does not match layout");
        }

        std::vector<DeviceInfo> devices = discover_devices(options);
        if (devices.size() > d.I1) {
            devices.resize(as_size(d.I1, "device trim"));
        }
        const BaseSchedule base = build_base_schedule(
            support, output_layout, options.tree, d);
        if (output_layout.kind ==
                FullTensorGpuNativeOutputKind::CompactByI1 &&
            base.tree != FullTensorGpuNativeContractionTree::
                CanonicalReflectionOrbits) {
            throw std::invalid_argument(
                "compact semi-fused output requires the canonical "
                "reflection-orbit tree");
        }
        SelectedPlan selected = choose_plan(
            d, support, output_layout, options, base, devices);
        stats.plan = selected.public_plan;
        if (options.progress_begin != nullptr) {
            options.progress_begin(
                options.progress_context,
                checked_mul(
                    d.I1,
                    base.total_rows_per_i1,
                    "semi-fused total progress rows"));
        }
        struct ThreadResult {
            std::uint64_t i1_begin = 0;
            std::size_t commit_offset = 0;
            std::vector<double> output;
            FullTensorSemiFusedDeviceStats stats{};
            std::string error;
            std::exception_ptr failure{};
            FullTensorSemiFusedFailureKind failure_kind =
                FullTensorSemiFusedFailureKind::None;
        };
        std::vector<ThreadResult> results(devices.size());
        std::vector<std::thread> threads;
        threads.reserve(devices.size());

        const std::uint64_t base_count = d.I1 / devices.size();
        const std::uint64_t remainder = d.I1 % devices.size();
        std::uint64_t a_begin = 0;
        for (std::size_t i = 0; i < devices.size(); ++i) {
            const std::uint64_t count =
                base_count + (i < remainder ? 1u : 0u);
            const std::uint64_t begin = a_begin;
            const std::uint64_t end = begin + count;
            a_begin = end;
            results[i].i1_begin = begin;
            threads.emplace_back([&, i, begin, end] {
                try {
                    Worker worker(
                        devices[i], spins, support, boosters, output_layout,
                        options, d, selected, begin, end);
                    worker.run();
                    results[i].stats = worker.stats();
                    results[i].output =
                        std::move(worker).take_host_result();
                } catch (const std::exception& failure) {
                    results[i].error = failure.what();
                    results[i].failure = std::current_exception();
                    results[i].failure_kind = classify_failure(failure);
                } catch (...) {
                    results[i].error =
                        "unknown semi-fused full-tensor worker failure";
                    results[i].failure_kind =
                        FullTensorSemiFusedFailureKind::Fatal;
                }
            });
        }
        for (auto& thread : threads) thread.join();
        const ThreadResult* worker_failure = nullptr;
        for (const auto& result : results) {
            if (result.error.empty()) continue;
            if (worker_failure == nullptr ||
                result.failure_kind !=
                    FullTensorSemiFusedFailureKind::MemoryPressure) {
                worker_failure = &result;
            }
            if (result.failure_kind !=
                FullTensorSemiFusedFailureKind::MemoryPressure)
                break;
        }
        if (worker_failure != nullptr) {
            error = worker_failure->error;
            stats.failure = worker_failure->failure;
            stats.failure_kind = worker_failure->failure_kind;
            stats.total_seconds = std::chrono::duration<double>(
                Clock::now() - total_begin).count();
            return false;
        }

        // Complete every potentially throwing validation/allocation before
        // touching caller memory.  This makes a failed row-capacity attempt
        // safe to retry without a duplicate full-output scratch buffer.
        stats.devices.reserve(results.size());
        std::size_t commit_cursor = 0;
        for (auto& result : results) {
            const std::uint64_t begin = checked_mul(
                result.i1_begin, values_per_i1,
                "semi-fused commit offset");
            if (begin != commit_cursor || begin > host_output_values ||
                result.output.size() > host_output_values - begin) {
                throw std::logic_error(
                    "semi-fused worker outputs do not form one commit range");
            }
            result.commit_offset = as_size(
                begin, "semi-fused commit offset");
            commit_cursor += result.output.size();
        }
        if (commit_cursor != host_output_values) {
            throw std::logic_error(
                "semi-fused worker outputs do not cover the output tensor");
        }

        if (zero_output && host_output_values != 0) {
            std::fill(host_output, host_output + host_output_values, 0.0);
        }
        for (const auto& result : results) {
            if (zero_output) {
                std::copy(result.output.begin(), result.output.end(),
                          host_output + result.commit_offset);
            } else {
                for (std::size_t i = 0; i < result.output.size(); ++i) {
                    host_output[result.commit_offset + i] += result.output[i];
                }
            }
            stats.devices.push_back(result.stats);
        }
        stats.total_seconds =
            std::chrono::duration<double>(Clock::now() - total_begin).count();
        return true;
    } catch (const std::exception& failure) {
        error = failure.what();
        stats.failure = std::current_exception();
        stats.failure_kind = classify_failure(failure);
    } catch (...) {
        error = "unknown semi-fused CUDA backend failure";
        stats.failure_kind = FullTensorSemiFusedFailureKind::Fatal;
    }
    stats.total_seconds =
        std::chrono::duration<double>(Clock::now() - total_begin).count();
    return false;
}

} // namespace b4gpu
