// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "bridge.h"

#include <sl2cfoamxx.hpp>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace s = sl2cfoamxx;

struct SJXError {
    int32_t kind = SJX_RUNTIME;
    int32_t code = 0;
    int32_t system_code = 0;
    std::string message, path1, path2;
};

struct SJXTensor {
    s::FullTensor value;
    explicit SJXTensor(s::FullTensor tensor) : value(std::move(tensor)) {}
};

struct SJXProgressQueue {
    static constexpr std::size_t capacity = 256;
    std::array<SJXProgressEvent, capacity> events{};
    std::size_t begin = 0, size = 0;
    std::mutex mutex;
    std::condition_variable space;

    void push(const s::ProgressEvent& source)
    {
        const SJXProgressEvent event{
            static_cast<int32_t>(source.operation),
            static_cast<int32_t>(source.phase),
            static_cast<int32_t>(source.backend),
            static_cast<int32_t>(source.status), source.attempt,
            source.eta_available ? 1 : 0, source.completed, source.total,
            source.elapsed_seconds, source.eta_seconds};
        const auto running = static_cast<int32_t>(s::ProgressStatus::running);
        std::unique_lock lock(mutex);
        if (size != 0 && event.status == running) {
            auto& last = events[(begin + size - 1) % capacity];
            if (last.status == running && last.operation == event.operation &&
                last.phase == event.phase && last.attempt == event.attempt) {
                last = event;
                return;
            }
        }
        if (size == capacity) {
            // Running samples are replaceable; lifecycle events are not.
            std::size_t discard = 0;
            while (discard < size &&
                   events[(begin + discard) % capacity].status != running)
                ++discard;
            if (discard < size) {
                for (std::size_t i = discard; i + 1 < size; ++i)
                    events[(begin + i) % capacity] =
                        events[(begin + i + 1) % capacity];
                --size;
            } else if (event.status == running) {
                return;
            } else {
                // Only lifecycle-only saturation applies backpressure. The
                // Julia wrapper keeps polling even if its callback throws.
                space.wait(lock, [&] { return size < capacity; });
            }
        }
        events[(begin + size) % capacity] = event;
        ++size;
    }
};

namespace {

SJXError allocation_error;

SJXError* make_error(int32_t kind, const char* message, int32_t code = 0,
                     int32_t system_code = 0, const std::string& path1 = {},
                     const std::string& path2 = {}) noexcept
{
    try {
        auto error = std::make_unique<SJXError>();
        error->kind = kind;
        error->code = code;
        error->system_code = system_code;
        error->message = message;
        error->path1 = path1;
        error->path2 = path2;
        return error.release();
    } catch (...) {
        return &allocation_error;
    }
}

SJXError* capture_error() noexcept
{
    try {
        try { throw; }
        catch (const s::Error& error) {
            return make_error(SJX_LIBRARY, error.what(),
                              static_cast<int32_t>(error.code()));
        } catch (const std::filesystem::filesystem_error& error) {
            return make_error(SJX_FILESYSTEM, error.what(), 0,
                              error.code().value(), error.path1().string(),
                              error.path2().string());
        } catch (const std::invalid_argument& error) {
            return make_error(SJX_INVALID_ARGUMENT, error.what());
        } catch (const std::out_of_range& error) {
            return make_error(SJX_BOUNDS, error.what());
        } catch (const std::length_error& error) {
            return make_error(SJX_OVERFLOW, error.what());
        } catch (const std::overflow_error& error) {
            return make_error(SJX_OVERFLOW, error.what());
        } catch (const std::bad_alloc&) {
            return &allocation_error;
        } catch (const std::exception& error) {
            return make_error(SJX_RUNTIME, error.what());
        } catch (...) {
            return make_error(SJX_RUNTIME, "unknown native exception");
        }
    } catch (...) { return &allocation_error; }
}

template <class Function>
SJXError* protect(Function&& function) noexcept
{
    try { function(); return nullptr; }
    catch (...) { return capture_error(); }
}

template <class T> T& checked(T* pointer)
{
    if (pointer == nullptr) throw std::invalid_argument("null bridge argument");
    return *pointer;
}

std::size_t native_size(uint64_t value)
{
    if (value > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("value does not fit native size_t");
    return static_cast<std::size_t>(value);
}

template <class Enum> Enum enum_value(int32_t value, int32_t maximum)
{
    if (value < 0 || value > maximum)
        throw std::invalid_argument("unknown bridge enum value");
    return static_cast<Enum>(value);
}

bool flag(int32_t value)
{
    if (value != 0 && value != 1)
        throw std::invalid_argument("bridge boolean must be zero or one");
    return value != 0;
}

char* copy_string(std::string_view source)
{
    if (source.size() == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("string is too large");
    auto* result = static_cast<char*>(std::malloc(source.size() + 1));
    if (!result) throw std::bad_alloc();
    if (!source.empty()) std::memcpy(result, source.data(), source.size());
    result[source.size()] = '\0';
    return result;
}

const char* file_path(const char* value)
{
    checked(value);
    return value;
}

template <std::size_t N> std::array<int, N> spins(const int32_t* values)
{
    checked(values);
    std::array<int, N> result;
    for (std::size_t i = 0; i < N; ++i) {
        if (!std::in_range<int>(values[i]))
            throw std::overflow_error("doubled label does not fit C++ int");
        result[i] = static_cast<int>(values[i]);
    }
    return result;
}

template <std::size_t N>
std::array<s::NormalAngles, N> angles(const double* values)
{
    checked(values);
    std::array<s::NormalAngles, N> result;
    for (std::size_t i = 0; i < N; ++i)
        result[i] = {values[2 * i], values[2 * i + 1]};
    return result;
}

s::CoherentCoefficients coefficients(const SJXComplex* values, uint64_t count)
{
    const auto size = native_size(count);
    if (size) checked(values);
    s::CoherentCoefficients result(size);
    for (std::size_t i = 0; i < size; ++i)
        result[i] = {values[i].re, values[i].im};
    return result;
}

s::FullTensor::shape_type shape(const uint64_t* values)
{
    checked(values);
    s::FullTensor::shape_type result;
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = native_size(values[i]);
    return result;
}

template <class T, void (*Dispose)(T*)>
struct RecordGuard {
    T* output;
    explicit RecordGuard(T* value) : output(&checked(value)) { *output = {}; }
    ~RecordGuard() { if (output) Dispose(output); }
    void release() noexcept { output = nullptr; }
};

void export_config(const s::LibraryConfig& source, SJXConfig* output)
{
    RecordGuard<SJXConfig, sjx_config_dispose> guard(output);
    output->backend = static_cast<int32_t>(source.backend);
    output->progress_bar = source.progress_bar;
    output->simplicity_constraint = static_cast<int32_t>(source.simplicity_constraint);
    output->verbosity = static_cast<int32_t>(source.verbosity);
    output->printing_style = static_cast<int32_t>(source.printing_style);
    output->tensor_storage = static_cast<int32_t>(source.tensors.storage);
    output->reuse_existing = source.tensors.reuse_existing;
    output->check_closure = source.coherent_amplitude.check_closure;
    output->full_tensor_cpu_final_tile = static_cast<int32_t>(source.low_level.full_tensor_cpu_final_tile);
    output->booster_rho_tile = source.low_level.booster_rho_tile;
    output->coherent_gpu_pipeline_slots = source.low_level.coherent_gpu_pipeline_slots;
    output->full_tensor_gpu_producer_waves_per_sm = source.low_level.full_tensor_gpu_producer_waves_per_sm;
    output->full_tensor_cpu_p_batch = source.low_level.full_tensor_cpu_p_batch;
    output->cpu_threads = source.cpu_threads;
    output->memory_budget_bytes = source.memory_budget_bytes;
    output->maximum_payload_size_bytes = source.tensors.maximum_payload_size_bytes;
    output->full_tensor_gpu_row_capacity = source.low_level.full_tensor_gpu_row_capacity;
    output->pruning_ratio = source.coherent_amplitude.pruning_ratio;
    output->directory = copy_string(source.tensors.directory.string());
    output->output_path = copy_string(source.tensors.output_path.string());
    guard.release();
}

s::LibraryConfig import_config(const SJXConfig& source)
{
    s::LibraryConfig output;
    output.backend = enum_value<s::Backend>(source.backend, 1);
    output.progress_bar = flag(source.progress_bar);
    output.simplicity_constraint = enum_value<s::SimplicityConstraint>(source.simplicity_constraint, 1);
    output.verbosity = enum_value<s::Verbosity>(source.verbosity, 3);
    output.printing_style = enum_value<s::PrintingStyle>(source.printing_style, 1);
    output.tensors.storage = enum_value<s::TensorStorage>(source.tensor_storage, 3);
    output.tensors.reuse_existing = flag(source.reuse_existing);
    output.coherent_amplitude.check_closure = flag(source.check_closure);
    output.low_level.full_tensor_cpu_final_tile = enum_value<s::FullTensorCpuFinalTile>(source.full_tensor_cpu_final_tile, 3);
    output.low_level.booster_rho_tile = source.booster_rho_tile;
    output.low_level.coherent_gpu_pipeline_slots = source.coherent_gpu_pipeline_slots;
    output.low_level.full_tensor_gpu_producer_waves_per_sm = source.full_tensor_gpu_producer_waves_per_sm;
    output.low_level.full_tensor_cpu_p_batch = source.full_tensor_cpu_p_batch;
    output.cpu_threads = native_size(source.cpu_threads);
    output.memory_budget_bytes = source.memory_budget_bytes;
    output.tensors.maximum_payload_size_bytes = source.maximum_payload_size_bytes;
    output.low_level.full_tensor_gpu_row_capacity = source.full_tensor_gpu_row_capacity;
    output.coherent_amplitude.pruning_ratio = source.pruning_ratio;
    output.tensors.directory = source.directory ? source.directory : "";
    output.tensors.output_path = source.output_path ? source.output_path : "";
    return output;
}

void export_tensor_info(const s::FullTensor& tensor, SJXTensorInfo* output)
{
    RecordGuard<SJXTensorInfo, sjx_tensor_info_dispose> guard(output);
    const auto info = tensor.metadata();
    std::copy(info.logical_shape.begin(), info.logical_shape.end(), output->shape);
    std::copy(tensor.strides().begin(), tensor.strides().end(), output->strides);
    output->logical_values = info.logical_values;
    output->stored_values = info.stored_values;
    output->stored_bytes = tensor.stored_bytes();
    output->dense_bytes = tensor.dense_bytes();
    output->has_scientific = info.scientific.has_value();
    if (info.scientific) {
        const auto& scientific = *info.scientific;
        std::copy(scientific.spins.begin(), scientific.spins.end(), output->spins);
        output->Dl = scientific.Dl;
        output->gamma = scientific.gamma;
        output->simplicity_constraint = static_cast<int32_t>(scientific.simplicity_constraint);
    }
    output->storage = static_cast<int32_t>(info.storage);
    output->layout = static_cast<uint32_t>(info.layout);
    output->reassociated_layout = info.reassociated_layout
        ? static_cast<int32_t>(*info.reassociated_layout) : -1;
    output->mapped = info.mapped;
    output->writable = info.writable;
    output->symmetry_compressed = tensor.is_symmetry_compressed();
    output->dense = tensor.is_dense();
    output->path = copy_string(info.path.string());
    guard.release();
}

template <class Function>
SJXError* tensor_result(SJXTensor** output, Function&& compute) noexcept
{
    if (output) *output = nullptr;
    return protect([&] {
        checked(output);
        auto result = std::make_unique<SJXTensor>(compute());
        *output = result.release();
    });
}

s::ProgressCallback callback(SJXProgressQueue* queue)
{
    return [queue](const s::ProgressEvent& event) { queue->push(event); };
}

} // namespace

extern "C" {

uint32_t sjx_abi_version(void) noexcept { return 1; }
int32_t sjx_error_kind(const SJXError* error) noexcept
{ return error == &allocation_error ? SJX_ALLOCATION : (error ? error->kind : 0); }
int32_t sjx_error_code(const SJXError* error) noexcept
{ return error ? error->code : 0; }
int32_t sjx_error_system_code(const SJXError* error) noexcept
{ return error ? error->system_code : 0; }
const char* sjx_error_message(const SJXError* error) noexcept
{ return error == &allocation_error ? "native allocation failed" : (error ? error->message.c_str() : ""); }
const char* sjx_error_path1(const SJXError* error) noexcept
{ return error ? error->path1.c_str() : ""; }
const char* sjx_error_path2(const SJXError* error) noexcept
{ return error ? error->path2.c_str() : ""; }
void sjx_error_free(SJXError* error) noexcept
{ if (error != &allocation_error) delete error; }
void sjx_free(void* pointer) noexcept { std::free(pointer); }

SJXError* sjx_default_config(SJXConfig* output) noexcept
{ if (output) *output = {}; return protect([&] { export_config(s::default_config(), output); }); }
SJXError* sjx_current_config(SJXConfig* output) noexcept
{ if (output) *output = {}; return protect([&] { export_config(s::current_config(), output); }); }
SJXError* sjx_configure(const SJXConfig* config) noexcept
{ return protect([&] { s::configure(import_config(checked(config))); }); }
void sjx_config_dispose(SJXConfig* output) noexcept
{
    if (!output) return;
    std::free(output->directory); std::free(output->output_path); *output = {};
}

SJXError* sjx_build_info(SJXBuildInfo* output) noexcept
{
    if (output) *output = {};
    return protect([&] {
        RecordGuard<SJXBuildInfo, sjx_build_info_dispose> guard(output);
        const auto source = s::build_info();
        output->native_arch_enabled = source.native_arch_enabled;
        output->cuda_backend_built = source.cuda_backend_built;
        output->cuda_fast_math_requested = source.cuda_fast_math_requested;
#define SJX_COPY_BUILD_FIELD(name) output->name = copy_string(source.name)
        SJX_COPY_BUILD_FIELD(version); SJX_COPY_BUILD_FIELD(compiler_id);
        SJX_COPY_BUILD_FIELD(compiler_version); SJX_COPY_BUILD_FIELD(build_type);
        SJX_COPY_BUILD_FIELD(blas_vendor_selector); SJX_COPY_BUILD_FIELD(blas_provider);
        SJX_COPY_BUILD_FIELD(cuda_toolkit_version); SJX_COPY_BUILD_FIELD(cuda_architectures);
        SJX_COPY_BUILD_FIELD(cutensor_version);
#undef SJX_COPY_BUILD_FIELD
        guard.release();
    });
}
void sjx_build_info_dispose(SJXBuildInfo* output) noexcept
{
    if (!output) return;
    std::free(output->version); std::free(output->compiler_id);
    std::free(output->compiler_version); std::free(output->build_type);
    std::free(output->blas_vendor_selector); std::free(output->blas_provider);
    std::free(output->cuda_toolkit_version); std::free(output->cuda_architectures);
    std::free(output->cutensor_version); *output = {};
}
SJXError* sjx_backend_available(int32_t backend, int32_t* output) noexcept
{ return protect([&] { checked(output) = s::backend_available(enum_value<s::Backend>(backend, 1)); }); }
SJXError* sjx_cuda_device_count(uint64_t* output) noexcept
{ return protect([&] { checked(output) = s::cuda_device_count(); }); }
SJXError* sjx_cuda_device_info(uint64_t ordinal, SJXDeviceInfo* output) noexcept
{
    if (output) *output = {};
    return protect([&] {
        RecordGuard<SJXDeviceInfo, sjx_device_info_dispose> guard(output);
        const auto source = s::cuda_device_info(native_size(ordinal));
        output->ordinal = source.ordinal;
        output->total_memory_bytes = source.total_memory_bytes;
        output->compute_capability_major = source.compute_capability_major;
        output->compute_capability_minor = source.compute_capability_minor;
        output->multiprocessor_count = source.multiprocessor_count;
        output->warp_size = source.warp_size;
        output->max_threads_per_block = source.max_threads_per_block;
        output->name = copy_string(source.name);
        output->pci_bus_id = copy_string(source.pci_bus_id);
        guard.release();
    });
}
void sjx_device_info_dispose(SJXDeviceInfo* output) noexcept
{ if (output) { std::free(output->name); std::free(output->pci_bus_id); *output = {}; } }

SJXError* sjx_progress_create(SJXProgressQueue** output) noexcept
{
    if (output) *output = nullptr;
    return protect([&] { checked(output); *output = new SJXProgressQueue; });
}
SJXError* sjx_progress_poll(SJXProgressQueue* queue, SJXProgressEvent* output,
                          uint64_t capacity, uint64_t* count) noexcept
{
    if (count) *count = 0;
    return protect([&] {
        checked(count); checked(queue);
        if (capacity) checked(output);
        std::lock_guard lock(queue->mutex);
        const std::size_t n = std::min(native_size(capacity), queue->size);
        for (std::size_t i = 0; i < n; ++i)
            output[i] = queue->events[(queue->begin + i) % SJXProgressQueue::capacity];
        queue->begin = (queue->begin + n) % SJXProgressQueue::capacity;
        queue->size -= n;
        *count = n;
        queue->space.notify_all();
    });
}
void sjx_progress_free(SJXProgressQueue* queue) noexcept { delete queue; }

SJXError* sjx_boundary_angles(int32_t preset, double* output) noexcept
{
    return protect([&] {
        checked(output);
        if (preset != 0 && preset != 1)
            throw std::invalid_argument("unknown boundary angle preset");
        const auto& values = preset == 0 ? s::lorentzian_boundary_angles
                                         : s::euclidean_boundary_angles;
        for (std::size_t row = 0; row < values.size(); ++row) {
            output[2 * row] = values[row][0];
            output[2 * row + 1] = values[row][1];
        }
    });
}
SJXError* sjx_indices_from_intertwiners(const int32_t* boundary,
    const int32_t* intertwiners, uint64_t* output) noexcept
{
    return protect([&] {
        checked(output);
        const auto result = s::indices_from_intertwiners(spins<10>(boundary), spins<5>(intertwiners));
        std::copy(result.begin(), result.end(), output);
    });
}
SJXError* sjx_compute_amplitude(const int32_t* boundary, const int32_t* intertwiners,
    uint64_t Dl, double gamma, int32_t policy, double* output) noexcept
{
    return protect([&] {
        checked(output);
        *output = s::compute_amplitude(spins<10>(boundary), spins<5>(intertwiners),
            native_size(Dl), gamma, enum_value<s::AmplitudeCachePolicy>(policy, 1));
    });
}
SJXError* sjx_compute_coherent_amplitude(const int32_t* boundary,
    uint64_t Dl, double gamma, const double* normals, const int32_t* backend,
    SJXProgressQueue* queue, SJXComplex* output) noexcept
{
    return protect([&] {
        checked(output);
        const auto input = spins<10>(boundary);
        const auto directions = angles<20>(normals);
        if (queue && !backend)
            throw std::invalid_argument("an explicit backend is required with progress");
        std::complex<double> value;
        if (!backend) value = s::compute_coherent_amplitude(input, native_size(Dl), gamma, directions);
        else if (!queue) value = s::compute_coherent_amplitude(input, native_size(Dl), gamma,
            directions, enum_value<s::Backend>(*backend, 1));
        else value = s::compute_coherent_amplitude(input, native_size(Dl), gamma,
            directions, enum_value<s::Backend>(*backend, 1), callback(queue));
        *output = {value.real(), value.imag()};
    });
}
SJXError* sjx_compute_coherent_amplitude_coefficients(const int32_t* boundary,
    uint64_t Dl, double gamma,
    const SJXComplex* C1, uint64_t count1, const SJXComplex* C2, uint64_t count2,
    const SJXComplex* C3, uint64_t count3, const SJXComplex* C4, uint64_t count4,
    const SJXComplex* C5, uint64_t count5,
    const int32_t* backend, SJXProgressQueue* queue, SJXComplex* output) noexcept
{
    return protect([&] {
        checked(output);
        const auto input = spins<10>(boundary);
        const auto c1 = coefficients(C1, count1), c2 = coefficients(C2, count2),
                   c3 = coefficients(C3, count3), c4 = coefficients(C4, count4),
                   c5 = coefficients(C5, count5);
        if (queue && !backend)
            throw std::invalid_argument("an explicit backend is required with progress");
        std::complex<double> value;
        if (!backend) value = s::compute_coherent_amplitude(input, native_size(Dl), gamma,
            c1, c2, c3, c4, c5);
        else if (!queue) value = s::compute_coherent_amplitude(input, native_size(Dl), gamma,
            c1, c2, c3, c4, c5, enum_value<s::Backend>(*backend, 1));
        else value = s::compute_coherent_amplitude(input, native_size(Dl), gamma,
            c1, c2, c3, c4, c5, enum_value<s::Backend>(*backend, 1), callback(queue));
        *output = {value.real(), value.imag()};
    });
}
SJXError* sjx_compute_full_tensor(const int32_t* boundary, uint64_t Dl, double gamma,
    const int32_t* backend, SJXProgressQueue* queue, SJXTensor** output) noexcept
{
    return tensor_result(output, [&] {
        const auto input = spins<10>(boundary);
        if (queue && !backend)
            throw std::invalid_argument("an explicit backend is required with progress");
        if (!backend) return s::compute_full_tensor(input, native_size(Dl), gamma);
        if (!queue) return s::compute_full_tensor(input, native_size(Dl), gamma,
            enum_value<s::Backend>(*backend, 1));
        return s::compute_full_tensor(input, native_size(Dl), gamma,
            enum_value<s::Backend>(*backend, 1), callback(queue));
    });
}
SJXError* sjx_compute_coherent_coefficients(const int32_t* boundary,
    const double* normals, SJXComplex** output, uint64_t* count) noexcept
{
    if (output) *output = nullptr;
    if (count) *count = 0;
    return protect([&] {
        checked(output); checked(count);
        const auto values = s::compute_coherent_coefficients(spins<4>(boundary), angles<4>(normals));
        if (values.size() > std::numeric_limits<std::size_t>::max() / sizeof(SJXComplex))
            throw std::overflow_error("coefficient output size overflow");
        if (values.empty()) return;
        auto* result = static_cast<SJXComplex*>(std::malloc(values.size() * sizeof(SJXComplex)));
        if (!result) throw std::bad_alloc();
        for (std::size_t i = 0; i < values.size(); ++i)
            result[i] = {values[i].real(), values[i].imag()};
        *output = result;
        *count = values.size();
    });
}
SJXError* sjx_check_closure(const int32_t* boundary, const double* normals,
    double atol, double rtol, SJXClosureResult* output) noexcept
{
    if (output) *output = {};
    return protect([&] {
        checked(output);
        const auto result = s::check_closure(spins<10>(boundary), angles<20>(normals), atol, rtol);
        output->all_closed = result.all_closed;
        for (std::size_t i = 0; i < result.nodes.size(); ++i) {
            const auto& source = result.nodes[i];
            auto& node = output->nodes[i];
            std::copy(source.vector.begin(), source.vector.end(), node.vector);
            node.residual_norm = source.residual_norm;
            node.scale = source.scale;
            node.relative_residual = source.relative_residual;
            node.closed = source.closed;
        }
    });
}

SJXError* sjx_tensor_create(const uint64_t* dimensions, int32_t init,
    uint32_t layout, SJXTensor** output) noexcept
{
    return tensor_result(output, [&] {
        if (layout > 2) throw std::invalid_argument("unknown tensor layout");
        return s::FullTensor(shape(dimensions), enum_value<s::TensorInit>(init, 1),
                             static_cast<s::TensorLayout>(layout));
    });
}
SJXError* sjx_tensor_create_mapped(const char* path, const uint64_t* dimensions,
    int32_t init, uint32_t layout, SJXTensor** output) noexcept
{
    return tensor_result(output, [&] {
        if (layout > 2) throw std::invalid_argument("unknown tensor layout");
        return s::FullTensor::create_mapped(file_path(path), shape(dimensions),
            enum_value<s::TensorInit>(init, 1), static_cast<s::TensorLayout>(layout));
    });
}
SJXError* sjx_tensor_load(const char* path, SJXTensor** output) noexcept
{ return tensor_result(output, [&] { return s::FullTensor::load(file_path(path)); }); }
SJXError* sjx_tensor_map(const char* path, int32_t access, SJXTensor** output) noexcept
{ return tensor_result(output, [&] { return s::FullTensor::map(file_path(path), enum_value<s::TensorAccess>(access, 1)); }); }
void sjx_tensor_free(SJXTensor* tensor) noexcept { delete tensor; }
SJXError* sjx_tensor_info(const SJXTensor* tensor, SJXTensorInfo* output) noexcept
{ if (output) *output = {}; return protect([&] { export_tensor_info(checked(tensor).value, output); }); }
void sjx_tensor_info_dispose(SJXTensorInfo* output) noexcept
{ if (output) { std::free(output->path); *output = {}; } }
SJXError* sjx_tensor_file_info(const char* path, SJXTensorInfo* output) noexcept
{
    if (output) *output = {};
    return protect([&] {
        const auto tensor = s::FullTensor::map(file_path(path), s::TensorAccess::read_only);
        export_tensor_info(tensor, output);
    });
}
SJXError* sjx_tensor_get(const SJXTensor* tensor, const uint64_t* indices, double* output) noexcept
{ return protect([&] { checked(output) = checked(tensor).value.at(shape(indices)); }); }
SJXError* sjx_tensor_set(SJXTensor* tensor, const uint64_t* indices, double value) noexcept
{ return protect([&] { checked(tensor).value.at(shape(indices)) = value; }); }
SJXError* sjx_tensor_get_flat(const SJXTensor* tensor, uint64_t index, double* output) noexcept
{ return protect([&] { checked(output) = checked(tensor).value.at_flat(native_size(index)); }); }
SJXError* sjx_tensor_set_flat(SJXTensor* tensor, uint64_t index, double value) noexcept
{ return protect([&] { checked(tensor).value.at_flat(native_size(index)) = value; }); }
SJXError* sjx_tensor_payload(const SJXTensor* handle, const double** data,
    uint64_t* count, int32_t* writable) noexcept
{
    if (data) *data = nullptr;
    return protect([&] {
        checked(data); checked(count); checked(writable);
        const auto& tensor = checked(handle).value;
        *data = tensor.data(); *count = tensor.stored_size(); *writable = tensor.writable();
    });
}
SJXError* sjx_tensor_copy_values(const SJXTensor* handle, int32_t canonical,
    double* destination, uint64_t count) noexcept
{
    return protect([&] {
        const auto& tensor = checked(handle).value;
        const bool expand = flag(canonical);
        const auto expected = expand ? tensor.logical_size() : tensor.stored_size();
        if (count != expected) throw std::invalid_argument("tensor copy destination has incorrect size");
        if (count == 0) return;
        checked(destination);
        if (!expand || tensor.is_dense()) {
            std::copy_n(tensor.data(), expected, destination);
            return;
        }
        const auto& d = tensor.shape();
        std::size_t offset = 0;
        for (std::size_t i5 = 0; i5 < d[4]; ++i5)
        for (std::size_t i4 = 0; i4 < d[3]; ++i4)
        for (std::size_t i3 = 0; i3 < d[2]; ++i3)
        for (std::size_t i2 = 0; i2 < d[1]; ++i2)
        for (std::size_t i1 = 0; i1 < d[0]; ++i1)
            destination[offset++] = tensor.at(i1, i2, i3, i4, i5);
    });
}
SJXError* sjx_tensor_clone(const SJXTensor* tensor, SJXTensor** output) noexcept
{ return tensor_result(output, [&] { return checked(tensor).value.clone(); }); }
SJXError* sjx_tensor_dense_copy(const SJXTensor* tensor, SJXTensor** output) noexcept
{ return tensor_result(output, [&] { return checked(tensor).value.dense_copy(); }); }
SJXError* sjx_tensor_fill(SJXTensor* tensor, double value) noexcept
{ return protect([&] { checked(tensor).value.fill(value); }); }
SJXError* sjx_tensor_save(const SJXTensor* tensor, const char* path) noexcept
{ return protect([&] { checked(tensor).value.save(file_path(path)); }); }
SJXError* sjx_tensor_flush(const SJXTensor* tensor) noexcept
{ return protect([&] { checked(tensor).value.flush(); }); }
SJXError* sjx_tensor_convert_storage(const SJXTensor* tensor, int32_t storage,
    const char* path, int32_t overwrite, SJXTensor** output) noexcept
{
    return tensor_result(output, [&] {
        return checked(tensor).value.convert_storage(enum_value<s::TensorStorage>(storage, 3),
            file_path(path), flag(overwrite));
    });
}
SJXError* sjx_tensor_format(const SJXTensor* tensor, uint64_t maximum,
    int32_t precision, int32_t metadata, char** output) noexcept
{
    if (output) *output = nullptr;
    return protect([&] {
        checked(output);
        s::FullTensorPrintOptions options;
        options.max_values = maximum == UINT64_MAX ? s::FullTensorPrintOptions::all_values : native_size(maximum);
        options.precision = precision;
        options.include_metadata = flag(metadata);
        std::ostringstream text;
        checked(tensor).value.print(text, options);
        *output = copy_string(text.str());
    });
}
SJXError* sjx_tensor_tag(const SJXTensor* tensor, uint8_t* output, uint64_t count) noexcept
{
    return protect([&] {
        if (count != s::tensor_tag_bytes) throw std::invalid_argument("tensor tag buffer must contain 1024 bytes");
        checked(output);
        const auto& tag = checked(tensor).value.tag();
        std::copy(tag.begin(), tag.end(), output);
    });
}

} // extern "C"
