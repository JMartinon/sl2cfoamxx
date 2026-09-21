// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include <WolframLibrary.h>
#include <sl2cfoamxx.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

#if defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace s = sl2cfoamxx;
namespace {

constexpr const char* tensor_manager = "SL2CfoamXXTensor";
constexpr const char* job_manager = "SL2CfoamXXJob";

bool retain_image_for_thread_destructors() noexcept
{
#if defined(__APPLE__)
    // Wolfram unloads LibraryLink images before the kernel thread exits.
    // GCC's emulated TLS registers destructors in this image and in the C++
    // dependency, which dyld otherwise unmaps before invoking those callbacks.
    // Keep one process-lifetime loader reference, including its dependency
    // graph: NODELETE alone followed by dlclose does not retain their TLS code.
    // WolframLibrary_uninitialize still releases tensors and joins jobs.
    // Reuse the reference across LibraryLink reloads; the pointer has no destructor.
    static void* const resident_image = []() noexcept -> void* {
        static const char image_anchor = 0;
        Dl_info image{};
        if (!dladdr(&image_anchor, &image) || !image.dli_fname) return nullptr;
        return dlopen(image.dli_fname, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    }();
    return resident_image != nullptr;
#else
    return true;
#endif
}

// Only serialized return storage is thread-local. The kernel copies it before
// another call; error state belongs to this call or to an explicitly owned job.
thread_local std::string response;

struct Json {
    std::string text;
    void raw(std::string_view value) { text += value; }
    void string(std::string_view value)
    {
        text += '"';
        constexpr char hex[] = "0123456789abcdef";
        for (char byte : value) {
            const auto c = static_cast<unsigned char>(byte);
            if (c == '"' || c == '\\') { text += '\\'; text += static_cast<char>(c); }
            else if (c < 0x20) {
                text += "\\u00"; text += hex[c >> 4]; text += hex[c & 15];
            } else text += static_cast<char>(c);
        }
        text += '"';
    }
    void number(double value)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("non-finite scalar cannot be represented in JSON; use TensorArray for non-finite tensor data");
        char buffer[64];
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                          std::chars_format::general, 17);
        if (result.ec != std::errc{}) throw std::overflow_error("number formatting failed");
        text.append(buffer, result.ptr);
        // RawJSON distinguishes an integer token from a machine-real token.
        // Preserve the C++ scalar type even for exact-valued doubles and -0.
        if (std::string_view(buffer, result.ptr).find_first_of(".eE") == std::string_view::npos)
            text += ".0";
    }
    template <class T> void integer(T value)
    {
        char buffer[32];
        const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
        if (result.ec != std::errc{}) throw std::overflow_error("integer formatting failed");
        text.append(buffer, result.ptr);
    }
    void boolean(bool value) { raw(value ? "true" : "false"); }
    void key(std::string_view name) { string(name); raw(":"); }
    template <class Range> void integers(const Range& values)
    {
        raw("["); bool first = true;
        for (const auto value : values) { if (!first) raw(","); first = false; integer(value); }
        raw("]");
    }
    void complex(std::complex<double> value)
    { raw("["); number(value.real()); raw(","); number(value.imag()); raw("]"); }
};

void error_json(Json& json, std::string_view kind, std::string_view message,
                int code = 0, int system_code = 0,
                std::string_view path1 = {}, std::string_view path2 = {})
{
    json.raw("{\"ok\":false,\"error\":{");
    json.key("kind"); json.string(kind);
    json.raw(","); json.key("code"); json.integer(code);
    json.raw(","); json.key("message"); json.string(message);
    json.raw(","); json.key("system_code"); json.integer(system_code);
    json.raw(","); json.key("path1"); json.string(path1);
    json.raw(","); json.key("path2"); json.string(path2);
    json.raw("}}");
}

Json current_error()
{
    Json result;
    try { throw; }
    catch (const s::Error& error) {
        error_json(result, "Library", error.what(), static_cast<int>(error.code()));
    } catch (const std::filesystem::filesystem_error& error) {
        error_json(result, "Filesystem", error.what(), 0, error.code().value(),
                   error.path1().string(), error.path2().string());
    } catch (const std::invalid_argument& error) { error_json(result, "InvalidArgument", error.what()); }
      catch (const std::out_of_range& error) { error_json(result, "Bounds", error.what()); }
      catch (const std::length_error& error) { error_json(result, "Overflow", error.what()); }
      catch (const std::overflow_error& error) { error_json(result, "Overflow", error.what()); }
      catch (const std::bad_alloc&) { error_json(result, "Allocation", "native allocation failed"); }
      catch (const std::exception& error) { error_json(result, "Runtime", error.what()); }
      catch (...) { error_json(result, "Runtime", "unknown native exception"); }
    return result;
}

template <class T> T nonnegative(mint value)
{
    if (value < 0) throw std::invalid_argument("expected a nonnegative integer");
    if (!std::in_range<T>(value)) throw std::overflow_error("integer exceeds native field range");
    return static_cast<T>(value);
}
template <class T> T enumeration(mint value, mint maximum)
{
    if (value < 0 || value > maximum) throw std::invalid_argument("unknown enum value");
    return static_cast<T>(value);
}
bool flag(mint value) { return enumeration<int>(value, 1) != 0; }
std::size_t index(mint value)
{
    if (value <= 0) throw std::out_of_range("tensor indices are one-based positive integers");
    return nonnegative<std::size_t>(value - 1);
}

struct Arguments {
    const mint* integers;
    const double* reals;
    std::size_t ni, nr;
    std::string_view text1, text2;
    Arguments(WolframLibraryData library, MArgument* args)
    {
        const auto ints = MArgument_getMTensor(args[1]);
        const auto nums = MArgument_getMTensor(args[2]);
        if (!ints || !nums || library->MTensor_getType(ints) != MType_Integer ||
            library->MTensor_getType(nums) != MType_Real ||
            library->MTensor_getRank(ints) != 1 || library->MTensor_getRank(nums) != 1)
            throw std::invalid_argument("dispatch requires rank-one integer and real arrays");
        ni = nonnegative<std::size_t>(library->MTensor_getFlattenedLength(ints));
        nr = nonnegative<std::size_t>(library->MTensor_getFlattenedLength(nums));
        integers = library->MTensor_getIntegerData(ints);
        reals = library->MTensor_getRealData(nums);
        const auto* first = MArgument_getUTF8String(args[3]);
        const auto* second = MArgument_getUTF8String(args[4]);
        if (!first || !second) throw std::invalid_argument("null dispatch string");
        text1 = first; text2 = second;
    }
    void sizes(std::size_t ints, std::size_t nums, int strings = 0) const
    {
        if (ni != ints || nr != nums) throw std::invalid_argument("incorrect dispatch vector length");
        if ((strings == 0 && !text1.empty()) || (strings < 2 && !text2.empty()))
            throw std::invalid_argument("unexpected dispatch string");
    }
    template <std::size_t N> std::array<int, N> spins(std::size_t start = 0) const
    {
        std::array<int, N> result;
        for (std::size_t i = 0; i < N; ++i) result[i] = nonnegative<int>(integers[start + i]);
        return result;
    }
    template <std::size_t N> std::array<s::NormalAngles, N> angles(std::size_t start = 0) const
    {
        std::array<s::NormalAngles, N> result;
        for (std::size_t i = 0; i < N; ++i) result[i] = {reals[start + 2*i], reals[start + 2*i + 1]};
        return result;
    }
    std::array<s::CoherentCoefficients, 5> coefficients(std::size_t start) const
    {
        sizes(start + 5, nr);
        if (nr == 0) throw std::invalid_argument("missing coherent amplitude gamma");
        std::array<std::size_t, 5> counts;
        std::size_t offset = 1;
        for (std::size_t n = 0; n < counts.size(); ++n) {
            counts[n] = nonnegative<std::size_t>(integers[start + n]);
            if (counts[n] > (nr - offset) / 2)
                throw std::invalid_argument("coefficient values do not match vector lengths");
            offset += 2 * counts[n];
        }
        if (offset != nr)
            throw std::invalid_argument("coefficient values do not match vector lengths");
        std::array<s::CoherentCoefficients, 5> result;
        offset = 1;
        for (std::size_t n = 0; n < result.size(); ++n) {
            result[n].reserve(counts[n]);
            for (std::size_t k = 0; k < counts[n]; ++k, offset += 2)
                result[n].emplace_back(reals[offset], reals[offset + 1]);
        }
        return result;
    }
    s::FullTensor::shape_type shape(std::size_t start) const
    {
        s::FullTensor::shape_type result;
        for (std::size_t i = 0; i < 5; ++i) result[i] = nonnegative<std::size_t>(integers[start + i]);
        return result;
    }
    s::FullTensor::shape_type indices(std::size_t start) const
    {
        s::FullTensor::shape_type result;
        for (std::size_t i = 0; i < 5; ++i) result[i] = index(integers[start + i]);
        return result;
    }
};

struct TensorSlot {
    std::mutex mutex;
    std::shared_ptr<s::FullTensor> tensor;
    bool live = true, reserved = false;
};
struct Reservation {
    std::shared_ptr<TensorSlot> slot;
    explicit Reservation(std::shared_ptr<TensorSlot> value) : slot(std::move(value))
    {
        std::lock_guard lock(slot->mutex);
        if (!slot->live || slot->tensor || slot->reserved)
            throw std::invalid_argument("destination tensor must be live and empty");
        slot->reserved = true;
    }
    ~Reservation() { std::lock_guard lock(slot->mutex); slot->reserved = false; }
    void commit(std::shared_ptr<s::FullTensor> tensor)
    {
        std::lock_guard lock(slot->mutex);
        if (!slot->live) throw std::invalid_argument("destination tensor was released during computation");
        slot->tensor = std::move(tensor);
    }
};

struct Job {
    static constexpr std::size_t capacity = 256;
    std::mutex control, mutex;
    std::condition_variable space;
    std::array<s::ProgressEvent, capacity> events{};
    std::size_t begin = 0, size = 0;
    bool collecting = true, started = false, consumed = false;
    std::atomic<bool> done{false};
    std::thread worker;
    std::unique_ptr<Reservation> destination;
    std::shared_ptr<s::FullTensor> tensor;
    std::complex<double> amplitude{};
    std::exception_ptr failure;

    void push(const s::ProgressEvent& event)
    {
        std::unique_lock lock(mutex);
        if (!collecting) return;
        const auto running = s::ProgressStatus::running;
        if (size && event.status == running) {
            auto& last = events[(begin + size - 1) % capacity];
            if (last.status == running && last.operation == event.operation &&
                last.phase == event.phase && last.attempt == event.attempt) { last = event; return; }
        }
        if (size == capacity) {
            std::size_t discard = 0;
            while (discard < size && events[(begin + discard) % capacity].status != running) ++discard;
            if (discard < size) {
                for (std::size_t i = discard; i + 1 < size; ++i)
                    events[(begin + i) % capacity] = events[(begin + i + 1) % capacity];
                --size;
            } else if (event.status == running) return;
            else {
                space.wait(lock, [&] { return !collecting || size < capacity; });
                if (!collecting) return;
            }
        }
        events[(begin + size) % capacity] = event; ++size;
    }
    void stop_collecting()
    { std::lock_guard lock(mutex); collecting = false; space.notify_all(); }
    void join()
    {
        // Called while holding control; never require the kernel to poll
        // while it is joining during finalization, failure, or early finish.
        stop_collecting();
        if (worker.joinable()) worker.join();
    }
    ~Job() { std::lock_guard lock(control); join(); }
};

std::mutex registry_mutex;
std::unordered_map<mint, std::shared_ptr<TensorSlot>> tensors;
std::unordered_map<mint, std::shared_ptr<Job>> jobs;

std::shared_ptr<TensorSlot> tensor_slot(mint id)
{
    std::lock_guard lock(registry_mutex);
    const auto entry = tensors.find(id);
    if (entry == tensors.end()) throw std::invalid_argument("unknown or released tensor handle");
    return entry->second;
}
std::shared_ptr<s::FullTensor> tensor(mint id)
{
    const auto slot = tensor_slot(id);
    std::lock_guard lock(slot->mutex);
    if (!slot->live || !slot->tensor) throw std::invalid_argument("tensor has no value or was released");
    return slot->tensor;
}
std::shared_ptr<Job> job(mint id)
{
    std::lock_guard lock(registry_mutex);
    const auto entry = jobs.find(id);
    if (entry == jobs.end()) throw std::invalid_argument("unknown or released job handle");
    return entry->second;
}

void manage_tensor(WolframLibraryData, mbool mode, mint id) noexcept
{
    try {
        if (!mode) {
            auto value = std::make_shared<TensorSlot>();
            std::lock_guard lock(registry_mutex); tensors.emplace(id, std::move(value));
        } else {
            std::shared_ptr<TensorSlot> value;
            { std::lock_guard lock(registry_mutex);
              const auto entry = tensors.find(id);
              if (entry != tensors.end()) { value = std::move(entry->second); tensors.erase(entry); } }
            if (value) { std::lock_guard lock(value->mutex); value->live = false; }
        }
    } catch (...) { /* A void kernel manager must never propagate C++ exceptions. */ }
}
void manage_job(WolframLibraryData, mbool mode, mint id) noexcept
{
    try {
        if (!mode) {
            auto value = std::make_shared<Job>();
            std::lock_guard lock(registry_mutex); jobs.emplace(id, std::move(value));
        } else {
            std::shared_ptr<Job> value;
            { std::lock_guard lock(registry_mutex);
              const auto entry = jobs.find(id);
              if (entry != jobs.end()) { value = std::move(entry->second); jobs.erase(entry); } }
            // Join outside the registry mutex; workers do not touch the registry.
        }
    } catch (...) { }
}

void metadata(Json& json, const s::FullTensor& tensor)
{
    const auto info = tensor.metadata();
    json.raw("{"); json.key("shape"); json.integers(info.logical_shape);
    json.raw(","); json.key("strides"); json.integers(tensor.strides());
#define SFX_META_INT(name, value) json.raw(",\"" name "\":"); json.integer(value)
    SFX_META_INT("logical_values", info.logical_values);
    SFX_META_INT("stored_values", info.stored_values);
    SFX_META_INT("stored_bytes", tensor.stored_bytes());
    SFX_META_INT("dense_bytes", tensor.dense_bytes());
    SFX_META_INT("storage", static_cast<int>(info.storage));
    SFX_META_INT("layout", static_cast<std::uint32_t>(info.layout));
#undef SFX_META_INT
    json.raw(",\"reassociated_layout\":");
    if (info.reassociated_layout) json.integer(static_cast<int>(*info.reassociated_layout)); else json.raw("null");
    json.raw(",\"mapped\":"); json.boolean(info.mapped);
    json.raw(",\"writable\":"); json.boolean(info.writable);
    json.raw(",\"symmetry_compressed\":"); json.boolean(tensor.is_symmetry_compressed());
    json.raw(",\"dense\":"); json.boolean(tensor.is_dense());
    json.raw(",\"path\":"); if (info.path.empty()) json.raw("null"); else json.string(info.path.string());
    json.raw(",\"scientific\":");
    if (info.scientific) {
        const auto& scientific = *info.scientific;
        json.raw("{\"spins\":"); json.integers(scientific.spins);
        json.raw(",\"Dl\":"); json.integer(scientific.Dl);
        json.raw(",\"gamma\":"); json.number(scientific.gamma);
        json.raw(",\"simplicity_constraint\":"); json.integer(static_cast<int>(scientific.simplicity_constraint));
        json.raw("}");
    } else json.raw("null");
    json.raw("}");
}

void config(Json& json, const s::LibraryConfig& config)
{
    const std::array<std::uint64_t, 17> ints{
        static_cast<unsigned>(config.backend), config.progress_bar,
        static_cast<unsigned>(config.simplicity_constraint), static_cast<unsigned>(config.verbosity),
        static_cast<unsigned>(config.printing_style), static_cast<unsigned>(config.tensors.storage),
        config.tensors.reuse_existing, config.coherent_amplitude.check_closure,
        static_cast<unsigned>(config.low_level.full_tensor_cpu_final_tile),
        config.low_level.booster_rho_tile, config.low_level.coherent_gpu_pipeline_slots,
        config.low_level.full_tensor_gpu_producer_waves_per_sm, config.low_level.full_tensor_cpu_p_batch,
        config.cpu_threads, config.memory_budget_bytes, config.tensors.maximum_payload_size_bytes,
        config.low_level.full_tensor_gpu_row_capacity};
    json.raw("{\"ints\":"); json.integers(ints);
    json.raw(",\"reals\":["); json.number(config.coherent_amplitude.pruning_ratio);
    json.raw("],\"directory\":"); json.string(config.tensors.directory.string());
    json.raw(",\"output_path\":"); json.string(config.tensors.output_path.string()); json.raw("}");
}
s::LibraryConfig config(const Arguments& args)
{
    args.sizes(17, 1, 2); const auto* i = args.integers;
    s::LibraryConfig value;
    value.backend = enumeration<s::Backend>(i[0], 1); value.progress_bar = flag(i[1]);
    value.simplicity_constraint = enumeration<s::SimplicityConstraint>(i[2], 1);
    value.verbosity = enumeration<s::Verbosity>(i[3], 3);
    value.printing_style = enumeration<s::PrintingStyle>(i[4], 1);
    value.tensors.storage = enumeration<s::TensorStorage>(i[5], 3);
    value.tensors.reuse_existing = flag(i[6]); value.coherent_amplitude.check_closure = flag(i[7]);
    value.low_level.full_tensor_cpu_final_tile = enumeration<s::FullTensorCpuFinalTile>(i[8], 3);
    value.low_level.booster_rho_tile = nonnegative<std::uint32_t>(i[9]);
    value.low_level.coherent_gpu_pipeline_slots = nonnegative<std::uint32_t>(i[10]);
    value.low_level.full_tensor_gpu_producer_waves_per_sm = nonnegative<std::uint32_t>(i[11]);
    value.low_level.full_tensor_cpu_p_batch = nonnegative<std::uint32_t>(i[12]);
    value.cpu_threads = nonnegative<std::size_t>(i[13]);
    value.memory_budget_bytes = nonnegative<std::uint64_t>(i[14]);
    value.tensors.maximum_payload_size_bytes = nonnegative<std::uint64_t>(i[15]);
    value.low_level.full_tensor_gpu_row_capacity = nonnegative<std::uint64_t>(i[16]);
    value.coherent_amplitude.pruning_ratio = args.reals[0];
    value.tensors.directory = args.text1; value.tensors.output_path = args.text2;
    return value;
}

void event_json(Json& json, const s::ProgressEvent& event)
{
    json.raw("{\"operation\":"); json.integer(static_cast<int>(event.operation));
    json.raw(",\"phase\":"); json.integer(static_cast<int>(event.phase));
    json.raw(",\"backend\":"); json.integer(static_cast<int>(event.backend));
    json.raw(",\"status\":"); json.integer(static_cast<int>(event.status));
    json.raw(",\"attempt\":"); json.integer(event.attempt);
    json.raw(",\"completed\":"); json.integer(event.completed);
    json.raw(",\"total\":"); json.integer(event.total);
    json.raw(",\"elapsed_seconds\":"); json.number(event.elapsed_seconds);
    json.raw(",\"eta_seconds\":"); json.number(event.eta_seconds);
    json.raw(",\"eta_available\":"); json.boolean(event.eta_available); json.raw("}");
}

template <class Function>
void produce(Json& json, mint destination, Function&& compute)
{
    Reservation reservation(tensor_slot(destination));
    auto result = std::make_shared<s::FullTensor>(compute());
    metadata(json, *result); // Serialize before committing, so failure is atomic.
    reservation.commit(std::move(result));
}

s::FullTensor full(const s::Spins& spins, std::size_t Dl, double gamma, mint backend)
{
    if (backend == -1) return s::compute_full_tensor(spins, Dl, gamma);
    return s::compute_full_tensor(spins, Dl, gamma, enumeration<s::Backend>(backend, 1));
}
std::complex<double> coherent(const s::Spins& spins, std::size_t Dl, double gamma,
                             const s::BoundaryAngles& angles, mint backend)
{
    if (backend == -1) return s::compute_coherent_amplitude(spins, Dl, gamma, angles);
    return s::compute_coherent_amplitude(spins, Dl, gamma, angles, enumeration<s::Backend>(backend, 1));
}
std::complex<double> coherent(const s::Spins& spins, std::size_t Dl, double gamma,
                             const std::array<s::CoherentCoefficients, 5>& coefficients, mint backend)
{
    const auto& c = coefficients;
    if (backend == -1)
        return s::compute_coherent_amplitude(spins, Dl, gamma, c[0], c[1], c[2], c[3], c[4]);
    return s::compute_coherent_amplitude(spins, Dl, gamma, c[0], c[1], c[2], c[3], c[4],
                                         enumeration<s::Backend>(backend, 1));
}

void dispatch(Json& json, WolframLibraryData library, mint opcode, const Arguments& args)
{
    const auto* i = args.integers; const auto* r = args.reals;
    switch (opcode) {
    case 0: args.sizes(0,0); json.integer(1); break;
    case 1: args.sizes(0,0); config(json, s::default_config()); break;
    case 2: args.sizes(0,0); config(json, s::current_config()); break;
    case 3: s::configure(config(args)); json.raw("null"); break;
    case 4: {
        args.sizes(0,0); const auto info = s::build_info(); json.raw("{");
#define SFX_BUILD_STRING(name) json.key(#name); json.string(info.name); json.raw(",")
        SFX_BUILD_STRING(version); SFX_BUILD_STRING(compiler_id); SFX_BUILD_STRING(compiler_version);
        SFX_BUILD_STRING(build_type); SFX_BUILD_STRING(blas_vendor_selector); SFX_BUILD_STRING(blas_provider);
        SFX_BUILD_STRING(cuda_toolkit_version); SFX_BUILD_STRING(cuda_architectures); SFX_BUILD_STRING(cutensor_version);
#undef SFX_BUILD_STRING
        json.key("native_arch_enabled"); json.boolean(info.native_arch_enabled);
        json.raw(",\"cuda_backend_built\":"); json.boolean(info.cuda_backend_built);
        json.raw(",\"cuda_fast_math_requested\":"); json.boolean(info.cuda_fast_math_requested); json.raw("}"); break;
    }
    case 5: args.sizes(1,0); json.boolean(s::backend_available(enumeration<s::Backend>(i[0],1))); break;
    case 6: args.sizes(0,0); json.integer(s::cuda_device_count()); break;
    case 7: {
        args.sizes(1,0); const auto info = s::cuda_device_info(nonnegative<std::size_t>(i[0]));
        json.raw("{\"ordinal\":"); json.integer(info.ordinal);
        json.raw(",\"name\":"); json.string(info.name);
        json.raw(",\"pci_bus_id\":"); json.string(info.pci_bus_id);
#define SFX_DEVICE_INT(name) json.raw(","); json.key(#name); json.integer(info.name)
        SFX_DEVICE_INT(compute_capability_major); SFX_DEVICE_INT(compute_capability_minor);
        SFX_DEVICE_INT(total_memory_bytes); SFX_DEVICE_INT(multiprocessor_count);
        SFX_DEVICE_INT(warp_size); SFX_DEVICE_INT(max_threads_per_block);
#undef SFX_DEVICE_INT
        json.raw("}"); break;
    }
    case 8: {
        args.sizes(1,0);
        const auto& angles = enumeration<int>(i[0],1) == 0
            ? s::lorentzian_boundary_angles : s::euclidean_boundary_angles;
        json.raw("[");
        for (std::size_t n=0;n<angles.size();++n) {
            if(n) json.raw(",");
            json.raw("["); json.number(angles[n][0]);
            json.raw(","); json.number(angles[n][1]); json.raw("]");
        }
        json.raw("]"); break;
    }
    case 10:
        args.sizes(17,1); json.number(s::compute_amplitude(args.spins<10>(2), args.spins<5>(12),
            nonnegative<std::size_t>(i[0]), r[0], enumeration<s::AmplitudeCachePolicy>(i[1],1))); break;
    case 11:
        args.sizes(12,41); json.complex(coherent(args.spins<10>(2), nonnegative<std::size_t>(i[0]),
            r[0], args.angles<20>(1), i[1])); break;
    case 12:
        args.sizes(13,1); produce(json, i[0], [&] {
            return full(args.spins<10>(3), nonnegative<std::size_t>(i[1]), r[0], i[2]); }); break;
    case 16: {
        const auto coefficients = args.coefficients(12);
        json.complex(coherent(args.spins<10>(2), nonnegative<std::size_t>(i[0]),
                              r[0], coefficients, i[1])); break;
    }
    case 13: {
        args.sizes(4,8); const auto result = s::compute_coherent_coefficients(args.spins<4>(), args.angles<4>());
        json.raw("["); bool first = true;
        for (const auto value : result) { if (!first) json.raw(","); first = false; json.complex(value); }
        json.raw("]"); break;
    }
    case 14: {
        args.sizes(10,42); const auto result = s::check_closure(args.spins<10>(), args.angles<20>(2), r[0], r[1]);
        json.raw("{\"all_closed\":"); json.boolean(result.all_closed); json.raw(",\"nodes\":[");
        bool first = true;
        for (const auto& node : result.nodes) {
            if (!first) json.raw(",");
            first = false;
            json.raw("{\"vector\":[");
            for (std::size_t n=0;n<3;++n) { if(n) json.raw(","); json.number(node.vector[n]); }
            json.raw("],\"residual_norm\":"); json.number(node.residual_norm);
            json.raw(",\"scale\":"); json.number(node.scale);
            json.raw(",\"relative_residual\":"); json.number(node.relative_residual);
            json.raw(",\"closed\":"); json.boolean(node.closed); json.raw("}");
        }
        json.raw("]}"); break;
    }
    case 15: {
        args.sizes(15,0); auto result = s::indices_from_intertwiners(args.spins<10>(), args.spins<5>(10));
        for (auto& value : result) ++value;
        json.integers(result); break;
    }
    case 20: case 21: {
        args.sizes(8,0,opcode == 21 ? 1 : 0);
        const auto init = enumeration<s::TensorInit>(i[1],1);
        const auto layout = enumeration<s::TensorLayout>(i[2],2); const auto shape = args.shape(3);
        produce(json, i[0], [&] { return opcode == 20 ? s::FullTensor(shape,init,layout) :
            s::FullTensor::create_mapped(args.text1,shape,init,layout); }); break;
    }
    case 22: args.sizes(1,0,1); produce(json,i[0],[&]{return s::FullTensor::load(args.text1);}); break;
    case 23: args.sizes(2,0,1); produce(json,i[0],[&]{return s::FullTensor::map(args.text1,enumeration<s::TensorAccess>(i[1],1));}); break;
    case 24: args.sizes(1,0); metadata(json,*tensor(i[0])); break;
    case 25: args.sizes(0,0,1); metadata(json,s::FullTensor::map(args.text1,s::TensorAccess::read_only)); break;
    case 26: args.sizes(6,0); json.number(std::as_const(*tensor(i[0])).at(args.indices(1))); break;
    case 27: args.sizes(6,1); tensor(i[0])->at(args.indices(1))=r[0]; json.raw("null"); break;
    case 28: args.sizes(2,0); json.number(std::as_const(*tensor(i[0])).at_flat(index(i[1]))); break;
    case 29: args.sizes(2,1); tensor(i[0])->at_flat(index(i[1]))=r[0]; json.raw("null"); break;
    case 30: args.sizes(1,1); tensor(i[0])->fill(r[0]); json.raw("null"); break;
    case 31: args.sizes(2,0); produce(json,i[0],[&]{return tensor(i[1])->clone();}); break;
    case 32: args.sizes(2,0); produce(json,i[0],[&]{return tensor(i[1])->dense_copy();}); break;
    case 33: args.sizes(1,0,1); tensor(i[0])->save(args.text1); json.raw("null"); break;
    case 34: args.sizes(1,0); tensor(i[0])->flush(); json.raw("null"); break;
    case 35:
        args.sizes(4,0,1); produce(json,i[0],[&]{return tensor(i[1])->convert_storage(
            enumeration<s::TensorStorage>(i[2],3),args.text1,flag(i[3]));}); break;
    case 36: {
        args.sizes(4,0); s::FullTensorPrintOptions options;
        options.max_values = i[1] == -1 ? s::FullTensorPrintOptions::all_values : nonnegative<std::size_t>(i[1]);
        options.precision = nonnegative<int>(i[2]); options.include_metadata = flag(i[3]);
        std::ostringstream output; tensor(i[0])->print(output,options); json.string(output.str()); break;
    }
    case 37: args.sizes(1,0); json.integers(tensor(i[0])->tag()); break;
    case 38: {
        args.sizes(1,0); (void)tensor_slot(i[0]);
        if (library->releaseManagedLibraryExpression(tensor_manager,i[0]))
            throw std::invalid_argument("unable to release managed tensor");
        json.raw("null"); break;
    }
    case 40: {
        if(args.ni != 6 || !args.text1.empty() || !args.text2.empty())
            throw std::invalid_argument("incorrect tensor value arguments");
        const auto shape = args.shape(1);
        std::size_t count = 1;
        for (auto d:shape) { if (d && count > std::numeric_limits<std::size_t>::max()/d) throw std::overflow_error("tensor size overflow"); count *= d; }
        if(args.nr != count) throw std::invalid_argument("tensor values do not match shape");
        produce(json,i[0],[&] {
            s::FullTensor value(shape,s::TensorInit::uninitialized); std::size_t offset=0;
            for(std::size_t a=0;a<shape[0];++a) for(std::size_t b=0;b<shape[1];++b)
            for(std::size_t c=0;c<shape[2];++c) for(std::size_t d=0;d<shape[3];++d)
            for(std::size_t e=0;e<shape[4];++e) value.at(a,b,c,d,e)=r[offset++];
            return value;
        }); break;
    }
    case 50: case 51: case 55: {
        const bool is_full = opcode == 50;
        const bool from_coefficients = opcode == 55;
        auto coefficients = from_coefficients ? args.coefficients(13)
            : std::array<s::CoherentCoefficients, 5>{};
        if (!from_coefficients) args.sizes(is_full?14:13,is_full?1:41);
        const auto target = job(i[0]); std::lock_guard lock(target->control);
        if(target->started) throw std::invalid_argument("job has already started");
        const std::size_t offset = is_full ? 1 : 0;
        const auto Dl = nonnegative<std::size_t>(i[1+offset]);
        const auto backend = enumeration<s::Backend>(i[2+offset],1);
        const auto spins = args.spins<10>(3+offset); const double gamma = r[0];
        const auto normals = opcode == 51 ? args.angles<20>(1) : s::BoundaryAngles{};
        if(is_full) target->destination = std::make_unique<Reservation>(tensor_slot(i[1]));
        try {
            target->worker = std::thread([p=target.get(),is_full,from_coefficients,Dl,backend,spins,gamma,normals,
                                          coefficients=std::move(coefficients)] {
                try {
                    auto progress = [p](const s::ProgressEvent& event) {p->push(event);};
                    if(is_full) p->tensor=std::make_shared<s::FullTensor>(s::compute_full_tensor(spins,Dl,gamma,backend,progress));
                    else if(from_coefficients) p->amplitude=s::compute_coherent_amplitude(spins,Dl,gamma,
                        coefficients[0],coefficients[1],coefficients[2],coefficients[3],coefficients[4],backend,progress);
                    else p->amplitude=s::compute_coherent_amplitude(spins,Dl,gamma,normals,backend,progress);
                } catch (...) { p->failure=std::current_exception(); }
                p->done.store(true,std::memory_order_release);
            });
            target->started=true;
        } catch (...) { target->destination.reset(); throw; }
        json.raw("null"); break;
    }
    case 52: {
        args.sizes(1,0); const auto target=job(i[0]);
        std::lock_guard control(target->control);
        if(!target->started) throw std::invalid_argument("job has not started");
        std::lock_guard lock(target->mutex);
        json.raw("{\"done\":"); json.boolean(target->done.load(std::memory_order_acquire)); json.raw(",\"events\":[");
        for(std::size_t n=0;n<target->size;++n) {if(n) json.raw(","); event_json(json,target->events[(target->begin+n)%Job::capacity]);}
        json.raw("]}"); target->begin=0; target->size=0; target->space.notify_all(); break;
    }
    case 53: {
        args.sizes(1,0); const auto target=job(i[0]); std::lock_guard lock(target->control);
        if(!target->started || target->consumed) throw std::invalid_argument("job has not started or result was already consumed");
        target->join();
        if(target->failure) std::rethrow_exception(target->failure);
        if(target->destination) {
            metadata(json,*target->tensor); target->destination->commit(target->tensor);
            target->tensor.reset(); target->destination.reset();
        } else json.complex(target->amplitude);
        target->consumed=true; break;
    }
    case 54: {
        args.sizes(1,0); auto target=job(i[0]);
        { std::lock_guard lock(target->control); target->join(); target->tensor.reset(); target->destination.reset(); target->failure=nullptr; target->consumed=true; }
        if(library->releaseManagedLibraryExpression(job_manager,i[0])) throw std::invalid_argument("unable to release managed job");
        json.raw("null"); break;
    }
    default: throw std::invalid_argument("unknown dispatch operation");
    }
}

struct InputStrings {
    WolframLibraryData library; char* first; char* second;
    ~InputStrings() { if(first) library->UTF8String_disown(first); if(second) library->UTF8String_disown(second); }
};

} // namespace

extern "C" {
DLLEXPORT mint WolframLibrary_getVersion() noexcept { return WolframLibraryVersion; }
DLLEXPORT int WolframLibrary_initialize(WolframLibraryData library) noexcept
{
    try {
        if (!retain_image_for_thread_destructors()) return LIBRARY_FUNCTION_ERROR;
        if(library->registerLibraryExpressionManager(tensor_manager,manage_tensor)) return LIBRARY_FUNCTION_ERROR;
        if(library->registerLibraryExpressionManager(job_manager,manage_job)) {
            library->unregisterLibraryExpressionManager(tensor_manager); return LIBRARY_FUNCTION_ERROR;
        }
        return LIBRARY_NO_ERROR;
    } catch (...) { return LIBRARY_FUNCTION_ERROR; }
}
DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData library) noexcept
{
    try {
        // These invoke managers; never hold registry_mutex across them.
        library->unregisterLibraryExpressionManager(job_manager);
        library->unregisterLibraryExpressionManager(tensor_manager);
        std::unordered_map<mint,std::shared_ptr<Job>> remaining_jobs;
        std::unordered_map<mint,std::shared_ptr<TensorSlot>> remaining_tensors;
        { std::lock_guard lock(registry_mutex); remaining_jobs.swap(jobs); remaining_tensors.swap(tensors); }
    } catch (...) { }
}
DLLEXPORT int sfx_dispatch(WolframLibraryData library, mint argc, MArgument* args, MArgument result) noexcept
{
    if(argc != 5) return LIBRARY_FUNCTION_ERROR;
    InputStrings strings{library,MArgument_getUTF8String(args[3]),MArgument_getUTF8String(args[4])};
    try {
        Json json;
        try {
            Arguments input(library,args); json.raw("{\"ok\":true,\"value\":");
            dispatch(json,library,MArgument_getInteger(args[0]),input); json.raw("}");
        } catch (...) { json=current_error(); }
        response=std::move(json.text); MArgument_setUTF8String(result,response.data()); return LIBRARY_NO_ERROR;
    } catch (...) {
        static char failure[]="{\"ok\":false,\"error\":{\"kind\":\"Allocation\",\"code\":0,\"message\":\"native error reporting allocation failed\",\"system_code\":0,\"path1\":\"\",\"path2\":\"\"}}";
        MArgument_setUTF8String(result,failure); return LIBRARY_NO_ERROR;
    }
}
DLLEXPORT int sfx_tensor_array(WolframLibraryData library, mint argc, MArgument* args, MArgument result) noexcept
{
    if(argc != 2) return LIBRARY_FUNCTION_ERROR;
    MTensor output=nullptr;
    try {
        const auto owner=tensor(MArgument_getInteger(args[0])); const auto& value=std::as_const(*owner);
        const bool physical=MArgument_getBoolean(args[1]) != 0;
        const auto count=physical ? value.stored_size() : value.logical_size();
        if(!std::in_range<mint>(count)) throw std::overflow_error("tensor array size exceeds mint");
        std::array<mint,5> dimensions{};
        if(physical) dimensions[0]=static_cast<mint>(count);
        else for(std::size_t n=0;n<5;++n) {
            if(!std::in_range<mint>(value.shape()[n])) throw std::overflow_error("tensor extent exceeds mint");
            dimensions[n]=static_cast<mint>(value.shape()[n]);
        }
        const int error=library->MTensor_new(MType_Real,physical?1:5,dimensions.data(),&output);
        if(error) {
            if(output) library->MTensor_free(output);
            return error;
        }
        double* data=library->MTensor_getRealData(output);
        if(physical) { if(count) std::copy_n(value.data(),count,data); }
        else {
            const auto& shape=value.shape(); std::size_t offset=0;
            for(std::size_t a=0;a<shape[0];++a) for(std::size_t b=0;b<shape[1];++b)
            for(std::size_t c=0;c<shape[2];++c) for(std::size_t d=0;d<shape[3];++d)
            for(std::size_t e=0;e<shape[4];++e) data[offset++]=value.at(a,b,c,d,e);
        }
        MArgument_setMTensor(result,output); output=nullptr; return LIBRARY_NO_ERROR;
    } catch(const std::bad_alloc&) {
        if(output) library->MTensor_free(output);
        return LIBRARY_MEMORY_ERROR;
    } catch(...) {
        if(output) library->MTensor_free(output);
        return LIBRARY_FUNCTION_ERROR;
    }
}
} // extern "C"
