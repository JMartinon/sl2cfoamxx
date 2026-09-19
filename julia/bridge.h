// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef SL2CFOAMXX_JULIA_BRIDGE_H
#define SL2CFOAMXX_JULIA_BRIDGE_H

/* Private Julia binding ABI. No Julia or C++ types cross this boundary.
 * Every fallible function returns NULL on success or an owned SJXError.
 * Output handles are NULL on failure; dispose errors with sjx_error_free.
 * All integer enum values match the public C++ API. Booleans use int32_t.
 * All char* fields in OUTPUT records are owned, NUL-terminated UTF-8 copies.
 * Dispose records using their matching dispose function. INPUT records borrow
 * their strings only for the call. No exception may cross any entry point. */
#include <stdint.h>

#if defined(_WIN32)
#define SJX_API __declspec(dllexport)
#else
#define SJX_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
#define SJX_NOEXCEPT noexcept
extern "C" {
#else
#define SJX_NOEXCEPT
#endif

typedef struct SJXError SJXError;
typedef struct SJXTensor SJXTensor;
typedef struct SJXProgressQueue SJXProgressQueue;

enum SJXErrorKind {
    SJX_INVALID_ARGUMENT = 1, SJX_BOUNDS = 2, SJX_OVERFLOW = 3,
    SJX_ALLOCATION = 4, SJX_FILESYSTEM = 5, SJX_LIBRARY = 6,
    SJX_RUNTIME = 7
};
typedef struct { double re, im; } SJXComplex;

typedef struct {
    int32_t backend, progress_bar, simplicity_constraint, verbosity;
    int32_t printing_style, tensor_storage, reuse_existing, check_closure;
    int32_t full_tensor_cpu_final_tile;
    uint32_t booster_rho_tile, coherent_gpu_pipeline_slots;
    uint32_t full_tensor_gpu_producer_waves_per_sm, full_tensor_cpu_p_batch;
    uint64_t cpu_threads, memory_budget_bytes, maximum_payload_size_bytes;
    uint64_t full_tensor_gpu_row_capacity;
    double pruning_ratio;
    char* directory;
    char* output_path;
} SJXConfig;

typedef struct {
    int32_t native_arch_enabled, cuda_backend_built, cuda_fast_math_requested;
    char* version;
    char* compiler_id;
    char* compiler_version;
    char* build_type;
    char* blas_vendor_selector;
    char* blas_provider;
    char* cuda_toolkit_version;
    char* cuda_architectures;
    char* cutensor_version;
} SJXBuildInfo;

typedef struct {
    uint64_t ordinal, total_memory_bytes;
    int32_t compute_capability_major, compute_capability_minor;
    int32_t multiprocessor_count, warp_size, max_threads_per_block;
    char* name;
    char* pci_bus_id;
} SJXDeviceInfo;

typedef struct {
    int32_t operation, phase, backend, status;
    uint32_t attempt;
    int32_t eta_available;
    uint64_t completed, total;
    double elapsed_seconds, eta_seconds;
} SJXProgressEvent;

typedef struct {
    double vector[3];
    double residual_norm, scale, relative_residual;
    int32_t closed;
} SJXClosureNode;
typedef struct {
    SJXClosureNode nodes[5];
    int32_t all_closed;
} SJXClosureResult;

typedef struct {
    uint64_t shape[5], strides[5];
    uint64_t logical_values, stored_values, stored_bytes, dense_bytes;
    uint64_t Dl;
    double gamma;
    int32_t spins[10];
    int32_t has_scientific, simplicity_constraint;
    int32_t storage;
    uint32_t layout;
    /* -1 means no reassociated layout. */
    int32_t reassociated_layout, mapped, writable, symmetry_compressed, dense;
    char* path;
} SJXTensorInfo;

SJX_API uint32_t sjx_abi_version(void) SJX_NOEXCEPT;
SJX_API int32_t sjx_error_kind(const SJXError*) SJX_NOEXCEPT;
SJX_API int32_t sjx_error_code(const SJXError*) SJX_NOEXCEPT;
SJX_API int32_t sjx_error_system_code(const SJXError*) SJX_NOEXCEPT;
/* Error strings remain valid until error_free, including filesystem paths. */
SJX_API const char* sjx_error_message(const SJXError*) SJX_NOEXCEPT;
SJX_API const char* sjx_error_path1(const SJXError*) SJX_NOEXCEPT;
SJX_API const char* sjx_error_path2(const SJXError*) SJX_NOEXCEPT;
SJX_API void sjx_error_free(SJXError*) SJX_NOEXCEPT;
SJX_API void sjx_free(void*) SJX_NOEXCEPT;

SJX_API SJXError* sjx_default_config(SJXConfig*) SJX_NOEXCEPT;
SJX_API SJXError* sjx_current_config(SJXConfig*) SJX_NOEXCEPT;
SJX_API SJXError* sjx_configure(const SJXConfig*) SJX_NOEXCEPT;
SJX_API void sjx_config_dispose(SJXConfig*) SJX_NOEXCEPT;
SJX_API SJXError* sjx_build_info(SJXBuildInfo*) SJX_NOEXCEPT;
SJX_API void sjx_build_info_dispose(SJXBuildInfo*) SJX_NOEXCEPT;
SJX_API SJXError* sjx_backend_available(int32_t backend, int32_t*) SJX_NOEXCEPT;
SJX_API SJXError* sjx_cuda_device_count(uint64_t*) SJX_NOEXCEPT;
SJX_API SJXError* sjx_cuda_device_info(uint64_t ordinal, SJXDeviceInfo*) SJX_NOEXCEPT;
SJX_API void sjx_device_info_dispose(SJXDeviceInfo*) SJX_NOEXCEPT;

/* Queue callbacks contain only native data. Never pass a Julia function to
 * native workers. Keep the queue alive until the threaded computation has
 * returned and drain it before free. The queue holds at most 256 events.
 * Running updates are coalesced/evicted; lifecycle events retain their order.
 * Lifecycle-only saturation waits for space. Poll is nonblocking and copies
 * up to capacity.
 * A queue belongs to one computation; callbacks may be delivered concurrently
 * with poll. Callback errors in Julia must not abandon draining the call. */
SJX_API SJXError* sjx_progress_create(SJXProgressQueue**) SJX_NOEXCEPT;
SJX_API SJXError* sjx_progress_poll(SJXProgressQueue*, SJXProgressEvent*,
                                   uint64_t capacity, uint64_t* count) SJX_NOEXCEPT;
SJX_API void sjx_progress_free(SJXProgressQueue*) SJX_NOEXCEPT;

/* Arrays have exact fixed lengths: spins10, intertwiners5, tetrahedron spins4.
 * Angle arrays are INTERLEAVED theta,phi (40 or 8 doubles), not Julia's matrix
 * storage order. Inputs and output locations must remain alive until return.
 * backend==NULL selects the configured default; otherwise *backend is exact.
 * An explicit backend is required when Queue is non-NULL, as in Python.
 * Queue==NULL preserves ordinary native callback/default semantics. */
SJX_API SJXError* sjx_indices_from_intertwiners(const int32_t* spins10,
    const int32_t* intertwiners5, uint64_t* indices5) SJX_NOEXCEPT;
SJX_API SJXError* sjx_compute_amplitude(const int32_t* spins10,
    const int32_t* intertwiners5, uint64_t Dl, double gamma,
    int32_t cache_policy, double* result) SJX_NOEXCEPT;
SJX_API SJXError* sjx_compute_coherent_amplitude(const int32_t* spins10,
    uint64_t Dl, double gamma, const double* angles40,
    const int32_t* backend, SJXProgressQueue*, SJXComplex* result) SJX_NOEXCEPT;
/* Each coefficient vector is in ascending admissible intertwiner order for
 * its node. Counts must match the corresponding tensor dimensions. Values
 * are used without conjugation or normalization; the C++ API validates them. */
SJX_API SJXError* sjx_compute_coherent_amplitude_coefficients(const int32_t* spins10,
    uint64_t Dl, double gamma,
    const SJXComplex* C1, uint64_t count1, const SJXComplex* C2, uint64_t count2,
    const SJXComplex* C3, uint64_t count3, const SJXComplex* C4, uint64_t count4,
    const SJXComplex* C5, uint64_t count5,
    const int32_t* backend, SJXProgressQueue*, SJXComplex* result) SJX_NOEXCEPT;
SJX_API SJXError* sjx_compute_full_tensor(const int32_t* spins10,
    uint64_t Dl, double gamma, const int32_t* backend,
    SJXProgressQueue*, SJXTensor** result) SJX_NOEXCEPT;
/* coefficients are an owned array; free with sjx_free. */
SJX_API SJXError* sjx_compute_coherent_coefficients(const int32_t* spins4,
    const double* angles8, SJXComplex** coefficients, uint64_t* count) SJX_NOEXCEPT;
SJX_API SJXError* sjx_check_closure(const int32_t* spins10,
    const double* angles40, double absolute_tolerance, double relative_tolerance,
    SJXClosureResult*) SJX_NOEXCEPT;

SJX_API SJXError* sjx_tensor_create(const uint64_t* shape5, int32_t init,
    uint32_t layout, SJXTensor**) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_create_mapped(const char* path,
    const uint64_t* shape5, int32_t init, uint32_t layout, SJXTensor**) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_load(const char* path, SJXTensor**) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_map(const char* path, int32_t access,
    SJXTensor**) SJX_NOEXCEPT;
SJX_API void sjx_tensor_free(SJXTensor*) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_info(const SJXTensor*, SJXTensorInfo*) SJX_NOEXCEPT;
SJX_API void sjx_tensor_info_dispose(SJXTensorInfo*) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_file_info(const char* path, SJXTensorInfo*) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_get(const SJXTensor*, const uint64_t* indices5,
    double*) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_set(SJXTensor*, const uint64_t* indices5,
    double value) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_get_flat(const SJXTensor*, uint64_t index,
    double*) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_set_flat(SJXTensor*, uint64_t index,
    double value) SJX_NOEXCEPT;
/* Payload pointer is BORROWED until tensor_free. Caller must pin the handle
 * and enforce writable==0 on every write, including array views. Pointer
 * addresses physical storage, which may be compressed/reassociated. */
SJX_API SJXError* sjx_tensor_payload(const SJXTensor*, const double** data,
    uint64_t* count, int32_t* writable) SJX_NOEXCEPT;
/* canonical!=0 expands canonical column-major values; otherwise physical.
 * Destination count must equal the corresponding complete tensor size. */
SJX_API SJXError* sjx_tensor_copy_values(const SJXTensor*, int32_t canonical,
    double* destination, uint64_t count) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_clone(const SJXTensor*, SJXTensor**) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_dense_copy(const SJXTensor*, SJXTensor**) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_fill(SJXTensor*, double value) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_save(const SJXTensor*, const char* path) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_flush(const SJXTensor*) SJX_NOEXCEPT;
SJX_API SJXError* sjx_tensor_convert_storage(const SJXTensor*, int32_t storage,
    const char* destination, int32_t overwrite, SJXTensor**) SJX_NOEXCEPT;
/* format result is owned; free with sjx_free. UINT64_MAX means all_values. */
SJX_API SJXError* sjx_tensor_format(const SJXTensor*, uint64_t max_values,
    int32_t precision, int32_t include_metadata, char** result) SJX_NOEXCEPT;
/* tag destination length must be exactly 1024. No tag/layout mutation is
 * exposed: metadata and existing zero-copy views must remain valid. */
SJX_API SJXError* sjx_tensor_tag(const SJXTensor*, uint8_t* destination,
    uint64_t count) SJX_NOEXCEPT;

#ifdef __cplusplus
}
#endif
#endif
