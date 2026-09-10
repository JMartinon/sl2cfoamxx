// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/error.hpp"
#include "sl2cfoamxx/export.hpp"
#include "sl2cfoamxx/types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sl2cfoamxx {

inline constexpr std::size_t tensor_max_rank = 128;
inline constexpr std::size_t tensor_tag_bytes = 1024;
inline constexpr std::size_t tensor_memory_alignment = 64;

enum class TensorInit {
    zero,
    uninitialized,

    // Compatibility spellings for the former internal class.
    Zero = zero,
    Uninitialized = uninitialized
};

enum class TensorZeroPolicy {
    automatic,
    serial,
    parallel,

    // Compatibility spellings for the former internal tensor implementation.
    Auto = automatic,
    Serial = serial,
    Parallel = parallel
};

enum class TensorAccess {
    read_only,
    read_write
};

// Tensor payloads are always column-major: the first index is contiguous.
enum class TensorLayout : std::uint32_t {
    generic_column_major = 0,
    canonical_dense_column_major = 1,
    // Rank-one by-i1 full-tensor payload. The full-tensor metadata tag records
    // whether its panels are reflection-triangular or rectangular.
    full_tensor_reassociated_by_i1 = 2,
    full_tensor_reassociated_reflection_compact_by_i1 =
        full_tensor_reassociated_by_i1,

    // Compatibility spellings for the former internal file format helpers.
    GenericColumnMajor = generic_column_major,
    CanonicalDenseColumnMajor = canonical_dense_column_major,
    FullTensorReassociatedByI1 = full_tensor_reassociated_by_i1,
    FullTensorReassociatedReflectionCompactByI1 =
        full_tensor_reassociated_reflection_compact_by_i1,

    // Legacy source spellings; the numeric on-disk layout remains unchanged.
    full_tensor_reassociated_by_a1 = full_tensor_reassociated_by_i1,
    full_tensor_reassociated_reflection_compact_by_a1 =
        full_tensor_reassociated_reflection_compact_by_i1,
    FullTensorReassociatedByA1 = FullTensorReassociatedByI1,
    FullTensorReassociatedReflectionCompactByA1 =
        FullTensorReassociatedReflectionCompactByI1
};

namespace tensor_disk {

inline constexpr std::uint32_t version = 1;
inline constexpr std::size_t magic_bytes = 16;
inline constexpr std::array<char, magic_bytes> magic{
    'S','L','2','C','T','E','N','S','O','R','B','I','N','\0','\0','\0'};

enum class ScalarCode : std::uint32_t {
    Unknown = 0,
    Float64 = 1,
    Float32 = 2,
    Int64 = 3,
    UInt64 = 4,
    Int32 = 5,
    UInt32 = 6,
    Int8 = 7,
    UInt8 = 8
};

template <typename T>
struct ScalarCodeOf {
    static constexpr ScalarCode value = ScalarCode::Unknown;
};

template <> struct ScalarCodeOf<double>        { static constexpr ScalarCode value = ScalarCode::Float64; };
template <> struct ScalarCodeOf<float>         { static constexpr ScalarCode value = ScalarCode::Float32; };
template <> struct ScalarCodeOf<std::int64_t>  { static constexpr ScalarCode value = ScalarCode::Int64; };
template <> struct ScalarCodeOf<std::uint64_t> { static constexpr ScalarCode value = ScalarCode::UInt64; };
template <> struct ScalarCodeOf<std::int32_t>  { static constexpr ScalarCode value = ScalarCode::Int32; };
template <> struct ScalarCodeOf<std::uint32_t> { static constexpr ScalarCode value = ScalarCode::UInt32; };
template <> struct ScalarCodeOf<std::int8_t>   { static constexpr ScalarCode value = ScalarCode::Int8; };
template <> struct ScalarCodeOf<std::uint8_t>  { static constexpr ScalarCode value = ScalarCode::UInt8; };

using LayoutCode = TensorLayout;

// V1 is retained byte-for-byte so existing .sl2cxx caches remain readable.
struct TensorTreeHeaderV1 {
    char magic[magic_bytes]{};
    std::uint32_t version = tensor_disk::version;
    std::uint32_t ndim = 0;
    std::uint32_t scalar_code = 0;
    std::uint32_t layout_code = 0;

    std::uint64_t sizeof_scalar = 0;
    std::uint64_t values_count = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t tag_bytes = tensor_tag_bytes;

    std::uint64_t dims[tensor_max_rank]{};
    std::uint64_t reserved_u64[32]{};
    std::uint8_t tag[tensor_tag_bytes]{};
};

static_assert(std::is_trivially_copyable_v<TensorTreeHeaderV1>);

[[nodiscard]] SL2CFOAMXX_API TensorTreeHeaderV1 read_header_file(
    const std::filesystem::path& file);

// Old V1 files used 4096 when the offset field was absent. New writers always
// store their actual offset, so this function is only a compatibility fallback.
[[nodiscard]] constexpr std::uint64_t
default_single_file_payload_offset() noexcept
{
    return 4096;
}

} // namespace tensor_disk

namespace detail {

class SL2CFOAMXX_API MappedRegion {
public:
    MappedRegion() = default;
    MappedRegion(const MappedRegion&) = delete;
    MappedRegion& operator=(const MappedRegion&) = delete;
    ~MappedRegion();

    [[nodiscard]] static std::unique_ptr<MappedRegion> create(
        const std::filesystem::path& file,
        std::size_t bytes);

    [[nodiscard]] static std::unique_ptr<MappedRegion> open(
        const std::filesystem::path& file,
        TensorAccess access);

    void flush() const;

    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool writable() const noexcept { return writable_; }

private:
    std::filesystem::path path_{};
    void* data_ = nullptr;
    std::size_t size_ = 0;
    int descriptor_ = -1;
    bool writable_ = false;
};

SL2CFOAMXX_API void write_tensor_file_atomic(
    const std::filesystem::path& file,
    const tensor_disk::TensorTreeHeaderV1& header,
    std::uint64_t payload_offset,
    const void* payload,
    std::size_t payload_bytes);

SL2CFOAMXX_API void read_tensor_file_payload(
    const std::filesystem::path& file,
    std::uint64_t payload_offset,
    void* payload,
    std::size_t payload_bytes);

[[nodiscard]] SL2CFOAMXX_API std::uint64_t checked_file_size(
    const std::filesystem::path& file);

template <typename T>
struct AlignedDelete {
    void operator()(T* pointer) const noexcept
    {
        if (pointer != nullptr) {
            ::operator delete[](
                pointer,
                std::align_val_t{std::max(tensor_memory_alignment, alignof(T))});
        }
    }
};

template <typename T>
using AlignedArray = std::unique_ptr<T[], AlignedDelete<T>>;

template <typename T>
[[nodiscard]] AlignedArray<T> allocate_aligned(std::size_t count)
{
    if (count == 0) return {};
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("tensor allocation size overflows size_t");
    }
    return AlignedArray<T>(static_cast<T*>(::operator new[](
        count * sizeof(T),
        std::align_val_t{std::max(tensor_memory_alignment, alignof(T))})));
}

template <std::size_t Rank>
[[nodiscard]] std::size_t checked_element_count(
    const std::array<std::size_t, Rank>& shape)
{
    std::size_t count = 1;
    for (const std::size_t extent : shape) {
        if (extent != 0 &&
            count > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::overflow_error("tensor shape product overflows size_t");
        }
        count *= extent;
    }
    return count;
}

template <typename T, std::size_t Rank>
[[nodiscard]] tensor_disk::TensorTreeHeaderV1 make_header(
    const std::array<std::size_t, Rank>& shape,
    TensorLayout layout,
    const std::array<std::uint8_t, tensor_tag_bytes>& tag,
    std::uint64_t payload_offset)
{
    static_assert(tensor_disk::ScalarCodeOf<T>::value !=
                      tensor_disk::ScalarCode::Unknown,
                  "this scalar type has no .sl2cxx file code");

    const std::size_t count = checked_element_count(shape);
    if (count > std::numeric_limits<std::uint64_t>::max() / sizeof(T)) {
        throw std::overflow_error("tensor payload size overflows uint64_t");
    }

    tensor_disk::TensorTreeHeaderV1 header{};
    std::copy(tensor_disk::magic.begin(), tensor_disk::magic.end(),
              header.magic);
    header.version = tensor_disk::version;
    header.ndim = static_cast<std::uint32_t>(Rank);
    header.scalar_code = static_cast<std::uint32_t>(
        tensor_disk::ScalarCodeOf<T>::value);
    header.layout_code = static_cast<std::uint32_t>(layout);
    header.sizeof_scalar = sizeof(T);
    header.values_count = static_cast<std::uint64_t>(count);
    header.payload_bytes = static_cast<std::uint64_t>(count) * sizeof(T);
    header.tag_bytes = tensor_tag_bytes;
    header.reserved_u64[0] = payload_offset;
    for (std::size_t i = 0; i < Rank; ++i) {
        header.dims[i] = static_cast<std::uint64_t>(shape[i]);
    }
    std::copy(tag.begin(), tag.end(), header.tag);
    return header;
}

template <typename T, std::size_t Rank>
[[nodiscard]] std::array<std::size_t, Rank> validate_header(
    const tensor_disk::TensorTreeHeaderV1& header,
    std::uint64_t file_bytes)
{
    static_assert(tensor_disk::ScalarCodeOf<T>::value !=
                      tensor_disk::ScalarCode::Unknown,
                  "this scalar type has no .sl2cxx file code");
    if (!std::equal(tensor_disk::magic.begin(), tensor_disk::magic.end(),
                    header.magic)) {
        throw Error(ErrorCode::invalid_tensor_file,
                    "invalid tensor file magic");
    }
    if (header.version != tensor_disk::version) {
        throw Error(ErrorCode::invalid_tensor_file,
                    "unsupported tensor file version");
    }
    if (header.ndim != Rank) {
        throw Error(ErrorCode::invalid_tensor_file,
                    "tensor rank does not match requested type");
    }
    if (header.scalar_code != static_cast<std::uint32_t>(
                                  tensor_disk::ScalarCodeOf<T>::value) ||
        header.sizeof_scalar != sizeof(T)) {
        throw Error(ErrorCode::invalid_tensor_file,
                    "tensor scalar type does not match requested type");
    }
    if (header.tag_bytes != tensor_tag_bytes) {
        throw Error(ErrorCode::invalid_tensor_file,
                    "unsupported tensor tag size");
    }
    if (header.layout_code > static_cast<std::uint32_t>(
                                 TensorLayout::FullTensorReassociatedReflectionCompactByI1)) {
        throw Error(ErrorCode::invalid_tensor_file,
                    "unsupported tensor layout code");
    }

    std::array<std::size_t, Rank> shape{};
    for (std::size_t i = 0; i < Rank; ++i) {
        if (header.dims[i] > std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("tensor extent does not fit size_t");
        }
        shape[i] = static_cast<std::size_t>(header.dims[i]);
    }

    const std::size_t count = checked_element_count(shape);
    if (header.values_count != static_cast<std::uint64_t>(count)) {
        throw Error(ErrorCode::invalid_tensor_file,
                    "tensor header contains an inconsistent value count");
    }
    if (count > std::numeric_limits<std::uint64_t>::max() / sizeof(T) ||
        header.payload_bytes != static_cast<std::uint64_t>(count) * sizeof(T)) {
        throw Error(ErrorCode::invalid_tensor_file,
                    "tensor header contains an inconsistent payload size");
    }

    const std::uint64_t offset = header.reserved_u64[0] != 0
        ? header.reserved_u64[0]
        : tensor_disk::default_single_file_payload_offset();
    if (offset < sizeof(tensor_disk::TensorTreeHeaderV1) ||
        offset > std::numeric_limits<std::uint64_t>::max() -
                     header.payload_bytes ||
        file_bytes < offset + header.payload_bytes) {
        throw Error(
            ErrorCode::invalid_tensor_file,
            "tensor file is truncated or has an invalid payload offset");
    }
    return shape;
}

template <typename Value>
[[nodiscard]] std::size_t checked_index_value(Value value)
{
    static_assert(std::is_integral_v<Value>, "tensor indices must be integral");
    if constexpr (std::is_signed_v<Value>) {
        if (value < 0) throw std::out_of_range("negative tensor index");
    }
    if constexpr (sizeof(Value) > sizeof(std::size_t)) {
        using Unsigned = std::make_unsigned_t<Value>;
        if (static_cast<Unsigned>(value) >
            std::numeric_limits<std::size_t>::max()) {
            throw std::out_of_range("tensor index does not fit size_t");
        }
    }
    return static_cast<std::size_t>(value);
}

} // namespace detail

template <typename T, std::size_t Rank>
class Tensor {
    static_assert(Rank > 0 && Rank <= tensor_max_rank,
                  "invalid tensor rank");
    static_assert(std::is_trivially_copyable_v<T>,
                  "Tensor requires a trivially copyable scalar type");

public:
    using value_type = T;
    using size_type = std::size_t;
    using shape_type = std::array<size_type, Rank>;
    static constexpr std::size_t rank = Rank;

    Tensor() = default;

    explicit Tensor(
        const shape_type& shape,
        TensorInit init = TensorInit::zero,
        TensorLayout layout = TensorLayout::GenericColumnMajor,
        TensorZeroPolicy zero_policy = TensorZeroPolicy::automatic)
        : shape_(shape), layout_(layout)
    {
        initialize_shape();
        allocate_owned(init, zero_policy);
    }

    explicit Tensor(
        const shape_type& shape,
        TensorInit init,
        TensorZeroPolicy zero_policy)
        : Tensor(shape, init, TensorLayout::GenericColumnMajor, zero_policy)
    {
    }

    // Source compatibility for existing internal construction sites.
    explicit Tensor(const shape_type& shape, bool zero_initialize)
        : Tensor(shape,
                 zero_initialize ? TensorInit::zero
                                 : TensorInit::uninitialized)
    {
    }

    Tensor(const Tensor& other)
        : Tensor(other.shape_, TensorInit::uninitialized, other.layout_)
    {
        tag_ = other.tag_;
        copy_payload_from(other.data_);
    }

    Tensor& operator=(const Tensor& other)
    {
        if (this != &other) {
            Tensor copy(other);
            swap(copy);
        }
        return *this;
    }

    Tensor(Tensor&& other) noexcept { move_from(std::move(other)); }

    Tensor& operator=(Tensor&& other) noexcept
    {
        if (this != &other) {
            close_storage();
            move_from(std::move(other));
        }
        return *this;
    }

    ~Tensor() = default;

    [[nodiscard]] static Tensor create_mapped(
        const std::filesystem::path& file,
        const shape_type& shape,
        TensorInit init = TensorInit::zero,
        TensorLayout layout = TensorLayout::GenericColumnMajor,
        const std::array<std::uint8_t, tensor_tag_bytes>& tag = {})
    {
        Tensor output;
        output.shape_ = shape;
        output.layout_ = layout;
        output.tag_ = tag;
        output.initialize_shape();

        constexpr std::uint64_t payload_offset = 65536;
        const auto header = detail::make_header<T, Rank>(
            shape, layout, tag, payload_offset);
        if (header.payload_bytes > std::numeric_limits<std::size_t>::max() ||
            payload_offset > std::numeric_limits<std::size_t>::max() -
                                 static_cast<std::size_t>(header.payload_bytes)) {
            throw std::overflow_error("mapped tensor file size overflows size_t");
        }

        const std::size_t total_bytes = static_cast<std::size_t>(payload_offset) +
                                        static_cast<std::size_t>(header.payload_bytes);
        output.mapping_ = detail::MappedRegion::create(file, total_bytes);
        output.path_ = file;
        output.read_only_ = false;
        output.payload_offset_ = payload_offset;
        std::memcpy(output.mapping_->data(), &header, sizeof(header));
        output.data_ = output.size_ == 0
            ? nullptr
            : reinterpret_cast<T*>(
                  static_cast<std::byte*>(output.mapping_->data()) +
                  payload_offset);
        if (init == TensorInit::zero && output.size_ != 0) output.zero();
        return output;
    }

    [[nodiscard]] static Tensor map(
        const std::filesystem::path& file,
        TensorAccess access = TensorAccess::read_only)
    {
        const auto header = tensor_disk::read_header_file(file);
        const std::uint64_t file_bytes = detail::checked_file_size(file);
        const shape_type shape = detail::validate_header<T, Rank>(
            header, file_bytes);
        const std::uint64_t payload_offset = header.reserved_u64[0] != 0
            ? header.reserved_u64[0]
            : tensor_disk::default_single_file_payload_offset();
        if (payload_offset > std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("tensor payload offset does not fit size_t");
        }

        Tensor output;
        output.shape_ = shape;
        output.layout_ = static_cast<TensorLayout>(header.layout_code);
        std::copy(std::begin(header.tag), std::end(header.tag),
                  output.tag_.begin());
        output.initialize_shape();
        output.mapping_ = detail::MappedRegion::open(file, access);
        output.path_ = file;
        output.read_only_ = access == TensorAccess::read_only;
        output.payload_offset_ = payload_offset;
        output.data_ = output.size_ == 0
            ? nullptr
            : reinterpret_cast<T*>(
                  static_cast<std::byte*>(output.mapping_->data()) +
                  static_cast<std::size_t>(payload_offset));
        return output;
    }

    // Compatibility names used by the internal full-tensor planner.
    [[nodiscard]] static Tensor create_file(
        const std::filesystem::path& file,
        const shape_type& shape,
        TensorInit init = TensorInit::zero,
        TensorLayout layout = TensorLayout::GenericColumnMajor,
        const std::array<std::uint8_t, tensor_tag_bytes>& tag = {})
    {
        return create_mapped(file, shape, init, layout, tag);
    }

    [[nodiscard]] static Tensor open_file_existing(
        const std::filesystem::path& file,
        bool read_only = false)
    {
        return map(file, read_only ? TensorAccess::read_only
                                   : TensorAccess::read_write);
    }

    void save(const std::filesystem::path& file) const
    {
        if (is_mapped() && file == path_) {
            flush();
            return;
        }
        constexpr std::uint64_t payload_offset = 65536;
        const auto header = detail::make_header<T, Rank>(
            shape_, layout_, tag_, payload_offset);
        detail::write_tensor_file_atomic(
            file, header, payload_offset, data_, size_bytes());
    }

    [[nodiscard]] static Tensor load(const std::filesystem::path& file)
    {
        const auto header = tensor_disk::read_header_file(file);
        const std::uint64_t file_bytes = detail::checked_file_size(file);
        const shape_type shape = detail::validate_header<T, Rank>(
            header, file_bytes);
        const std::uint64_t payload_offset = header.reserved_u64[0] != 0
            ? header.reserved_u64[0]
            : tensor_disk::default_single_file_payload_offset();

        Tensor output(shape, TensorInit::uninitialized,
                      static_cast<TensorLayout>(header.layout_code));
        std::copy(std::begin(header.tag), std::end(header.tag),
                  output.tag_.begin());
        detail::read_tensor_file_payload(
            file, payload_offset, output.data_, output.size_bytes());
        return output;
    }

    void flush() const
    {
        if (mapping_) mapping_->flush();
    }

    [[nodiscard]] T* data()
    {
        ensure_writable();
        return data_;
    }
    [[nodiscard]] const T* data() const noexcept { return data_; }

    [[nodiscard]] std::span<T> span()
    {
        ensure_writable();
        return {data_, size_};
    }
    [[nodiscard]] std::span<const T> span() const noexcept
    {
        return {data_, size_};
    }

    [[nodiscard]] T* begin()
    {
        ensure_writable();
        return data_;
    }
    [[nodiscard]] T* end()
    {
        ensure_writable();
        return size_ == 0 ? data_ : data_ + size_;
    }
    [[nodiscard]] const T* begin() const noexcept { return data_; }
    [[nodiscard]] const T* end() const noexcept
    {
        return size_ == 0 ? data_ : data_ + size_;
    }
    [[nodiscard]] const T* cbegin() const noexcept { return data_; }
    [[nodiscard]] const T* cend() const noexcept
    {
        return size_ == 0 ? data_ : data_ + size_;
    }

    [[nodiscard]] T& operator[](size_type index)
    {
        ensure_writable();
        return data_[index];
    }
    [[nodiscard]] const T& operator[](size_type index) const noexcept
    {
        return data_[index];
    }

    [[nodiscard]] T& at_flat(size_type index)
    {
        if (index >= size_) throw std::out_of_range("flat tensor index is out of range");
        return (*this)[index];
    }
    [[nodiscard]] const T& at_flat(size_type index) const
    {
        if (index >= size_) throw std::out_of_range("flat tensor index is out of range");
        return data_[index];
    }

    [[nodiscard]] T& operator()(const shape_type& indices)
    {
        ensure_writable();
        return data_[compute_index(indices)];
    }
    [[nodiscard]] const T& operator()(const shape_type& indices) const noexcept
    {
        return data_[compute_index(indices)];
    }

    template <typename... Indices>
    [[nodiscard]] T& operator()(Indices... indices)
    {
        static_assert(sizeof...(Indices) == Rank,
                      "incorrect number of tensor indices");
        return (*this)(shape_type{static_cast<size_type>(indices)...});
    }

    template <typename... Indices>
    [[nodiscard]] const T& operator()(Indices... indices) const
    {
        static_assert(sizeof...(Indices) == Rank,
                      "incorrect number of tensor indices");
        return (*this)(shape_type{static_cast<size_type>(indices)...});
    }

    [[nodiscard]] T& at(const shape_type& indices)
    {
        check_indices(indices);
        return (*this)(indices);
    }
    [[nodiscard]] const T& at(const shape_type& indices) const
    {
        check_indices(indices);
        return (*this)(indices);
    }

    template <typename... Indices>
    [[nodiscard]] T& at(Indices... indices)
    {
        static_assert(sizeof...(Indices) == Rank,
                      "incorrect number of tensor indices");
        return at(shape_type{detail::checked_index_value(indices)...});
    }

    template <typename... Indices>
    [[nodiscard]] const T& at(Indices... indices) const
    {
        static_assert(sizeof...(Indices) == Rank,
                      "incorrect number of tensor indices");
        return at(shape_type{detail::checked_index_value(indices)...});
    }

    void fill(const T& value)
    {
        ensure_writable();
        std::fill_n(data_, size_, value);
    }

    void zero(TensorZeroPolicy policy = TensorZeroPolicy::automatic)
    {
        ensure_writable();
        if (size_ == 0) return;

#ifdef _OPENMP
        const bool parallel_requested = policy != TensorZeroPolicy::serial;
        const bool large_enough =
            policy == TensorZeroPolicy::parallel ||
            size_bytes() >= parallel_zero_threshold_bytes();

        if (parallel_requested && large_enough && !omp_in_parallel()) {
            const int maximum_threads = omp_get_max_threads();
            if (maximum_threads > 1) {
                int thread_count = maximum_threads;
                if (policy == TensorZeroPolicy::automatic) {
                    const size_type wanted = std::max<size_type>(
                        1,
                        1 + (size_bytes() - 1) /
                                parallel_zero_min_bytes_per_thread());
                    thread_count = static_cast<int>(std::min<size_type>(
                        static_cast<size_type>(maximum_threads), wanted));
                }

                if (thread_count > 1) {
#pragma omp parallel num_threads(thread_count)
                    {
                        const size_type thread =
                            static_cast<size_type>(omp_get_thread_num());
                        const size_type threads =
                            static_cast<size_type>(omp_get_num_threads());
                        const size_type begin = balanced_partition(
                            size_, thread, threads);
                        const size_type end = balanced_partition(
                            size_, thread + 1, threads);
                        std::fill_n(data_ + begin, end - begin, T{});
                    }
                    return;
                }
            }
        }
#else
        (void)policy;
#endif

        std::fill_n(data_, size_, T{});
    }

    void zero_serial() { zero(TensorZeroPolicy::serial); }
    void zero_parallel() { zero(TensorZeroPolicy::parallel); }

    [[nodiscard]] static constexpr size_type
    parallel_zero_threshold_bytes() noexcept
    {
        return 32ULL * 1024ULL * 1024ULL;
    }

    [[nodiscard]] static constexpr size_type
    parallel_zero_min_bytes_per_thread() noexcept
    {
        return 8ULL * 1024ULL * 1024ULL;
    }

    void resize(const shape_type& new_shape)
    {
        if (shape_ == new_shape) return;
        Tensor replacement(new_shape, TensorInit::zero, layout_);
        replacement.tag_ = tag_;

        shape_type intersection{};
        std::size_t intersection_size = 1;
        for (std::size_t axis = 0; axis < Rank; ++axis) {
            intersection[axis] = std::min(shape_[axis], new_shape[axis]);
            intersection_size *= intersection[axis];
        }
        shape_type index{};
        for (std::size_t n = 0; n < intersection_size; ++n) {
            replacement(index) = (*this)(index);
            for (std::size_t axis = 0; axis < Rank; ++axis) {
                if (++index[axis] < intersection[axis]) break;
                index[axis] = 0;
            }
        }
        swap(replacement);
    }

    void reshape(const shape_type& new_shape)
    {
        ensure_writable();
        if (detail::checked_element_count(new_shape) != size_) {
            throw std::invalid_argument("reshape changes the tensor element count");
        }
        shape_ = new_shape;
        initialize_strides();
        sync_header_to_mapping();
    }

    void clear() noexcept
    {
        close_storage();
        reset_metadata();
    }

    [[nodiscard]] Tensor clone() const { return Tensor(*this); }

    [[nodiscard]] const shape_type& shape() const noexcept { return shape_; }
    [[nodiscard]] const shape_type& dims() const noexcept { return shape_; }
    [[nodiscard]] const shape_type& strides() const noexcept { return strides_; }
    [[nodiscard]] size_type extent(std::size_t axis) const
    {
        if (axis >= Rank) throw std::out_of_range("tensor axis is out of range");
        return shape_[axis];
    }
    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] size_type size_bytes() const noexcept { return size_ * sizeof(T); }
    [[nodiscard]] std::uint64_t bytes() const noexcept
    {
        return static_cast<std::uint64_t>(size_bytes());
    }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] TensorLayout layout() const noexcept { return layout_; }
    [[nodiscard]] TensorLayout layout_code() const noexcept { return layout_; }
    [[nodiscard]] TensorStorage storage() const noexcept
    {
        return mapping_ ? TensorStorage::on_disk_dense
                        : TensorStorage::in_memory_dense;
    }
    [[nodiscard]] bool is_mapped() const noexcept
    {
        return static_cast<bool>(mapping_);
    }
    [[nodiscard]] bool writable() const noexcept { return !read_only_; }
    [[nodiscard]] bool is_read_only() const noexcept { return read_only_; }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] const std::filesystem::path& root_path() const noexcept { return path_; }

    void set_layout(TensorLayout layout)
    {
        ensure_writable();
        layout_ = layout;
        sync_header_to_mapping();
    }

    void set_tag(std::span<const std::uint8_t> tag)
    {
        ensure_writable();
        if (tag.size() > tensor_tag_bytes) {
            throw std::length_error("tensor tag exceeds 1024 bytes");
        }
        tag_.fill(0);
        std::copy(tag.begin(), tag.end(), tag_.begin());
        sync_header_to_mapping();
    }

    void set_tag(const std::vector<std::uint8_t>& tag) { set_tag(std::span(tag)); }

    void set_tag(const std::array<std::uint8_t, tensor_tag_bytes>& tag)
    {
        set_tag(std::span<const std::uint8_t>(tag));
    }

    [[nodiscard]] const std::array<std::uint8_t, tensor_tag_bytes>& tag() const noexcept
    {
        return tag_;
    }
    [[nodiscard]] const std::array<std::uint8_t, tensor_tag_bytes>& get_tag() const noexcept
    {
        return tag_;
    }

    [[nodiscard]] size_type compute_index(const shape_type& indices) const noexcept
    {
        size_type offset = 0;
        for (std::size_t axis = 0; axis < Rank; ++axis) {
            offset += strides_[axis] * indices[axis];
        }
        return offset;
    }

    void swap(Tensor& other) noexcept
    {
        using std::swap;
        swap(shape_, other.shape_);
        swap(strides_, other.strides_);
        swap(size_, other.size_);
        swap(owned_, other.owned_);
        swap(mapping_, other.mapping_);
        swap(data_, other.data_);
        swap(path_, other.path_);
        swap(read_only_, other.read_only_);
        swap(payload_offset_, other.payload_offset_);
        swap(layout_, other.layout_);
        swap(tag_, other.tag_);
    }

private:
    [[nodiscard]] static size_type balanced_partition(
        size_type size,
        size_type part,
        size_type parts) noexcept
    {
        return (size / parts) * part +
               std::min(size % parts, part);
    }

    void copy_payload_from(const T* source) noexcept
    {
        if (size_ == 0) return;

#ifdef _OPENMP
        if (size_bytes() >= parallel_zero_threshold_bytes() &&
            !omp_in_parallel()) {
            const int maximum_threads = omp_get_max_threads();
            if (maximum_threads > 1) {
                const size_type wanted = std::max<size_type>(
                    1,
                    1 + (size_bytes() - 1) /
                            parallel_zero_min_bytes_per_thread());
                const int thread_count = static_cast<int>(
                    std::min<size_type>(
                        static_cast<size_type>(maximum_threads), wanted));
                if (thread_count > 1) {
#pragma omp parallel num_threads(thread_count)
                    {
                        const size_type thread = static_cast<size_type>(
                            omp_get_thread_num());
                        const size_type threads = static_cast<size_type>(
                            omp_get_num_threads());
                        const size_type begin = balanced_partition(
                            size_, thread, threads);
                        const size_type end = balanced_partition(
                            size_, thread + 1, threads);
                        std::memcpy(
                            data_ + begin,
                            source + begin,
                            (end - begin) * sizeof(T));
                    }
                    return;
                }
            }
        }
#endif

        std::memcpy(data_, source, size_bytes());
    }

    void initialize_shape()
    {
        size_ = detail::checked_element_count(shape_);
        initialize_strides();
    }

    void initialize_strides() noexcept
    {
        size_type stride = 1;
        for (std::size_t axis = 0; axis < Rank; ++axis) {
            strides_[axis] = stride;
            stride *= shape_[axis];
        }
    }

    void allocate_owned(TensorInit init, TensorZeroPolicy zero_policy)
    {
        owned_ = detail::allocate_aligned<T>(size_);
        data_ = owned_.get();
        if (init == TensorInit::zero && size_ != 0) zero(zero_policy);
    }

    void ensure_writable() const
    {
        if (read_only_) [[unlikely]] {
            throw Error(ErrorCode::unsupported_operation,
                        "cannot modify a read-only mapped tensor");
        }
    }

    void check_indices(const shape_type& indices) const
    {
        for (std::size_t axis = 0; axis < Rank; ++axis) {
            if (indices[axis] >= shape_[axis]) {
                throw std::out_of_range("tensor index is out of range");
            }
        }
    }

    void sync_header_to_mapping()
    {
        if (!mapping_) return;
        const auto header = detail::make_header<T, Rank>(
            shape_, layout_, tag_, payload_offset_);
        std::memcpy(mapping_->data(), &header, sizeof(header));
    }

    void close_storage() noexcept
    {
        mapping_.reset();
        owned_.reset();
        data_ = nullptr;
        path_.clear();
        read_only_ = false;
        payload_offset_ = 0;
    }

    void reset_metadata() noexcept
    {
        shape_.fill(0);
        strides_.fill(0);
        size_ = 0;
        layout_ = TensorLayout::GenericColumnMajor;
        tag_.fill(0);
    }

    void move_from(Tensor&& other) noexcept
    {
        shape_ = other.shape_;
        strides_ = other.strides_;
        size_ = other.size_;
        owned_ = std::move(other.owned_);
        mapping_ = std::move(other.mapping_);
        data_ = other.data_;
        path_ = std::move(other.path_);
        read_only_ = other.read_only_;
        payload_offset_ = other.payload_offset_;
        layout_ = other.layout_;
        tag_ = other.tag_;

        other.data_ = nullptr;
        other.path_.clear();
        other.read_only_ = false;
        other.payload_offset_ = 0;
        other.reset_metadata();
    }

    shape_type shape_{};
    shape_type strides_{};
    size_type size_ = 0;
    detail::AlignedArray<T> owned_{};
    std::unique_ptr<detail::MappedRegion> mapping_{};
    T* data_ = nullptr;
    std::filesystem::path path_{};
    bool read_only_ = false;
    std::uint64_t payload_offset_ = 0;
    TensorLayout layout_ = TensorLayout::GenericColumnMajor;
    std::array<std::uint8_t, tensor_tag_bytes> tag_{};
};

template <typename T, std::size_t Rank>
void swap(Tensor<T, Rank>& left, Tensor<T, Rank>& right) noexcept
{
    left.swap(right);
}

} // namespace sl2cfoamxx
