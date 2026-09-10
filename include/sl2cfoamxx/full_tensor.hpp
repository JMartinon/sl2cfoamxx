// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/export.hpp"
#include "sl2cfoamxx/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <limits>
#include <optional>
#include <span>
#include <variant>

namespace sl2cfoamxx {

// Physical ordering used by a reassociated full-tensor payload.  Rectangular
// storage contains every canonical entry in by-i1 panel order.  Reflection
// triangular storage contains one representative of
//
//   A(i1,i2,i3,i4,i5) = A(i1,i5,i4,i3,i2).
enum class FullTensorReassociatedLayout : std::uint8_t {
    reflection_triangular_by_i1,
    rectangular_by_i1,

    // Legacy source spellings; values and persisted files are unchanged.
    reflection_triangular_by_a1 = reflection_triangular_by_i1,
    rectangular_by_a1 = rectangular_by_i1
};

// Controls the explicit, canonical diagnostic dump produced by
// FullTensor::print(). The default limit prevents an accidental multi-million
// line dump; request all_values only when a complete dump is intentional.
struct FullTensorPrintOptions {
    static constexpr std::size_t all_values =
        std::numeric_limits<std::size_t>::max();

    std::size_t max_values = 4096;
    int precision = std::numeric_limits<double>::max_digits10;
    bool include_metadata = true;
};

// Scientific inputs attached to tensors produced by compute_full_tensor().
// A FullTensor constructed only as a general rank-five container has no such
// metadata, even when its structural tag happens to contain zero-valued
// placeholders for these fields.
struct FullTensorScientificMetadata {
    DSpins spins{};
    std::size_t Dl = 0;
    double gamma = 0.0;
    SimplicityConstraint simplicity_constraint =
        SimplicityConstraint::gamma_j;
};

// Typed view of the optional scientific inputs and the always-available
// representation/backing of the current container. `storage` is the public
// dense/reassociated storage family plus its memory/on-disk backing; use
// `reassociated_layout` or FullTensor::is_symmetry_compressed() to distinguish
// triangular compression from a rectangular reassociated payload. `layout`
// describes the physical payload order. This is the preferred public
// alternative to decoding FullTensor::tag() in applications or bindings.
struct FullTensorMetadata {
    std::optional<FullTensorScientificMetadata> scientific{};
    std::array<std::size_t, 5> logical_shape{};
    std::size_t logical_values = 0;
    std::size_t stored_values = 0;
    TensorStorage storage = TensorStorage::in_memory_dense;
    TensorLayout layout = TensorLayout::CanonicalDenseColumnMajor;
    std::optional<FullTensorReassociatedLayout> reassociated_layout{};
    bool mapped = false;
    bool writable = true;
    std::filesystem::path path{};
};

// A rank-five logical full tensor whose physical payload may either be dense
// canonical storage or the reassociated representation produced directly by
// the contraction backends.  Canonical operator()/at() access works for both
// representations.  Raw iteration, data(), span(), operator[] and size()
// intentionally expose the physical stored payload.
class SL2CFOAMXX_API FullTensor {
public:
    using value_type = double;
    using size_type = std::size_t;
    using shape_type = std::array<size_type, 5>;
    using strides_type = shape_type;
    using tag_type = std::array<std::uint8_t, tensor_tag_bytes>;
    using dense_tensor_type = Tensor<double, 5>;
    using reassociated_tensor_type = Tensor<double, 1>;
    static constexpr std::size_t rank = 5;

    FullTensor();

    explicit FullTensor(
        const shape_type& shape,
        TensorInit init = TensorInit::zero,
        TensorLayout layout = TensorLayout::CanonicalDenseColumnMajor);

    explicit FullTensor(const shape_type& shape, bool zero_initialize);

    // Dense Tensor compatibility. A generic column-major Tensor is normalized
    // to the canonical dense full-tensor layout.
    FullTensor(dense_tensor_type tensor);
    FullTensor& operator=(dense_tensor_type tensor);

    FullTensor(const FullTensor&) = default;
    FullTensor& operator=(const FullTensor&) = default;
    FullTensor(FullTensor&& other) noexcept;
    FullTensor& operator=(FullTensor&& other) noexcept;
    ~FullTensor() = default;

    [[nodiscard]] static FullTensor create_mapped(
        const std::filesystem::path& file,
        const shape_type& shape,
        TensorInit init = TensorInit::zero,
        TensorLayout layout = TensorLayout::CanonicalDenseColumnMajor,
        const tag_type& tag = {});

    [[nodiscard]] static FullTensor create_file(
        const std::filesystem::path& file,
        const shape_type& shape,
        TensorInit init = TensorInit::zero,
        TensorLayout layout = TensorLayout::CanonicalDenseColumnMajor,
        const tag_type& tag = {})
    {
        return create_mapped(file, shape, init, layout, tag);
    }

    // Construct from a backend-produced reassociated payload. The one-argument
    // overload reads the canonical dimensions and storage order from its binary
    // tag. The explicit overload is also useful while constructing a new result;
    // an empty tag is replaced by structurally valid metadata.
    [[nodiscard]] static FullTensor from_reassociated(
        reassociated_tensor_type tensor);

    [[nodiscard]] static FullTensor from_reassociated(
        reassociated_tensor_type tensor,
        const shape_type& canonical_shape,
        FullTensorReassociatedLayout layout,
        const tag_type& tag = {});

    [[nodiscard]] static FullTensor load(const std::filesystem::path& file);

    [[nodiscard]] static FullTensor map(
        const std::filesystem::path& file,
        TensorAccess access = TensorAccess::read_only);

    [[nodiscard]] static FullTensor open_file_existing(
        const std::filesystem::path& file,
        bool read_only = false)
    {
        return map(file, read_only ? TensorAccess::read_only
                                   : TensorAccess::read_write);
    }

    void save(const std::filesystem::path& file) const;
    void flush() const;

    // Losslessly convert an existing mapped full tensor between the compact
    // on-disk symmetry representation and canonical dense on-disk storage.
    // The source is never modified. The destination path is explicit and is
    // rejected when it already exists unless overwrite_existing is true.
    // Dense-to-symmetry conversion additionally requires every reflected pair
    // to have identical double-precision bits; otherwise it would discard
    // information and throws. The returned tensor is a writable mapping of the
    // atomically published destination file.
    [[nodiscard]] FullTensor convert_storage(
        TensorStorage destination_storage,
        const std::filesystem::path& destination_path,
        bool overwrite_existing = false) const;

    // Print logical canonical A[i1,i2,i3,i4,i5] entries, independent of the
    // physical dense/reassociated/symmetry-compressed representation. Indices
    // are zero-based and i1 varies fastest. The stream overload preserves the
    // caller's formatting state.
    void print(
        std::ostream& output,
        const FullTensorPrintOptions& options = {}) const;
    void print(const FullTensorPrintOptions& options = {}) const;

    [[nodiscard]] double* data();
    [[nodiscard]] const double* data() const noexcept;

    [[nodiscard]] std::span<double> span();
    [[nodiscard]] std::span<const double> span() const noexcept;

    [[nodiscard]] double* begin();
    [[nodiscard]] double* end();
    [[nodiscard]] const double* begin() const noexcept;
    [[nodiscard]] const double* end() const noexcept;
    [[nodiscard]] const double* cbegin() const noexcept;
    [[nodiscard]] const double* cend() const noexcept;

    // Flat access addresses physical storage. Use operator()/at() below for
    // canonical (i1,i2,i3,i4,i5) access independent of representation.
    [[nodiscard]] double& operator[](size_type index);
    [[nodiscard]] const double& operator[](size_type index) const noexcept;
    [[nodiscard]] double& at_flat(size_type index);
    [[nodiscard]] const double& at_flat(size_type index) const;

    [[nodiscard]] double& operator()(const shape_type& indices);
    [[nodiscard]] const double& operator()(const shape_type& indices) const;

    template <typename... Indices>
    [[nodiscard]] double& operator()(Indices... indices)
    {
        static_assert(sizeof...(Indices) == rank,
                      "incorrect number of full-tensor indices");
        return (*this)(shape_type{static_cast<size_type>(indices)...});
    }

    template <typename... Indices>
    [[nodiscard]] const double& operator()(Indices... indices) const
    {
        static_assert(sizeof...(Indices) == rank,
                      "incorrect number of full-tensor indices");
        return (*this)(shape_type{static_cast<size_type>(indices)...});
    }

    [[nodiscard]] double& at(const shape_type& indices);
    [[nodiscard]] const double& at(const shape_type& indices) const;

    template <typename... Indices>
    [[nodiscard]] double& at(Indices... indices)
    {
        static_assert(sizeof...(Indices) == rank,
                      "incorrect number of full-tensor indices");
        return at(shape_type{detail::checked_index_value(indices)...});
    }

    template <typename... Indices>
    [[nodiscard]] const double& at(Indices... indices) const
    {
        static_assert(sizeof...(Indices) == rank,
                      "incorrect number of full-tensor indices");
        return at(shape_type{detail::checked_index_value(indices)...});
    }

    void fill(double value);
    void zero();
    void clear() noexcept;

    [[nodiscard]] FullTensor clone() const;
    [[nodiscard]] FullTensor dense_copy() const;

    // shape()/dims()/strides() describe the logical canonical tensor, whereas
    // size()/stored_size() and raw spans describe its physical representation.
    [[nodiscard]] const shape_type& shape() const noexcept;
    [[nodiscard]] const shape_type& dims() const noexcept;
    [[nodiscard]] const strides_type& strides() const noexcept;
    [[nodiscard]] size_type extent(std::size_t axis) const;

    [[nodiscard]] size_type size() const noexcept;
    [[nodiscard]] size_type stored_size() const noexcept;
    [[nodiscard]] size_type logical_size() const noexcept;
    [[nodiscard]] size_type dense_size() const noexcept;
    [[nodiscard]] size_type size_bytes() const noexcept;
    [[nodiscard]] size_type stored_bytes() const noexcept;
    [[nodiscard]] size_type dense_bytes() const noexcept;
    [[nodiscard]] std::uint64_t bytes() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    [[nodiscard]] bool is_dense() const noexcept;
    [[nodiscard]] bool is_reassociated() const noexcept;
    [[nodiscard]] bool is_symmetry_compressed() const noexcept;
    [[nodiscard]] FullTensorReassociatedLayout reassociated_layout() const;

    [[nodiscard]] TensorStorage storage() const noexcept;
    [[nodiscard]] bool is_mapped() const noexcept;
    [[nodiscard]] bool writable() const noexcept;
    [[nodiscard]] bool is_read_only() const noexcept;
    [[nodiscard]] const std::filesystem::path& path() const noexcept;
    [[nodiscard]] const std::filesystem::path& root_path() const noexcept;
    [[nodiscard]] TensorLayout layout() const noexcept;
    [[nodiscard]] TensorLayout layout_code() const noexcept;
    [[nodiscard]] const tag_type& tag() const noexcept;
    [[nodiscard]] const tag_type& get_tag() const noexcept;

    // Decode the binary tag and combine it with the current physical storage
    // state. A malformed tag throws ErrorCode::invalid_tensor_file.
    [[nodiscard]] FullTensorMetadata metadata() const;

    void set_layout(TensorLayout layout);
    void set_tag(std::span<const std::uint8_t> tag);
    void set_tag(const tag_type& tag);

    void swap(FullTensor& other) noexcept;

private:
    enum class Representation : std::uint8_t {
        dense,
        reflection_triangular_by_i1,
        rectangular_by_i1
    };

    using storage_variant =
        std::variant<dense_tensor_type, reassociated_tensor_type>;

    FullTensor(
        reassociated_tensor_type tensor,
        const shape_type& canonical_shape,
        Representation representation);

    void initialize_logical_metadata(const shape_type& shape);
    void validate_representation() const;
    void check_indices(const shape_type& indices) const;
    [[nodiscard]] size_type physical_index(
        const shape_type& indices) const noexcept;

    [[nodiscard]] dense_tensor_type& dense_tensor();
    [[nodiscard]] const dense_tensor_type& dense_tensor() const;
    [[nodiscard]] reassociated_tensor_type& reassociated_tensor();
    [[nodiscard]] const reassociated_tensor_type& reassociated_tensor() const;

    storage_variant storage_{};
    shape_type logical_shape_{};
    strides_type logical_strides_{};
    size_type logical_size_ = 0;
    size_type triangular_panel_size_ = 0;
    Representation representation_ = Representation::dense;
};

inline void swap(FullTensor& left, FullTensor& right) noexcept
{
    left.swap(right);
}

} // namespace sl2cfoamxx
