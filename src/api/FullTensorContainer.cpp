// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "sl2cfoamxx/full_tensor.hpp"

#include "api/ConfigurationInternal.hpp"
#include "core/Ieee754.hpp"
#include "core/tensors.hpp"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sl2cfoamxx {
namespace {

using PublicOrder = FullTensorReassociatedLayout;
using InternalOrder = FullTensorReassociatedStorageOrder;
using Shape = FullTensor::shape_type;
using Tag = FullTensor::tag_type;

// Tags predate the distinction between a backend-produced tensor with the
// legitimate all-zero input and a general FullTensor whose mandatory tag used
// zero placeholders. Keep the V1 tag and all existing cache readers intact,
// and mark only newly-created structural tags in an otherwise reserved slot.
// An absent marker therefore continues to mean a scientific/backend tag, as
// it did for every existing cache file.
inline constexpr std::size_t structural_provenance_tag_slot = 2;
inline constexpr std::uint64_t structural_provenance_tag_marker =
    0x5354525543545552ULL; // "STRUCTUR"

class StreamFormatGuard {
public:
    explicit StreamFormatGuard(std::ostream& output) noexcept
        : output_(output), flags_(output.flags()),
          precision_(output.precision()), fill_(output.fill())
    {}

    ~StreamFormatGuard()
    {
        output_.flags(flags_);
        output_.precision(precision_);
        output_.fill(fill_);
    }

    StreamFormatGuard(const StreamFormatGuard&) = delete;
    StreamFormatGuard& operator=(const StreamFormatGuard&) = delete;

private:
    std::ostream& output_;
    std::ios_base::fmtflags flags_;
    std::streamsize precision_;
    char fill_;
};

[[nodiscard]] bool tag_is_zero(const Tag& tag) noexcept
{
    return std::all_of(tag.begin(), tag.end(), [](std::uint8_t value) {
        return value == 0;
    });
}

[[nodiscard]] std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    const char* description)
{
    if (left != 0 &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(description);
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_logical_size(const Shape& shape)
{
    std::size_t count = 1;
    for (const std::size_t extent : shape) {
        count = checked_multiply(
            count, extent, "full-tensor logical shape overflows size_t");
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        throw std::overflow_error(
            "full-tensor logical byte count overflows size_t");
    }
    return count;
}

[[nodiscard]] std::size_t checked_triangular_size(std::size_t extent)
{
    if (extent == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(
            "full-tensor triangular extent overflows size_t");
    }
    std::size_t left = extent;
    std::size_t right = extent + 1;
    if ((left & 1u) == 0) {
        left /= 2;
    } else {
        right /= 2;
    }
    return checked_multiply(
        left, right, "full-tensor triangular payload overflows size_t");
}

[[nodiscard]] InternalOrder to_internal_order(PublicOrder order) noexcept
{
    switch (order) {
    case PublicOrder::reflection_triangular_by_i1:
        return InternalOrder::ReflectionTriangularByI1;
    case PublicOrder::rectangular_by_i1:
        return InternalOrder::RectangularByI1;
    }
    return InternalOrder::Unknown;
}

[[nodiscard]] PublicOrder to_public_order(InternalOrder order)
{
    switch (order) {
    case InternalOrder::ReflectionTriangularByI1:
        return PublicOrder::reflection_triangular_by_i1;
    case InternalOrder::RectangularByI1:
        return PublicOrder::rectangular_by_i1;
    case InternalOrder::Unknown:
        break;
    }
    throw Error(
        ErrorCode::invalid_tensor_file,
        "full-tensor metadata has no reassociated storage order");
}

[[nodiscard]] FullTensorBinaryTagV1 parse_tag(const Tag& bytes)
{
    FullTensorBinaryTagV1 tag{};
    if (!read_full_tensor_binary_tag(bytes, tag)) {
        throw Error(
            ErrorCode::invalid_tensor_file,
            "full-tensor metadata tag is missing or invalid");
    }
    return tag;
}

[[nodiscard]] Shape canonical_shape_from_tag(
    const FullTensorBinaryTagV1& tag)
{
    Shape shape{};
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        if (tag.canonical_dims[axis] >
            std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error(
                "full-tensor canonical extent does not fit size_t");
        }
        shape[axis] = static_cast<std::size_t>(tag.canonical_dims[axis]);
    }
    (void)checked_logical_size(shape);
    return shape;
}

void validate_common_tag(
    const FullTensorBinaryTagV1& tag,
    TensorLayout expected_layout,
    const Shape& canonical_shape,
    std::size_t physical_values)
{
    if (tag.layout_code != static_cast<std::uint32_t>(expected_layout)) {
        throw Error(
            ErrorCode::invalid_tensor_file,
            "full-tensor tag layout does not match its payload layout");
    }
    if (tag.values_count != static_cast<std::uint64_t>(physical_values)) {
        throw Error(
            ErrorCode::invalid_tensor_file,
            "full-tensor tag value count does not match its payload");
    }
    for (std::size_t axis = 0; axis < canonical_shape.size(); ++axis) {
        if (tag.canonical_dims[axis] !=
            static_cast<std::uint64_t>(canonical_shape[axis])) {
            throw Error(
                ErrorCode::invalid_tensor_file,
                "full-tensor tag canonical dimensions do not match its payload");
        }
    }
    SimplicityConstraint ignored{};
    if (!read_full_tensor_simplicity_constraint(tag, ignored)) {
        throw Error(
            ErrorCode::invalid_tensor_file,
            "full-tensor tag has an unknown simplicity constraint");
    }
}

void validate_dense_tag(
    const Tag& bytes,
    const Shape& canonical_shape,
    std::size_t physical_values)
{
    const FullTensorBinaryTagV1 tag = parse_tag(bytes);
    validate_common_tag(
        tag,
        TensorLayout::CanonicalDenseColumnMajor,
        canonical_shape,
        physical_values);
}

void validate_reassociated_tag(
    const Tag& bytes,
    const Shape& canonical_shape,
    std::size_t physical_values,
    PublicOrder expected_order)
{
    const FullTensorBinaryTagV1 tag = parse_tag(bytes);
    validate_common_tag(
        tag,
        TensorLayout::FullTensorReassociatedReflectionCompactByI1,
        canonical_shape,
        physical_values);
    if (full_tensor_reassociated_storage_order(tag) !=
        to_internal_order(expected_order)) {
        throw Error(
            ErrorCode::invalid_tensor_file,
            "full-tensor tag reassociated storage order does not match its payload");
    }
    if (expected_order == PublicOrder::reflection_triangular_by_i1 &&
        !(tag.spins[0] == tag.spins[3] &&
          tag.spins[1] == tag.spins[2] &&
          tag.spins[4] == tag.spins[9] &&
          tag.spins[5] == tag.spins[8])) {
        throw Error(
            ErrorCode::invalid_tensor_file,
            "reflection-compressed full tensor has non-invariant boundary spins");
    }
}

[[nodiscard]] Tag make_structural_dense_tag(
    const Shape& shape,
    std::size_t values)
{
    const std::array<int, 10> spins{};
    const Tag bytes = make_full_tensor_binary_tag_canonical(
        spins, 0, 0.0, shape, values, SimplicityConstraint::gamma_j);
    FullTensorBinaryTagV1 tag{};
    (void)read_full_tensor_binary_tag(bytes, tag);
    tag.reserved[structural_provenance_tag_slot] =
        structural_provenance_tag_marker;
    return serialize_full_tensor_binary_tag(tag);
}

[[nodiscard]] Tag make_structural_reassociated_tag(
    const Shape& shape,
    std::size_t values,
    PublicOrder order)
{
    const std::array<int, 10> spins{};
    const Tag bytes =
        make_full_tensor_binary_tag_reassociated_with_storage_order(
        spins,
        0,
        0.0,
        shape,
        values,
        to_internal_order(order),
        SimplicityConstraint::gamma_j);
    FullTensorBinaryTagV1 tag{};
    (void)read_full_tensor_binary_tag(bytes, tag);
    tag.reserved[structural_provenance_tag_slot] =
        structural_provenance_tag_marker;
    return serialize_full_tensor_binary_tag(tag);
}

[[nodiscard]] bool is_structural_tag(
    const FullTensorBinaryTagV1& tag) noexcept
{
    return tag.reserved[structural_provenance_tag_slot] ==
        structural_provenance_tag_marker;
}

[[nodiscard]] Tag canonicalized_dense_tag(
    const Tag& bytes,
    const Shape& shape,
    std::size_t values)
{
    FullTensorBinaryTagV1 tag = parse_tag(bytes);
    tag.layout_code = static_cast<std::uint32_t>(
        TensorLayout::CanonicalDenseColumnMajor);
    tag.values_count = static_cast<std::uint64_t>(values);
    tag.reserved[full_tensor_storage_order_tag_slot] =
        static_cast<std::uint64_t>(InternalOrder::Unknown);
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        tag.canonical_dims[axis] = static_cast<std::uint64_t>(shape[axis]);
    }
    return serialize_full_tensor_binary_tag(tag);
}

[[nodiscard]] Tag canonicalized_reflection_tag(
    const Tag& bytes,
    const Shape& shape,
    std::size_t values)
{
    FullTensorBinaryTagV1 tag = parse_tag(bytes);
    tag.layout_code = static_cast<std::uint32_t>(
        TensorLayout::FullTensorReassociatedReflectionCompactByI1);
    tag.values_count = static_cast<std::uint64_t>(values);
    set_full_tensor_reassociated_storage_order(
        tag, InternalOrder::ReflectionTriangularByI1);
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        tag.canonical_dims[axis] = static_cast<std::uint64_t>(shape[axis]);
    }
    const Tag converted = serialize_full_tensor_binary_tag(tag);
    validate_reassociated_tag(
        converted,
        shape,
        values,
        PublicOrder::reflection_triangular_by_i1);
    return converted;
}

[[nodiscard]] std::filesystem::path normalized_conversion_path(
    const std::filesystem::path& path)
{
    if (path.empty()) {
        throw std::invalid_argument(
            "full-tensor conversion requires a nonempty destination path");
    }
    std::error_code error;
    const std::filesystem::path normalized =
        std::filesystem::weakly_canonical(path, error);
    if (error) {
        throw std::filesystem::filesystem_error(
            "cannot resolve the full-tensor conversion destination",
            path,
            error);
    }
    return normalized;
}

[[nodiscard]] bool paths_identify_same_file(
    const std::filesystem::path& left,
    const std::filesystem::path& right)
{
    std::error_code error;
    const std::filesystem::path normalized_left =
        std::filesystem::weakly_canonical(left, error);
    if (error) {
        throw std::filesystem::filesystem_error(
            "cannot resolve the full-tensor conversion source", left, error);
    }
    if (normalized_left == right) return true;

    const std::filesystem::file_status status =
        std::filesystem::symlink_status(right, error);
    if (error && error != std::errc::no_such_file_or_directory) {
        throw std::filesystem::filesystem_error(
            "cannot inspect the full-tensor conversion destination",
            right,
            error);
    }
    if (error || status.type() == std::filesystem::file_type::not_found) {
        return false;
    }

    error.clear();
    const bool equivalent = std::filesystem::equivalent(left, right, error);
    if (error) {
        throw std::filesystem::filesystem_error(
            "cannot compare the full-tensor conversion paths",
            left,
            right,
            error);
    }
    return equivalent;
}

[[nodiscard]] bool path_exists_without_following_symlinks(
    const std::filesystem::path& path)
{
    std::error_code error;
    const std::filesystem::file_status status =
        std::filesystem::symlink_status(path, error);
    if (error && error != std::errc::no_such_file_or_directory) {
        throw std::filesystem::filesystem_error(
            "cannot inspect the full-tensor conversion destination",
            path,
            error);
    }
    return !error && status.type() != std::filesystem::file_type::not_found;
}

[[nodiscard]] bool same_double_bits(double left, double right) noexcept
{
    return b4utils::ieee754::double_bits(left) ==
           b4utils::ieee754::double_bits(right);
}

void copy_canonical_values(
    const FullTensor& source,
    FullTensor& destination)
{
    const Shape& shape = source.shape();
    const std::size_t I1 = shape[0];
    const std::size_t I2 = shape[1];
    const std::size_t I3 = shape[2];
    const std::size_t I4 = shape[3];
    const std::size_t outer_count =
        I1 == 0 ? 0 : destination.logical_size() / I1;
    double* const output = destination.data();

    const auto copy_outer_column = [&](std::size_t outer) {
        std::size_t remainder = outer;
        const std::size_t ii2 = remainder % I2;
        remainder /= I2;
        const std::size_t ii3 = remainder % I3;
        remainder /= I3;
        const std::size_t ii4 = remainder % I4;
        const std::size_t ii5 = remainder / I4;
        double* const column = output + outer * I1;
        for (std::size_t ii1 = 0; ii1 < I1; ++ii1) {
            column[ii1] = source(ii1, ii2, ii3, ii4, ii5);
        }
    };

#ifdef _OPENMP
    if (!omp_in_parallel() &&
        destination.size_bytes() >=
            Tensor<double, 5>::parallel_zero_threshold_bytes() &&
        outer_count <= static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max())) {
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t outer = 0;
             outer < static_cast<std::ptrdiff_t>(outer_count); ++outer) {
            copy_outer_column(static_cast<std::size_t>(outer));
        }
        return;
    }
#endif

    for (std::size_t outer = 0; outer < outer_count; ++outer) {
        copy_outer_column(outer);
    }
}

void copy_reflection_representatives(
    const FullTensor& source,
    FullTensor& destination)
{
    const Shape& shape = source.shape();
    if (shape[1] != shape[4] || shape[2] != shape[3]) {
        throw Error(
            ErrorCode::unsupported_operation,
            "dense full tensor cannot be represented by reflection storage: "
            "canonical axis extents are incompatible");
    }

    const std::size_t rows = checked_multiply(
        shape[1],
        shape[2],
        "full-tensor reflection row extent overflows size_t");
    const std::size_t triangular_panel_size = checked_triangular_size(rows);
    const std::size_t tasks = checked_multiply(
        shape[0], rows, "full-tensor reflection copy extent overflows size_t");
    double* const output = destination.data();
    std::atomic<bool> mismatch{false};

    const auto copy_row = [&](std::size_t task) {
        const std::size_t ii1 = task / rows;
        const std::size_t high = task % rows;
        const std::size_t high_first = high % shape[1];
        const std::size_t high_second = high / shape[1];
        // Equivalent to high*(high+1)/2 without overflowing the intermediate;
        // checked_triangular_size(rows) above proved the result is representable.
        const std::size_t high_offset = high % 2 == 0
            ? (high / 2) * (high + 1)
            : high * ((high + 1) / 2);
        for (std::size_t low = 0; low <= high; ++low) {
            const std::size_t low_first = low % shape[1];
            const std::size_t low_second = low / shape[1];
            const double value = source(
                ii1,
                low_first,
                low_second,
                high_second,
                high_first);
            const double reflected = source(
                ii1,
                high_first,
                high_second,
                low_second,
                low_first);
            if (!same_double_bits(value, reflected)) {
                mismatch.store(true, std::memory_order_relaxed);
            } else {
                output[high_offset + low + triangular_panel_size * ii1] =
                    value;
            }
        }
    };

#ifdef _OPENMP
    if (!omp_in_parallel() &&
        destination.size_bytes() >=
            Tensor<double, 5>::parallel_zero_threshold_bytes() &&
        tasks <= static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max())) {
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t task = 0;
             task < static_cast<std::ptrdiff_t>(tasks); ++task) {
            copy_row(static_cast<std::size_t>(task));
        }
    } else
#endif
    {
        for (std::size_t task = 0; task < tasks; ++task) {
            copy_row(task);
        }
    }

    if (mismatch.load(std::memory_order_relaxed)) {
        throw Error(
            ErrorCode::unsupported_operation,
            "dense full tensor cannot be losslessly converted to reflection "
            "storage: a reflected value pair differs");
    }
}

} // namespace

FullTensor::FullTensor()
    : FullTensor(shape_type{}, TensorInit::zero)
{
}

FullTensor::FullTensor(
    const shape_type& shape,
    TensorInit init,
    TensorLayout layout)
    : FullTensor(dense_tensor_type(shape, init, layout))
{
}

FullTensor::FullTensor(const shape_type& shape, bool zero_initialize)
    : FullTensor(
          shape,
          zero_initialize ? TensorInit::zero : TensorInit::uninitialized)
{
}

FullTensor::FullTensor(dense_tensor_type tensor)
    : representation_(Representation::dense)
{
    if (tensor.layout() == TensorLayout::GenericColumnMajor) {
        if (!tensor.writable()) {
            throw Error(
                ErrorCode::unsupported_operation,
                "cannot normalize a read-only generic tensor as a full tensor");
        }
        tensor.set_layout(TensorLayout::CanonicalDenseColumnMajor);
    }
    if (tensor.layout() != TensorLayout::CanonicalDenseColumnMajor) {
        throw std::invalid_argument(
            "dense FullTensor construction requires canonical column-major layout");
    }

    initialize_logical_metadata(tensor.shape());
    if (tag_is_zero(tensor.tag())) {
        if (!tensor.writable()) {
            throw Error(
                ErrorCode::invalid_tensor_file,
                "read-only full tensor has no valid metadata tag");
        }
        tensor.set_tag(make_structural_dense_tag(tensor.shape(), tensor.size()));
    } else {
        validate_dense_tag(tensor.tag(), tensor.shape(), tensor.size());
    }
    storage_ = std::move(tensor);
    validate_representation();
}

FullTensor& FullTensor::operator=(dense_tensor_type tensor)
{
    FullTensor replacement(std::move(tensor));
    swap(replacement);
    return *this;
}

FullTensor::FullTensor(FullTensor&& other) noexcept
    : storage_(std::move(other.storage_)),
      logical_shape_(other.logical_shape_),
      logical_strides_(other.logical_strides_),
      logical_size_(other.logical_size_),
      triangular_panel_size_(other.triangular_panel_size_),
      representation_(other.representation_)
{
    other.clear();
}

FullTensor& FullTensor::operator=(FullTensor&& other) noexcept
{
    if (this != &other) {
        storage_ = std::move(other.storage_);
        logical_shape_ = other.logical_shape_;
        logical_strides_ = other.logical_strides_;
        logical_size_ = other.logical_size_;
        triangular_panel_size_ = other.triangular_panel_size_;
        representation_ = other.representation_;
        other.clear();
    }
    return *this;
}

FullTensor::FullTensor(
    reassociated_tensor_type tensor,
    const shape_type& canonical_shape,
    Representation representation)
    : storage_(std::move(tensor)), representation_(representation)
{
    initialize_logical_metadata(canonical_shape);
    validate_representation();
    const PublicOrder order = reassociated_layout();
    validate_reassociated_tag(
        reassociated_tensor().tag(),
        logical_shape_,
        reassociated_tensor().size(),
        order);
}

FullTensor FullTensor::create_mapped(
    const std::filesystem::path& file,
    const shape_type& shape,
    TensorInit init,
    TensorLayout layout,
    const tag_type& tag)
{
    if (layout == TensorLayout::GenericColumnMajor) {
        layout = TensorLayout::CanonicalDenseColumnMajor;
    }
    if (layout != TensorLayout::CanonicalDenseColumnMajor) {
        throw std::invalid_argument(
            "FullTensor::create_mapped creates only dense canonical tensors; "
            "use from_reassociated for a reassociated payload");
    }
    const std::size_t values = checked_logical_size(shape);
    const tag_type actual_tag = tag_is_zero(tag)
        ? make_structural_dense_tag(shape, values)
        : tag;
    validate_dense_tag(actual_tag, shape, values);
    return FullTensor(dense_tensor_type::create_mapped(
        file, shape, init, layout, actual_tag));
}

FullTensor FullTensor::from_reassociated(reassociated_tensor_type tensor)
{
    if (tensor.layout() !=
        TensorLayout::FullTensorReassociatedReflectionCompactByI1) {
        throw std::invalid_argument(
            "reassociated FullTensor payload has the wrong tensor layout");
    }
    const FullTensorBinaryTagV1 tag = parse_tag(tensor.tag());
    const shape_type shape = canonical_shape_from_tag(tag);
    const FullTensorReassociatedLayout order =
        to_public_order(full_tensor_reassociated_storage_order(tag));
    const Representation representation =
        order == FullTensorReassociatedLayout::reflection_triangular_by_i1
            ? Representation::reflection_triangular_by_i1
            : Representation::rectangular_by_i1;
    return FullTensor(std::move(tensor), shape, representation);
}

FullTensor FullTensor::from_reassociated(
    reassociated_tensor_type tensor,
    const shape_type& canonical_shape,
    FullTensorReassociatedLayout layout,
    const tag_type& tag)
{
    const TensorLayout physical_layout =
        TensorLayout::FullTensorReassociatedReflectionCompactByI1;
    if (tensor.layout() != physical_layout) {
        if (tensor.layout() != TensorLayout::GenericColumnMajor) {
            throw std::invalid_argument(
                "reassociated FullTensor payload has the wrong tensor layout");
        }
        if (!tensor.writable()) {
            throw Error(
                ErrorCode::unsupported_operation,
                "cannot normalize a read-only reassociated tensor layout");
        }
        tensor.set_layout(physical_layout);
    }

    tag_type actual_tag = tag;
    if (tag_is_zero(actual_tag)) {
        FullTensorBinaryTagV1 existing{};
        if (read_full_tensor_binary_tag(tensor.tag(), existing)) {
            actual_tag = tensor.tag();
        } else {
            actual_tag = make_structural_reassociated_tag(
                canonical_shape, tensor.size(), layout);
        }
    }
    validate_reassociated_tag(
        actual_tag, canonical_shape, tensor.size(), layout);
    if (tensor.tag() != actual_tag) {
        tensor.set_tag(actual_tag);
    }

    const Representation representation =
        layout == FullTensorReassociatedLayout::reflection_triangular_by_i1
            ? Representation::reflection_triangular_by_i1
            : Representation::rectangular_by_i1;
    return FullTensor(std::move(tensor), canonical_shape, representation);
}

FullTensor FullTensor::load(const std::filesystem::path& file)
{
    const tensor_disk::TensorTreeHeaderV1 header =
        tensor_disk::read_header_file(file);
    const TensorLayout layout = static_cast<TensorLayout>(header.layout_code);
    if (header.ndim == 5 &&
        layout == TensorLayout::CanonicalDenseColumnMajor) {
        return FullTensor(dense_tensor_type::load(file));
    }
    if (header.ndim == 1 &&
        layout ==
            TensorLayout::FullTensorReassociatedReflectionCompactByI1) {
        return from_reassociated(reassociated_tensor_type::load(file));
    }
    throw Error(
        ErrorCode::invalid_tensor_file,
        "file is not a supported dense or reassociated full tensor");
}

FullTensor FullTensor::map(
    const std::filesystem::path& file,
    TensorAccess access)
{
    const tensor_disk::TensorTreeHeaderV1 header =
        tensor_disk::read_header_file(file);
    const TensorLayout layout = static_cast<TensorLayout>(header.layout_code);
    if (header.ndim == 5 &&
        layout == TensorLayout::CanonicalDenseColumnMajor) {
        return FullTensor(dense_tensor_type::map(file, access));
    }
    if (header.ndim == 1 &&
        layout ==
            TensorLayout::FullTensorReassociatedReflectionCompactByI1) {
        return from_reassociated(reassociated_tensor_type::map(file, access));
    }
    throw Error(
        ErrorCode::invalid_tensor_file,
        "file is not a supported dense or reassociated full tensor");
}

void FullTensor::save(const std::filesystem::path& file) const
{
    validate_representation();
    if (is_dense()) {
        validate_dense_tag(tag(), logical_shape_, size());
        dense_tensor().save(file);
    } else {
        validate_reassociated_tag(
            tag(), logical_shape_, size(), reassociated_layout());
        reassociated_tensor().save(file);
    }
}

void FullTensor::flush() const
{
    if (is_dense()) {
        dense_tensor().flush();
    } else {
        reassociated_tensor().flush();
    }
}

FullTensor FullTensor::convert_storage(
    TensorStorage destination_storage,
    const std::filesystem::path& destination_path,
    bool overwrite_existing) const
{
    if (!is_mapped() ||
        (storage() != TensorStorage::on_disk_sym &&
         storage() != TensorStorage::on_disk_dense)) {
        throw std::invalid_argument(
            "full-tensor storage conversion requires an on-disk source");
    }
    if (destination_storage != TensorStorage::on_disk_sym &&
        destination_storage != TensorStorage::on_disk_dense) {
        throw std::invalid_argument(
            "full-tensor storage conversion destination must be "
            "TensorStorage::on_disk_sym or TensorStorage::on_disk_dense");
    }
    if (destination_storage == storage()) {
        throw std::invalid_argument(
            "full-tensor storage conversion requires the opposite on-disk "
            "storage mode");
    }

    validate_representation();
    if (is_dense()) {
        validate_dense_tag(tag(), logical_shape_, size());
    } else {
        validate_reassociated_tag(
            tag(), logical_shape_, size(), reassociated_layout());
    }

    const std::filesystem::path destination =
        normalized_conversion_path(destination_path);
    if (paths_identify_same_file(path(), destination)) {
        throw std::invalid_argument(
            "full-tensor conversion destination must differ from its source");
    }

    detail::TensorOutputLease output_lease(destination);
    if (!overwrite_existing &&
        path_exists_without_following_symlinks(destination)) {
        throw std::filesystem::filesystem_error(
            "full-tensor conversion destination already exists; pass "
            "overwrite_existing=true to replace it",
            destination,
            std::make_error_code(std::errc::file_exists));
    }

    detail::AtomicTensorOutput publication(destination);
    if (destination_storage == TensorStorage::on_disk_dense) {
        const Tag destination_tag = canonicalized_dense_tag(
            tag(), logical_shape_, logical_size_);
        {
            FullTensor converted = FullTensor::create_mapped(
                publication.temporary_path(),
                logical_shape_,
                TensorInit::uninitialized,
                TensorLayout::CanonicalDenseColumnMajor,
                destination_tag);
            copy_canonical_values(*this, converted);
            converted.flush();
        }
    } else {
        if (!is_dense()) {
            throw Error(
                ErrorCode::internal_failure,
                "only dense full tensors can be converted to reflection storage");
        }
        const std::size_t rows = checked_multiply(
            logical_shape_[1],
            logical_shape_[2],
            "full-tensor reflection row extent overflows size_t");
        const std::size_t stored_values = checked_multiply(
            logical_shape_[0],
            checked_triangular_size(rows),
            "reflection-compressed full-tensor payload overflows size_t");
        const Tag destination_tag = canonicalized_reflection_tag(
            tag(), logical_shape_, stored_values);
        {
            reassociated_tensor_type payload =
                reassociated_tensor_type::create_mapped(
                    publication.temporary_path(),
                    {{stored_values}},
                    TensorInit::uninitialized,
                    TensorLayout::
                        FullTensorReassociatedReflectionCompactByI1,
                    destination_tag);
            FullTensor converted = FullTensor::from_reassociated(
                std::move(payload),
                logical_shape_,
                FullTensorReassociatedLayout::reflection_triangular_by_i1,
                destination_tag);
            copy_reflection_representatives(*this, converted);
            converted.flush();
        }
    }

    publication.publish();
    return FullTensor::map(destination, TensorAccess::read_write);
}

double* FullTensor::data()
{
    return is_dense() ? dense_tensor().data() : reassociated_tensor().data();
}

const double* FullTensor::data() const noexcept
{
    return is_dense() ? dense_tensor().data() : reassociated_tensor().data();
}

std::span<double> FullTensor::span()
{
    return is_dense() ? dense_tensor().span() : reassociated_tensor().span();
}

std::span<const double> FullTensor::span() const noexcept
{
    return is_dense() ? dense_tensor().span() : reassociated_tensor().span();
}

double* FullTensor::begin()
{
    return is_dense() ? dense_tensor().begin() : reassociated_tensor().begin();
}

double* FullTensor::end()
{
    return is_dense() ? dense_tensor().end() : reassociated_tensor().end();
}

const double* FullTensor::begin() const noexcept
{
    return is_dense() ? dense_tensor().begin() : reassociated_tensor().begin();
}

const double* FullTensor::end() const noexcept
{
    return is_dense() ? dense_tensor().end() : reassociated_tensor().end();
}

const double* FullTensor::cbegin() const noexcept
{
    return begin();
}

const double* FullTensor::cend() const noexcept
{
    return end();
}

double& FullTensor::operator[](size_type index)
{
    return is_dense() ? dense_tensor()[index] : reassociated_tensor()[index];
}

const double& FullTensor::operator[](size_type index) const noexcept
{
    return is_dense() ? dense_tensor()[index] : reassociated_tensor()[index];
}

double& FullTensor::at_flat(size_type index)
{
    if (index >= size()) {
        throw std::out_of_range("flat full-tensor index is out of range");
    }
    return (*this)[index];
}

const double& FullTensor::at_flat(size_type index) const
{
    if (index >= size()) {
        throw std::out_of_range("flat full-tensor index is out of range");
    }
    return (*this)[index];
}

double& FullTensor::operator()(const shape_type& indices)
{
    return data()[physical_index(indices)];
}

const double& FullTensor::operator()(const shape_type& indices) const
{
    return data()[physical_index(indices)];
}

double& FullTensor::at(const shape_type& indices)
{
    check_indices(indices);
    return (*this)(indices);
}

const double& FullTensor::at(const shape_type& indices) const
{
    check_indices(indices);
    return (*this)(indices);
}

void FullTensor::fill(double value)
{
    if (is_dense()) {
        dense_tensor().fill(value);
    } else {
        reassociated_tensor().fill(value);
    }
}

void FullTensor::zero()
{
    if (is_dense()) {
        dense_tensor().zero();
    } else {
        reassociated_tensor().zero();
    }
}

void FullTensor::clear() noexcept
{
    storage_.template emplace<dense_tensor_type>();
    dense_tensor().set_layout(TensorLayout::CanonicalDenseColumnMajor);
    dense_tensor().set_tag(make_structural_dense_tag(shape_type{}, 0));
    logical_shape_.fill(0);
    logical_strides_.fill(0);
    logical_size_ = 0;
    triangular_panel_size_ = 0;
    representation_ = Representation::dense;
}

FullTensor FullTensor::clone() const
{
    return FullTensor(*this);
}

FullTensor FullTensor::dense_copy() const
{
    if (is_dense()) {
        return clone();
    }

    FullTensor output(logical_shape_, TensorInit::uninitialized);
    const std::size_t I1 = logical_shape_[0];
    const std::size_t I2 = logical_shape_[1];
    const std::size_t I3 = logical_shape_[2];
    const std::size_t I4 = logical_shape_[3];
    const std::size_t outer_count = I1 == 0 ? 0 : logical_size_ / I1;
    double* const destination = output.data();

    const auto copy_outer_column = [&](std::size_t outer) {
        std::size_t remainder = outer;
        const std::size_t ii2 = remainder % I2;
        remainder /= I2;
        const std::size_t ii3 = remainder % I3;
        remainder /= I3;
        const std::size_t ii4 = remainder % I4;
        const std::size_t ii5 = remainder / I4;
        double* const column = destination + outer * I1;
        for (std::size_t ii1 = 0; ii1 < I1; ++ii1) {
            column[ii1] = (*this)(ii1, ii2, ii3, ii4, ii5);
        }
    };

#ifdef _OPENMP
    // Static contiguous column ownership makes the copy itself establish the
    // destination's NUMA placement. Keep tiny copies serial to avoid launching
    // an OpenMP team solely for first touch.
    if (!omp_in_parallel() &&
        output.size_bytes() >=
            Tensor<double, 5>::parallel_zero_threshold_bytes() &&
        outer_count <= static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max())) {
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t outer = 0;
             outer < static_cast<std::ptrdiff_t>(outer_count); ++outer) {
            copy_outer_column(static_cast<std::size_t>(outer));
        }
    } else
#endif
    {
        for (std::size_t outer = 0; outer < outer_count; ++outer) {
            copy_outer_column(outer);
        }
    }
    output.set_tag(canonicalized_dense_tag(
        tag(), logical_shape_, logical_size_));
    return output;
}

const FullTensor::shape_type& FullTensor::shape() const noexcept
{
    return logical_shape_;
}

const FullTensor::shape_type& FullTensor::dims() const noexcept
{
    return logical_shape_;
}

const FullTensor::strides_type& FullTensor::strides() const noexcept
{
    return logical_strides_;
}

FullTensor::size_type FullTensor::extent(std::size_t axis) const
{
    if (axis >= rank) {
        throw std::out_of_range("full-tensor axis is out of range");
    }
    return logical_shape_[axis];
}

FullTensor::size_type FullTensor::size() const noexcept
{
    return is_dense() ? dense_tensor().size() : reassociated_tensor().size();
}

FullTensor::size_type FullTensor::stored_size() const noexcept
{
    return size();
}

FullTensor::size_type FullTensor::logical_size() const noexcept
{
    return logical_size_;
}

FullTensor::size_type FullTensor::dense_size() const noexcept
{
    return logical_size_;
}

FullTensor::size_type FullTensor::size_bytes() const noexcept
{
    return stored_bytes();
}

FullTensor::size_type FullTensor::stored_bytes() const noexcept
{
    return size() * sizeof(double);
}

FullTensor::size_type FullTensor::dense_bytes() const noexcept
{
    return logical_size_ * sizeof(double);
}

std::uint64_t FullTensor::bytes() const noexcept
{
    return static_cast<std::uint64_t>(stored_bytes());
}

bool FullTensor::empty() const noexcept
{
    return size() == 0;
}

bool FullTensor::is_dense() const noexcept
{
    return representation_ == Representation::dense;
}

bool FullTensor::is_reassociated() const noexcept
{
    return !is_dense();
}

bool FullTensor::is_symmetry_compressed() const noexcept
{
    return representation_ == Representation::reflection_triangular_by_i1;
}

FullTensorReassociatedLayout FullTensor::reassociated_layout() const
{
    switch (representation_) {
    case Representation::reflection_triangular_by_i1:
        return FullTensorReassociatedLayout::reflection_triangular_by_i1;
    case Representation::rectangular_by_i1:
        return FullTensorReassociatedLayout::rectangular_by_i1;
    case Representation::dense:
        break;
    }
    throw Error(ErrorCode::unsupported_operation,
                "dense full tensor has no reassociated layout");
}

TensorStorage FullTensor::storage() const noexcept
{
    if (is_dense()) {
        return is_mapped() ? TensorStorage::on_disk_dense
                           : TensorStorage::in_memory_dense;
    }
    return is_mapped() ? TensorStorage::on_disk_sym
                       : TensorStorage::in_memory_sym;
}

bool FullTensor::is_mapped() const noexcept
{
    return is_dense() ? dense_tensor().is_mapped()
                      : reassociated_tensor().is_mapped();
}

bool FullTensor::writable() const noexcept
{
    return is_dense() ? dense_tensor().writable()
                      : reassociated_tensor().writable();
}

bool FullTensor::is_read_only() const noexcept
{
    return !writable();
}

const std::filesystem::path& FullTensor::path() const noexcept
{
    return is_dense() ? dense_tensor().path() : reassociated_tensor().path();
}

const std::filesystem::path& FullTensor::root_path() const noexcept
{
    return path();
}

TensorLayout FullTensor::layout() const noexcept
{
    return is_dense() ? dense_tensor().layout()
                      : reassociated_tensor().layout();
}

TensorLayout FullTensor::layout_code() const noexcept
{
    return layout();
}

const FullTensor::tag_type& FullTensor::tag() const noexcept
{
    return is_dense() ? dense_tensor().tag() : reassociated_tensor().tag();
}

const FullTensor::tag_type& FullTensor::get_tag() const noexcept
{
    return tag();
}

FullTensorMetadata FullTensor::metadata() const
{
    const FullTensorBinaryTagV1 binary = parse_tag(tag());
    SimplicityConstraint simplicity{};
    if (!read_full_tensor_simplicity_constraint(binary, simplicity)) {
        throw Error(
            ErrorCode::invalid_tensor_file,
            "full-tensor tag has an unknown simplicity constraint");
    }
    if (binary.Dl > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(
            "full-tensor shell cutoff does not fit size_t");
    }

    FullTensorMetadata result;
    if (!is_structural_tag(binary)) {
        FullTensorScientificMetadata scientific;
        for (std::size_t index = 0; index < scientific.spins.size(); ++index) {
            if (binary.spins[index] < std::numeric_limits<int>::min() ||
                binary.spins[index] > std::numeric_limits<int>::max()) {
                throw std::overflow_error(
                    "full-tensor doubled spin does not fit int");
            }
            scientific.spins[index] = static_cast<int>(binary.spins[index]);
        }
        scientific.Dl = static_cast<std::size_t>(binary.Dl);
        scientific.gamma = binary.gamma;
        scientific.simplicity_constraint = simplicity;
        result.scientific = scientific;
    }
    result.logical_shape = logical_shape_;
    result.logical_values = logical_size_;
    result.stored_values = size();
    result.storage = storage();
    result.layout = layout();
    if (is_reassociated()) {
        result.reassociated_layout = reassociated_layout();
    }
    result.mapped = is_mapped();
    result.writable = writable();
    result.path = path();
    return result;
}

void FullTensor::print(
    std::ostream& output,
    const FullTensorPrintOptions& options) const
{
    if (options.precision < 0) {
        throw std::invalid_argument(
            "full-tensor print precision must be nonnegative");
    }

    const StreamFormatGuard format_guard(output);
    if (options.include_metadata) {
        output << "FullTensor shape=["
               << logical_shape_[0] << ',' << logical_shape_[1] << ','
               << logical_shape_[2] << ',' << logical_shape_[3] << ','
               << logical_shape_[4] << "] axes=(i1,i2,i3,i4,i5)\n"
               << "           storage=" << storage()
               << " logical=" << logical_size_
               << " stored=" << size() << '\n';
    }

    output << std::scientific << std::setprecision(options.precision);
    const size_type printed = std::min(logical_size_, options.max_values);
    for (size_type linear = 0; linear < printed; ++linear) {
        shape_type indices{};
        size_type remainder = linear;
        for (std::size_t axis = 0; axis < rank; ++axis) {
            indices[axis] = remainder % logical_shape_[axis];
            remainder /= logical_shape_[axis];
        }
        output << "A[" << indices[0] << ',' << indices[1] << ','
               << indices[2] << ',' << indices[3] << ',' << indices[4]
               << "] = " << (*this)(indices) << '\n';
    }

    if (printed < logical_size_) {
        output << "... " << (logical_size_ - printed)
               << " logical values omitted (increase max_values or use "
                  "FullTensorPrintOptions::all_values)\n";
    }
}

void FullTensor::print(const FullTensorPrintOptions& options) const
{
    print(std::cout, options);
}

void FullTensor::set_layout(TensorLayout layout)
{
    const TensorLayout expected = is_dense()
        ? TensorLayout::CanonicalDenseColumnMajor
        : TensorLayout::FullTensorReassociatedReflectionCompactByI1;
    if (layout != expected) {
        throw std::invalid_argument(
            "cannot change a FullTensor to an incompatible physical layout");
    }
    if (is_dense()) {
        dense_tensor().set_layout(layout);
    } else {
        reassociated_tensor().set_layout(layout);
    }
}

void FullTensor::set_tag(std::span<const std::uint8_t> new_tag)
{
    if (new_tag.size() > tensor_tag_bytes) {
        throw std::length_error("full-tensor tag exceeds 1024 bytes");
    }
    tag_type candidate{};
    std::copy(new_tag.begin(), new_tag.end(), candidate.begin());
    if (is_dense()) {
        validate_dense_tag(candidate, logical_shape_, size());
        dense_tensor().set_tag(candidate);
    } else {
        validate_reassociated_tag(
            candidate, logical_shape_, size(), reassociated_layout());
        reassociated_tensor().set_tag(candidate);
    }
}

void FullTensor::set_tag(const tag_type& new_tag)
{
    set_tag(std::span<const std::uint8_t>(new_tag));
}

void FullTensor::swap(FullTensor& other) noexcept
{
    using std::swap;
    swap(storage_, other.storage_);
    swap(logical_shape_, other.logical_shape_);
    swap(logical_strides_, other.logical_strides_);
    swap(logical_size_, other.logical_size_);
    swap(triangular_panel_size_, other.triangular_panel_size_);
    swap(representation_, other.representation_);
}

void FullTensor::initialize_logical_metadata(const shape_type& shape)
{
    logical_shape_ = shape;
    logical_size_ = checked_logical_size(shape);
    std::size_t stride = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
        logical_strides_[axis] = stride;
        stride = checked_multiply(
            stride,
            logical_shape_[axis],
            "full-tensor logical strides overflow size_t");
    }
    triangular_panel_size_ = 0;
    if (representation_ == Representation::reflection_triangular_by_i1) {
        const std::size_t rows = checked_multiply(
            logical_shape_[1],
            logical_shape_[2],
            "full-tensor reassociated row extent overflows size_t");
        triangular_panel_size_ = checked_triangular_size(rows);
    }
}

void FullTensor::validate_representation() const
{
    if (is_dense()) {
        if (dense_tensor().shape() != logical_shape_ ||
            dense_tensor().size() != logical_size_ ||
            dense_tensor().layout() !=
                TensorLayout::CanonicalDenseColumnMajor) {
            throw Error(
                ErrorCode::internal_failure,
                "dense FullTensor representation is internally inconsistent");
        }
        return;
    }

    if (reassociated_tensor().layout() !=
        TensorLayout::FullTensorReassociatedReflectionCompactByI1) {
        throw Error(
            ErrorCode::internal_failure,
            "reassociated FullTensor representation has the wrong layout code");
    }

    std::size_t expected = logical_size_;
    if (representation_ == Representation::reflection_triangular_by_i1) {
        if (logical_shape_[1] != logical_shape_[4] ||
            logical_shape_[2] != logical_shape_[3]) {
            throw Error(
                ErrorCode::internal_failure,
                "reflection-compressed full tensor has incompatible canonical dimensions");
        }
        expected = checked_multiply(
            logical_shape_[0],
            triangular_panel_size_,
            "reflection-compressed full-tensor payload overflows size_t");
    }
    if (reassociated_tensor().size() != expected) {
        throw Error(
            ErrorCode::internal_failure,
            "reassociated full-tensor payload count does not match its canonical dimensions");
    }
}

void FullTensor::check_indices(const shape_type& indices) const
{
    for (std::size_t axis = 0; axis < rank; ++axis) {
        if (indices[axis] >= logical_shape_[axis]) {
            throw std::out_of_range("full-tensor index is out of range");
        }
    }
}

FullTensor::size_type FullTensor::physical_index(
    const shape_type& indices) const noexcept
{
    if (representation_ == Representation::dense) {
        return indices[0] + logical_shape_[0] *
            (indices[1] + logical_shape_[1] *
             (indices[2] + logical_shape_[2] *
              (indices[3] + logical_shape_[3] * indices[4])));
    }

    const std::size_t rows = logical_shape_[1] * logical_shape_[2];
    const std::size_t row = indices[1] + logical_shape_[1] * indices[2];
    if (representation_ == Representation::rectangular_by_i1) {
        const std::size_t columns = logical_shape_[3] * logical_shape_[4];
        const std::size_t column =
            indices[3] + logical_shape_[3] * indices[4];
        return row + rows * (column + columns * indices[0]);
    }

    const std::size_t reflected_column =
        indices[4] + logical_shape_[1] * indices[3];
    const std::size_t low = std::min(row, reflected_column);
    const std::size_t high = std::max(row, reflected_column);
    const std::size_t triangular = high * (high + 1u) / 2u + low;
    return triangular + triangular_panel_size_ * indices[0];
}

FullTensor::dense_tensor_type& FullTensor::dense_tensor()
{
    return std::get<dense_tensor_type>(storage_);
}

const FullTensor::dense_tensor_type& FullTensor::dense_tensor() const
{
    return std::get<dense_tensor_type>(storage_);
}

FullTensor::reassociated_tensor_type& FullTensor::reassociated_tensor()
{
    return std::get<reassociated_tensor_type>(storage_);
}

const FullTensor::reassociated_tensor_type&
FullTensor::reassociated_tensor() const
{
    return std::get<reassociated_tensor_type>(storage_);
}

} // namespace sl2cfoamxx
