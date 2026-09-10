// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Internal compatibility layer. The implementation now lives in the public,
// namespaced header; historical kernels can keep their concise unqualified
// Tensor<T, Rank> spelling while they are migrated incrementally.
#include "sl2cfoamxx/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

using sl2cfoamxx::Tensor;
using sl2cfoamxx::TensorAccess;
using sl2cfoamxx::TensorInit;
using sl2cfoamxx::TensorLayout;
using sl2cfoamxx::TensorStorage;
using sl2cfoamxx::SimplicityConstraint;
using sl2cfoamxx::TensorZeroPolicy;

inline constexpr std::size_t NUM_KEYS_MAX = sl2cfoamxx::tensor_max_rank;
inline constexpr std::size_t TAG_BYTES = sl2cfoamxx::tensor_tag_bytes;

namespace tensor_disk = sl2cfoamxx::tensor_disk;

// Metadata stored in the generic tensor tag for full-tensor cache files.
struct FullTensorBinaryTagV1 {
    char magic[16]{};
    std::uint32_t version = 1;
    std::uint32_t layout_code = 0;
    std::uint64_t Dl = 0;
    double gamma = 0.0;
    std::int32_t spins[10]{};
    std::uint64_t canonical_dims[5]{};
    std::uint64_t values_count = 0;
    std::uint64_t reserved[32]{};
};

static_assert(std::is_trivially_copyable_v<FullTensorBinaryTagV1>);
static_assert(sizeof(FullTensorBinaryTagV1) <= TAG_BYTES);

// FullTensorBinaryTagV1 reserves slot zero for the reassociated storage order.
// Slot one records the simplicity prescription.  Zero intentionally means
// gamma*j so tags written before this field was introduced remain valid.
inline constexpr std::size_t full_tensor_storage_order_tag_slot = 0;
inline constexpr std::size_t full_tensor_simplicity_constraint_tag_slot = 1;

inline constexpr std::uint64_t full_tensor_simplicity_constraint_code(
    SimplicityConstraint prescription) noexcept
{
    switch (prescription) {
    case SimplicityConstraint::gamma_j:
        return 0;
    case SimplicityConstraint::gamma_j_p1:
        return 1;
    }
    return static_cast<std::uint64_t>(-1);
}

inline bool read_full_tensor_simplicity_constraint(
    const FullTensorBinaryTagV1& tag,
    SimplicityConstraint& output) noexcept
{
    switch (tag.reserved[full_tensor_simplicity_constraint_tag_slot]) {
    case 0:
        output = SimplicityConstraint::gamma_j;
        return true;
    case 1:
        output = SimplicityConstraint::gamma_j_p1;
        return true;
    default:
        return false;
    }
}

inline bool full_tensor_simplicity_constraint_matches(
    const FullTensorBinaryTagV1& tag,
    SimplicityConstraint expected) noexcept
{
    const std::uint64_t expected_code =
        full_tensor_simplicity_constraint_code(expected);
    return expected_code != static_cast<std::uint64_t>(-1) &&
           tag.reserved[full_tensor_simplicity_constraint_tag_slot] ==
               expected_code;
}

inline void set_full_tensor_simplicity_constraint(
    FullTensorBinaryTagV1& tag,
    SimplicityConstraint prescription) noexcept
{
    tag.reserved[full_tensor_simplicity_constraint_tag_slot] =
        full_tensor_simplicity_constraint_code(prescription);
}

inline std::array<std::uint8_t, TAG_BYTES> make_full_tensor_binary_tag(
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma,
    const std::array<std::size_t, 5>& canonical_dims,
    std::uint64_t values_count,
    tensor_disk::LayoutCode layout_code,
    SimplicityConstraint simplicity = SimplicityConstraint::gamma_j)
{
    FullTensorBinaryTagV1 tag{};
    constexpr char magic[16]{
        'S','L','2','C','F','T','T','A','G','\0','\0','\0','\0','\0','\0','\0'};
    std::memcpy(tag.magic, magic, sizeof(tag.magic));
    tag.layout_code = static_cast<std::uint32_t>(layout_code);
    tag.Dl = static_cast<std::uint64_t>(Dl);
    tag.gamma = gamma;
    for (std::size_t i = 0; i < spins.size(); ++i) {
        tag.spins[i] = static_cast<std::int32_t>(spins[i]);
    }
    for (std::size_t i = 0; i < canonical_dims.size(); ++i) {
        tag.canonical_dims[i] = static_cast<std::uint64_t>(canonical_dims[i]);
    }
    tag.values_count = values_count;
    set_full_tensor_simplicity_constraint(tag, simplicity);

    std::array<std::uint8_t, TAG_BYTES> bytes{};
    std::memcpy(bytes.data(), &tag, sizeof(tag));
    return bytes;
}

inline std::array<std::uint8_t, TAG_BYTES>
make_full_tensor_binary_tag_reassociated(
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma,
    const std::array<std::size_t, 5>& canonical_dims,
    std::uint64_t values_count,
    SimplicityConstraint simplicity = SimplicityConstraint::gamma_j)
{
    return make_full_tensor_binary_tag(
        spins, Dl, gamma, canonical_dims, values_count,
        tensor_disk::LayoutCode::FullTensorReassociatedReflectionCompactByI1,
        simplicity);
}

inline std::array<std::uint8_t, TAG_BYTES>
make_full_tensor_binary_tag_canonical(
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma,
    const std::array<std::size_t, 5>& canonical_dims,
    std::uint64_t values_count,
    SimplicityConstraint simplicity = SimplicityConstraint::gamma_j)
{
    return make_full_tensor_binary_tag(
        spins, Dl, gamma, canonical_dims, values_count,
        tensor_disk::LayoutCode::CanonicalDenseColumnMajor,
        simplicity);
}

inline bool read_full_tensor_binary_tag(
    const std::array<std::uint8_t, TAG_BYTES>& bytes,
    FullTensorBinaryTagV1& output) noexcept
{
    std::memcpy(&output, bytes.data(), sizeof(output));
    constexpr char magic[16]{
        'S','L','2','C','F','T','T','A','G','\0','\0','\0','\0','\0','\0','\0'};
    return std::memcmp(output.magic, magic, sizeof(output.magic)) == 0 &&
           output.version == 1;
}

inline bool read_full_tensor_binary_tag(
    const tensor_disk::TensorTreeHeaderV1& header,
    FullTensorBinaryTagV1& output) noexcept
{
    std::array<std::uint8_t, TAG_BYTES> bytes{};
    std::memcpy(bytes.data(), header.tag, bytes.size());
    return read_full_tensor_binary_tag(bytes, output);
}

enum class FullTensorReassociatedStorageOrder : std::uint64_t {
    Unknown = 0,
    ReflectionTriangularByI1 = 1,
    RectangularByI1 = 2
};

inline FullTensorReassociatedStorageOrder
full_tensor_reassociated_storage_order(
    const FullTensorBinaryTagV1& tag) noexcept
{
    switch (tag.reserved[full_tensor_storage_order_tag_slot]) {
    case 1:
        return FullTensorReassociatedStorageOrder::ReflectionTriangularByI1;
    case 2:
        return FullTensorReassociatedStorageOrder::RectangularByI1;
    default:
        return FullTensorReassociatedStorageOrder::Unknown;
    }
}

inline void set_full_tensor_reassociated_storage_order(
    FullTensorBinaryTagV1& tag,
    FullTensorReassociatedStorageOrder order) noexcept
{
    tag.reserved[full_tensor_storage_order_tag_slot] =
        static_cast<std::uint64_t>(order);
}

inline std::array<std::uint8_t, TAG_BYTES>
serialize_full_tensor_binary_tag(const FullTensorBinaryTagV1& tag) noexcept
{
    std::array<std::uint8_t, TAG_BYTES> bytes{};
    std::memcpy(bytes.data(), &tag, sizeof(tag));
    return bytes;
}

inline std::array<std::uint8_t, TAG_BYTES>
make_full_tensor_binary_tag_reassociated_with_storage_order(
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma,
    const std::array<std::size_t, 5>& canonical_dims,
    std::uint64_t values_count,
    FullTensorReassociatedStorageOrder storage_order,
    SimplicityConstraint simplicity = SimplicityConstraint::gamma_j)
{
    FullTensorBinaryTagV1 tag{};
    const auto bytes = make_full_tensor_binary_tag_reassociated(
        spins, Dl, gamma, canonical_dims, values_count, simplicity);
    (void)read_full_tensor_binary_tag(bytes, tag);
    set_full_tensor_reassociated_storage_order(tag, storage_order);
    return serialize_full_tensor_binary_tag(tag);
}
