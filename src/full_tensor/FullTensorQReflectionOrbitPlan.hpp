// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "full_tensor/FullTensorInputSupport_HalfSjC.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace b4support {

// The full tensor reflection keeps (x,l34,l25,i1) fixed and exchanges
//
//   (l24,k3,i2,i3) <-> (l35,k4,i5,i4).
//
// Half-sjC storage already stores one representative of the l24/l35 pair,
// but the old contraction schedule still emitted both logical q blocks.  This
// plan is the backend-neutral proof that it is safe to execute one q member
// and account for its partner through a transposed output update.
enum class FullTensorQOrbitKind : std::uint8_t {
    Unpaired = 0,
    SelfReflected = 1,
    Paired = 2
};

// The two algebraically equivalent central associations are
//
//   left : L = C^T U;  Z = L^T V       (aggregate virtual intertwiner axis k4)
//   right: R = C V;    Z = U^T R       (aggregate virtual intertwiner axis k3).
//
// A consumer may override this hint, but following it minimizes the exact
// dense tail flop count for the support rectangle.
enum class FullTensorQTailAssociation : std::uint8_t {
    CTransposeTimesLeft = 0,
    CTimesRight = 1
};

struct FullTensorQOrbitMember {
    std::size_t logical_block = 0;
    std::uint32_t ix = 0;
    std::uint32_t il35 = 0;
    std::uint32_t il24 = 0;
    std::uint32_t il34 = 0;
    Range2D logical{};
    Range2D stored{};
    std::uint32_t storage_block = 0;
    bool transpose_from_storage = false;

    [[nodiscard]] std::uint64_t nk3() const noexcept {
        return logical.i_end - logical.i_begin;
    }

    [[nodiscard]] std::uint64_t nk4() const noexcept {
        return logical.j_end - logical.j_begin;
    }
};

struct FullTensorQOrbitDescriptor {
    // For a paired orbit the representative owns the physical half-sjC block,
    // avoiding a normalization transpose before the central contraction.
    FullTensorQOrbitMember representative{};
    FullTensorQOrbitMember reflected{};
    FullTensorQOrbitKind kind = FullTensorQOrbitKind::Unpaired;
    FullTensorQTailAssociation tail_association =
        FullTensorQTailAssociation::CTransposeTimesLeft;

    [[nodiscard]] bool has_distinct_partner() const noexcept {
        return kind == FullTensorQOrbitKind::Paired;
    }

    [[nodiscard]] std::uint64_t selected_virtual_extent() const noexcept {
        return tail_association ==
                FullTensorQTailAssociation::CTransposeTimesLeft
            ? representative.nk4()
            : representative.nk3();
    }
};

struct FullTensorQReflectionOrbitPlan {
    bool reflection_compatible = false;
    bool canonical_orbits_enabled = false;
    std::string fallback_reason;

    std::vector<FullTensorQOrbitDescriptor> orbits;

    // Logical block -> orbit ordinal. Empty logical blocks retain `invalid`.
    std::vector<std::size_t> logical_to_orbit;
    // Exact logical q involution. Empty blocks retain `invalid`; self or
    // unpaired fallback blocks map to themselves.
    std::vector<std::size_t> logical_partner;

    // Descriptors are grouped by x. Within every x range, paired (non-self)
    // representatives precede self-reflected representatives; the boundary is
    // retained for diagnostics even when a consumer mixes both kinds by
    // half-scaling self rows before a uniform reflection epilogue.
    std::vector<std::size_t> x_offsets;
    std::vector<std::size_t> x_paired_end;

    std::uint64_t logical_nonempty_q_blocks = 0;
    std::uint64_t canonical_q_orbits = 0;
    std::uint64_t paired_q_orbits = 0;
    std::uint64_t self_reflected_q_orbits = 0;
    std::uint64_t unpaired_q_blocks = 0;

    // These counts describe rows supplied to the final dense update.  The
    // logical value is the old always-C^T-left schedule.  The canonical value
    // uses one q member and the cheaper central association.
    std::uint64_t logical_aggregate_rows = 0;
    std::uint64_t canonical_aggregate_rows = 0;

    static constexpr std::size_t invalid_orbit() noexcept {
        return std::numeric_limits<std::size_t>::max();
    }

    [[nodiscard]] double q_block_fraction() const noexcept {
        return logical_nonempty_q_blocks != 0
            ? static_cast<double>(canonical_q_orbits) /
                  static_cast<double>(logical_nonempty_q_blocks)
            : 0.0;
    }

    [[nodiscard]] double aggregate_row_fraction() const noexcept {
        return logical_aggregate_rows != 0
            ? static_cast<double>(canonical_aggregate_rows) /
                  static_cast<double>(logical_aggregate_rows)
            : 0.0;
    }
};

namespace full_tensor_q_orbit_detail {

bool make_value_bijection(const std::vector<int>& from,
                                 const std::vector<int>& to,
                                 std::vector<std::size_t>& map);

bool mapped_range_is_contiguous_transpose(
    const FullTensorInputSupportPlan& support,
    const Range2D& a,
    const Range2D& b) noexcept;

FullTensorQOrbitMember make_member(
    const FullTensorInputSupportPlan& support,
    std::size_t ix,
    std::size_t il35,
    std::size_t il24,
    std::size_t il34);

FullTensorQTailAssociation select_tail_association(
    const FullTensorQOrbitMember& q,
    std::size_t L25,
    std::size_t M,
    std::size_t N) noexcept;

void build_logical_fallback(FullTensorQReflectionOrbitPlan& plan,
                                   const FullTensorInputSupportPlan& support,
                                   const std::string& reason);

} // namespace full_tensor_q_orbit_detail

FullTensorQReflectionOrbitPlan
build_full_tensor_q_reflection_orbit_plan(
    const FullTensorInputSupportPlan& support);

} // namespace b4support
