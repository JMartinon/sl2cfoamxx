// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "full_tensor/FullTensorPackedContainers.hpp"
#include "core/tensors.hpp"

namespace b4support {

// ============================================================================
// Packed full-tensor booster tensors
// ============================================================================
//
// Canonical dense layouts produced by fill_rank5_tensors are:
//
//   B2(l23,l24,l25,i2,k2)
//   B3(l34,l35,l23,i3,k3)
//   B4(l45,l24,l34,i4,k4)
//   B5(l25,l35,l45,i5,k5)
//
// Packed layouts used for contraction are support-range layouts:
//
//   B2: block (i2,l23,l24,l25) -> k2
//   B3: block (i3,k3,l34,l35)  -> l23
//   B4: block (i4,k4,l24,l34)  -> l45
//   B5: block (i5,l25,l35,l45) -> k5
// ============================================================================

// A fully streamed full-tensor calculation must not regenerate all four
// booster legs for every point in the Cartesian product of boundary intertwiner panels.  A
// booster leg only depends on its matching boundary intertwiner axis, so expose a leg-local
// producer which can be hoisted to the corresponding loop level.  The dense
// tensor below is deliberately only one leg/panel wide; it is packed and
// released before another leg is built, keeping the peak bounded by the
// largest individual leg rather than by B2+B3+B4+B5.
enum class FullTensorBoosterLeg : std::uint8_t {
    B2 = 2,
    B3 = 3,
    B4 = 4,
    B5 = 5
};

inline std::size_t flat_full_B2_canonical(
    std::size_t il23,
    std::size_t il24,
    std::size_t il25,
    std::size_t ii2,
    std::size_t ik2,
    std::size_t L23,
    std::size_t L24,
    std::size_t L25,
    std::size_t I2) noexcept
{
    return il23 + L23 * (il24 + L24 * (il25 + L25 * (ii2 + I2 * ik2)));
}

inline std::size_t flat_full_B3_canonical(
    std::size_t il34,
    std::size_t il35,
    std::size_t il23,
    std::size_t ii3,
    std::size_t ik3,
    std::size_t L34,
    std::size_t L35,
    std::size_t L23,
    std::size_t I3) noexcept
{
    return il34 + L34 * (il35 + L35 * (il23 + L23 * (ii3 + I3 * ik3)));
}

inline std::size_t flat_full_B4_canonical(
    std::size_t il45,
    std::size_t il24,
    std::size_t il34,
    std::size_t ii4,
    std::size_t ik4,
    std::size_t L45,
    std::size_t L24,
    std::size_t L34,
    std::size_t I4) noexcept
{
    return il45 + L45 * (il24 + L24 * (il34 + L34 * (ii4 + I4 * ik4)));
}

inline std::size_t flat_full_B5_canonical(
    std::size_t il25,
    std::size_t il35,
    std::size_t il45,
    std::size_t ii5,
    std::size_t ik5,
    std::size_t L25,
    std::size_t L35,
    std::size_t L45,
    std::size_t I5) noexcept
{
    return il25 + L25 * (il35 + L35 * (il45 + L45 * (ii5 + I5 * ik5)));
}

void pack_full_B2_from_canonical(
    const FullTensorInputSupportPlan& support,
    const Tensor<double, 5>& dense_B2,
    std::vector<double>& packed_B2);

void pack_full_B3_from_canonical(
    const FullTensorInputSupportPlan& support,
    const Tensor<double, 5>& dense_B3,
    std::vector<double>& packed_B3);

void pack_full_B4_from_canonical(
    const FullTensorInputSupportPlan& support,
    const Tensor<double, 5>& dense_B4,
    std::vector<double>& packed_B4);

void pack_full_B5_from_canonical(
    const FullTensorInputSupportPlan& support,
    const Tensor<double, 5>& dense_B5,
    std::vector<double>& packed_B5);

void fill_packed_full_tensor_boosters_via_canonical(
    const B4Spins& spins,
    const std::array<std::size_t, 6>& Dim_ls,
    double gamma,
    const std::vector<double>& FactTable,
    const FullTensorInputSupportPlan& support,
    PackedFullTensorBoosters& packed,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0);

void fill_packed_full_tensor_booster_leg_via_canonical(
    const B4Spins& spins,
    double gamma,
    const std::vector<double>& FactTable,
    const FullTensorInputSupportPlan& support,
    FullTensorBoosterLeg leg,
    PackedFullTensorBoosters& packed,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0);

inline void assemble_packed_full_tensor_booster_panels(
    const PackedFullTensorBoosters& B2_panel,
    const PackedFullTensorBoosters& B3_panel,
    const PackedFullTensorBoosters& B4_panel,
    const PackedFullTensorBoosters& B5_panel,
    PackedFullTensorBoosters& out)
{
    out.clear();
    out.B2_alias = B2_panel.B2_data();
    out.B3_alias = B3_panel.B3_data();
    out.B4_alias = B4_panel.B4_data();
    out.B5_alias = B5_panel.B5_data();
}

} // namespace b4support
