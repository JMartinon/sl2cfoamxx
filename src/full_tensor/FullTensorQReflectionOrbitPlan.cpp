// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "full_tensor/FullTensorQReflectionOrbitPlan.hpp"

namespace b4support {


namespace full_tensor_q_orbit_detail {

bool make_value_bijection(const std::vector<int>& from,
                                 const std::vector<int>& to,
                                 std::vector<std::size_t>& map)
{
    if (from.size() != to.size()) return false;
    map.assign(from.size(), 0);
    std::vector<bool> used(to.size(), false);
    for (std::size_t i = 0; i < from.size(); ++i) {
        const auto it = std::find(to.begin(), to.end(), from[i]);
        if (it == to.end()) return false;
        const std::size_t j = static_cast<std::size_t>(it - to.begin());
        if (used[j]) return false;
        used[j] = true;
        map[i] = j;
    }
    return true;
}

} // namespace full_tensor_q_orbit_detail

namespace full_tensor_q_orbit_detail {

bool mapped_range_is_contiguous_transpose(
    const FullTensorInputSupportPlan& support,
    const Range2D& a,
    const Range2D& b) noexcept
{
    if (a.i_end - a.i_begin != b.j_end - b.j_begin ||
        a.j_end - a.j_begin != b.i_end - b.i_begin) {
        return false;
    }
    for (std::uint32_t i = a.i_begin; i < a.i_end; ++i) {
        if (i >= support.sjC_k3_to_k4.size()) return false;
        const int mapped = support.sjC_k3_to_k4[i];
        const std::uint32_t expected =
            b.j_begin + static_cast<std::uint32_t>(i - a.i_begin);
        if (mapped < 0 || static_cast<std::uint32_t>(mapped) != expected) {
            return false;
        }
    }
    for (std::uint32_t j = a.j_begin; j < a.j_end; ++j) {
        if (j >= support.sjC_k4_to_k3.size()) return false;
        const int mapped = support.sjC_k4_to_k3[j];
        const std::uint32_t expected =
            b.i_begin + static_cast<std::uint32_t>(j - a.j_begin);
        if (mapped < 0 || static_cast<std::uint32_t>(mapped) != expected) {
            return false;
        }
    }
    return true;
}

} // namespace full_tensor_q_orbit_detail

namespace full_tensor_q_orbit_detail {

FullTensorQOrbitMember make_member(
    const FullTensorInputSupportPlan& support,
    std::size_t ix,
    std::size_t il35,
    std::size_t il24,
    std::size_t il34)
{
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t block = idx4(
        ix, il24, il34, il35, L24, L34, L35);
    const FullTensorSjCHalfBlockMap& map = support.sjC_map[block];

    FullTensorQOrbitMember member;
    member.logical_block = block;
    member.ix = static_cast<std::uint32_t>(ix);
    member.il35 = static_cast<std::uint32_t>(il35);
    member.il24 = static_cast<std::uint32_t>(il24);
    member.il34 = static_cast<std::uint32_t>(il34);
    member.logical = map.logical;
    member.storage_block = map.storage_block;
    member.transpose_from_storage = map.transpose_from_storage;
    if (map.storage_block < support.sjC.blocks.size()) {
        member.stored = support.sjC.blocks[map.storage_block];
    }
    return member;
}

} // namespace full_tensor_q_orbit_detail

namespace full_tensor_q_orbit_detail {

FullTensorQTailAssociation select_tail_association(
    const FullTensorQOrbitMember& q,
    std::size_t L25,
    std::size_t M,
    std::size_t N) noexcept
{
    const long double nk3 = static_cast<long double>(q.nk3());
    const long double nk4 = static_cast<long double>(q.nk4());
    const long double l25 = static_cast<long double>(L25);
    const long double m = static_cast<long double>(M);
    const long double n = static_cast<long double>(N);
    const long double left = 2.0L * l25 * nk3 * nk4 * m
                           + 2.0L * l25 * nk4 * m * n;
    const long double right = 2.0L * l25 * nk3 * nk4 * n
                            + 2.0L * l25 * nk3 * m * n;
    return right < left
        ? FullTensorQTailAssociation::CTimesRight
        : FullTensorQTailAssociation::CTransposeTimesLeft;
}

} // namespace full_tensor_q_orbit_detail

namespace full_tensor_q_orbit_detail {

void build_logical_fallback(FullTensorQReflectionOrbitPlan& plan,
                                   const FullTensorInputSupportPlan& support,
                                   const std::string& reason)
{
    plan.reflection_compatible = false;
    plan.canonical_orbits_enabled = false;
    plan.fallback_reason = reason;
    plan.orbits.clear();
    std::fill(plan.logical_to_orbit.begin(), plan.logical_to_orbit.end(),
              FullTensorQReflectionOrbitPlan::invalid_orbit());
    std::fill(plan.logical_partner.begin(), plan.logical_partner.end(),
              FullTensorQReflectionOrbitPlan::invalid_orbit());
    plan.logical_nonempty_q_blocks = 0;
    plan.canonical_q_orbits = 0;
    plan.paired_q_orbits = 0;
    plan.self_reflected_q_orbits = 0;
    plan.unpaired_q_blocks = 0;
    plan.logical_aggregate_rows = 0;
    plan.canonical_aggregate_rows = 0;

    const std::size_t X = support.channels.xvals.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t M = support.channels.i2.size() *
                          support.channels.i3.size();
    const std::size_t N = support.channels.i4.size() *
                          support.channels.i5.size();
    plan.x_offsets.assign(X + 1, 0);
    plan.x_paired_end.assign(X, 0);
    for (std::size_t ix = 0; ix < X; ++ix) {
        plan.x_offsets[ix] = plan.orbits.size();
        plan.x_paired_end[ix] = plan.orbits.size();
        for (std::size_t il35 = 0; il35 < L35; ++il35) {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                for (std::size_t il34 = 0; il34 < L34; ++il34) {
                    FullTensorQOrbitMember q = make_member(
                        support, ix, il35, il24, il34);
                    if (q.logical.empty()) continue;
                    FullTensorQOrbitDescriptor orbit;
                    orbit.representative = q;
                    orbit.reflected = q;
                    orbit.kind = FullTensorQOrbitKind::Unpaired;
                    orbit.tail_association = select_tail_association(
                        q, L25, M, N);
                    const std::size_t ordinal = plan.orbits.size();
                    plan.orbits.push_back(orbit);
                    plan.logical_to_orbit[q.logical_block] = ordinal;
                    plan.logical_partner[q.logical_block] = q.logical_block;
                    ++plan.logical_nonempty_q_blocks;
                    ++plan.unpaired_q_blocks;
                    plan.logical_aggregate_rows += L25 * q.nk4();
                    plan.canonical_aggregate_rows +=
                        L25 * orbit.selected_virtual_extent();
                }
            }
        }
        plan.x_offsets[ix + 1] = plan.orbits.size();
    }
    plan.canonical_q_orbits = plan.orbits.size();
}

} // namespace full_tensor_q_orbit_detail

FullTensorQReflectionOrbitPlan
build_full_tensor_q_reflection_orbit_plan(
    const FullTensorInputSupportPlan& support)
{
    FullTensorQReflectionOrbitPlan plan;
    const std::size_t X = support.channels.xvals.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t logical_blocks = X * L24 * L34 * L35;
    plan.logical_to_orbit.assign(
        logical_blocks, FullTensorQReflectionOrbitPlan::invalid_orbit());
    plan.logical_partner.assign(
        logical_blocks, FullTensorQReflectionOrbitPlan::invalid_orbit());

    if (support.sjC_map.size() != logical_blocks) {
        full_tensor_q_orbit_detail::build_logical_fallback(
            plan, support, "sjC logical metadata shape mismatch");
        return plan;
    }

    std::vector<std::size_t> l24_to_l35, l35_to_l24;
    std::vector<std::size_t> ignored;
    const bool axes_compatible =
        support.reflection_15432_preserves_spins &&
        full_tensor_q_orbit_detail::make_value_bijection(
            support.l24, support.l35, l24_to_l35) &&
        full_tensor_q_orbit_detail::make_value_bijection(
            support.l35, support.l24, l35_to_l24) &&
        full_tensor_q_orbit_detail::make_value_bijection(
            support.l23, support.l45, ignored) &&
        full_tensor_q_orbit_detail::make_value_bijection(
            support.channels.i2, support.channels.i5, ignored) &&
        full_tensor_q_orbit_detail::make_value_bijection(
            support.channels.i3, support.channels.i4, ignored) &&
        full_tensor_q_orbit_detail::make_value_bijection(
            support.channels.k2, support.channels.k5, ignored) &&
        full_tensor_q_orbit_detail::make_value_bijection(
            support.channels.k3, support.channels.k4, ignored);
    if (!axes_compatible) {
        full_tensor_q_orbit_detail::build_logical_fallback(
            plan, support,
            "boundary spins or reflected channel/lifted-spin axes are incompatible");
        return plan;
    }

    // CUDA descriptors use uint32 axis indices. Rejecting impossible shapes
    // here is preferable to silently truncating an otherwise valid CPU plan.
    constexpr std::size_t umax = std::numeric_limits<std::uint32_t>::max();
    if (X > umax || L24 > umax || L34 > umax || L35 > umax) {
        full_tensor_q_orbit_detail::build_logical_fallback(
            plan, support, "q axes exceed uint32 descriptor capacity");
        return plan;
    }

    std::vector<bool> visited(logical_blocks, false);
    const std::size_t M = support.channels.i2.size() *
                          support.channels.i3.size();
    const std::size_t N = support.channels.i4.size() *
                          support.channels.i5.size();
    std::string validation_error;

    for (std::size_t ix = 0; ix < X && validation_error.empty(); ++ix) {
        for (std::size_t il35 = 0; il35 < L35 && validation_error.empty(); ++il35) {
            for (std::size_t il24 = 0; il24 < L24 && validation_error.empty(); ++il24) {
                for (std::size_t il34 = 0; il34 < L34; ++il34) {
                    const FullTensorQOrbitMember q =
                        full_tensor_q_orbit_detail::make_member(
                            support, ix, il35, il24, il34);
                    if (q.logical.empty() || visited[q.logical_block]) continue;

                    const std::size_t pil35 = l24_to_l35[il24];
                    const std::size_t pil24 = l35_to_l24[il35];
                    const FullTensorQOrbitMember p =
                        full_tensor_q_orbit_detail::make_member(
                            support, ix, pil35, pil24, il34);
                    if (p.logical.empty()) {
                        validation_error =
                            "a nonempty q block has an empty reflected partner";
                        break;
                    }

                    const bool self = q.logical_block == p.logical_block;
                    if (!full_tensor_q_orbit_detail::mapped_range_is_contiguous_transpose(
                            support, q.logical, p.logical) ||
                        !full_tensor_q_orbit_detail::mapped_range_is_contiguous_transpose(
                            support, p.logical, q.logical)) {
                        validation_error =
                            "reflected sjC virtual intertwiner supports are not contiguous transposes";
                        break;
                    }
                    if (q.storage_block != p.storage_block) {
                        validation_error =
                            "reflected q members do not share one sjC storage block";
                        break;
                    }
                    if (self) {
                        if (q.transpose_from_storage) {
                            validation_error =
                                "self-reflected q block is unexpectedly transposed";
                            break;
                        }
                    } else if (q.transpose_from_storage ==
                               p.transpose_from_storage) {
                        validation_error =
                            "paired q orbit has no unique sjC storage owner";
                        break;
                    }

                    FullTensorQOrbitDescriptor orbit;
                    orbit.kind = self ? FullTensorQOrbitKind::SelfReflected
                                      : FullTensorQOrbitKind::Paired;
                    if (!self && q.transpose_from_storage) {
                        orbit.representative = p;
                        orbit.reflected = q;
                    } else {
                        orbit.representative = q;
                        orbit.reflected = p;
                    }
                    orbit.tail_association =
                        full_tensor_q_orbit_detail::select_tail_association(
                            orbit.representative, L25, M, N);

                    const std::size_t ordinal = plan.orbits.size();
                    plan.orbits.push_back(orbit);
                    plan.logical_to_orbit[q.logical_block] = ordinal;
                    plan.logical_to_orbit[p.logical_block] = ordinal;
                    plan.logical_partner[q.logical_block] = p.logical_block;
                    plan.logical_partner[p.logical_block] = q.logical_block;
                    visited[q.logical_block] = true;
                    visited[p.logical_block] = true;

                    plan.logical_nonempty_q_blocks += self ? 1u : 2u;
                    plan.logical_aggregate_rows += L25 * q.nk4();
                    if (!self) plan.logical_aggregate_rows += L25 * p.nk4();
                    plan.canonical_aggregate_rows +=
                        L25 * orbit.selected_virtual_extent();
                    if (self) {
                        ++plan.self_reflected_q_orbits;
                    } else {
                        ++plan.paired_q_orbits;
                    }
                }
            }
        }
    }

    if (!validation_error.empty()) {
        full_tensor_q_orbit_detail::build_logical_fallback(
            plan, support, validation_error);
        return plan;
    }

    plan.reflection_compatible = true;
    plan.canonical_orbits_enabled = true;

    std::stable_sort(
        plan.orbits.begin(), plan.orbits.end(),
        [](const FullTensorQOrbitDescriptor& a,
           const FullTensorQOrbitDescriptor& b) {
            if (a.representative.ix != b.representative.ix) {
                return a.representative.ix < b.representative.ix;
            }
            const unsigned ag = a.kind == FullTensorQOrbitKind::Paired ? 0u : 1u;
            const unsigned bg = b.kind == FullTensorQOrbitKind::Paired ? 0u : 1u;
            if (ag != bg) return ag < bg;
            return a.representative.logical_block <
                   b.representative.logical_block;
        });
    std::fill(plan.logical_to_orbit.begin(), plan.logical_to_orbit.end(),
              FullTensorQReflectionOrbitPlan::invalid_orbit());
    plan.x_offsets.assign(X + 1, 0);
    plan.x_paired_end.assign(X, 0);
    std::size_t cursor = 0;
    for (std::size_t ix = 0; ix < X; ++ix) {
        plan.x_offsets[ix] = cursor;
        while (cursor < plan.orbits.size() &&
               plan.orbits[cursor].representative.ix == ix &&
               plan.orbits[cursor].kind == FullTensorQOrbitKind::Paired) {
            ++cursor;
        }
        plan.x_paired_end[ix] = cursor;
        while (cursor < plan.orbits.size() &&
               plan.orbits[cursor].representative.ix == ix) {
            ++cursor;
        }
        plan.x_offsets[ix + 1] = cursor;
    }
    for (std::size_t ordinal = 0; ordinal < plan.orbits.size(); ++ordinal) {
        const FullTensorQOrbitDescriptor& orbit = plan.orbits[ordinal];
        plan.logical_to_orbit[orbit.representative.logical_block] = ordinal;
        plan.logical_to_orbit[orbit.reflected.logical_block] = ordinal;
    }
    plan.canonical_q_orbits = plan.orbits.size();
    return plan;
}

} // namespace b4support
