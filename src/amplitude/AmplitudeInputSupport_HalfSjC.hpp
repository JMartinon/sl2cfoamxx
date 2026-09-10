// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "amplitude/AmplitudeInputSupport.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <utility>

namespace b4support {

// Metadata for the half-stored single-amplitude sjC tensor.
//
// Logical sjC blocks are still indexed as
//
//   block (x,l24,l34,l35) -> k3 x k4
//
// but the payload stores only one representative under
//
//   C(x,l24,l34,l35; k3,k4) = C(x,l35,l34,l24; k4,k3).
//
// For a non-stored logical block, storage_block points to the stored swapped
// block and transpose_from_storage=true means reads must swap k3 <-> k4.
struct AmplitudeSjCHalfBlockMap {
    Range2D logical = Range2D{};
    std::uint32_t storage_block = 0u;
    bool transpose_from_storage = false;

    bool empty() const noexcept {
        return logical.empty();
    }
};

inline int amplitude_half_sjC_max_value_in_vector(const std::vector<int>& v)
{
    if (v.empty()) {
        return 0;
    }
    return *std::max_element(v.begin(), v.end());
}

inline std::vector<int> amplitude_half_sjC_make_value_to_index_map(
    const std::vector<int>& values,
    int max_value)
{
    std::vector<int> map(static_cast<std::size_t>(std::max(0, max_value)) + 1u, -1);

    for (std::size_t i = 0; i < values.size(); ++i) {
        const int v = values[i];
        if (v >= 0 && v <= max_value) {
            map[static_cast<std::size_t>(v)] = static_cast<int>(i);
        }
    }

    return map;
}

// Single-amplitude support plan with half-stored sjC.
//
// This intentionally derives from AmplitudeInputSupportPlan so existing booster
// filling and q-local T1/T3 builders, which only need the non-sjC parts or the
// stored sjC range for canonical q blocks, can keep accepting it as a base ref.
struct AmplitudeInputSupportPlan_HalfSjC : public AmplitudeInputSupportPlan {
    // plan.sjC.blocks contains physical storage-owning blocks only.
    // sjC_map contains the logical view for every block.
    std::vector<AmplitudeSjCHalfBlockMap> sjC_map;

    // Axis-index maps used by half-aware readers.  Entries are -1 when the
    // corresponding doubled-spin value does not exist in the swapped axis.
    std::vector<int> sjC_k3_to_k4;
    std::vector<int> sjC_k4_to_k3;

    std::uint64_t total_metadata_bytes() const noexcept {
        return AmplitudeInputSupportPlan::total_metadata_bytes()
             + static_cast<std::uint64_t>(sjC_map.size() * sizeof(AmplitudeSjCHalfBlockMap))
             + static_cast<std::uint64_t>((sjC_k3_to_k4.size() + sjC_k4_to_k3.size()) * sizeof(int));
    }

    void print(std::ostream& os) const {
        os << "\nAmplitude input support plan, half-stored sjC\n";
        os << "===========================================\n\n";

        sjA.print(os); os << "\n";
        sjB.print(os); os << "\n";
        sjC.print(os);
        os << "  note          = half-stored; use sjC_map for logical reflected blocks\n\n";
        sjD.print(os); os << "\n";
        sjE.print(os); os << "\n";

        B2.print(os); os << "\n";
        B3.print(os); os << "\n";
        B4.print(os); os << "\n";
        B5.print(os); os << "\n";

        os << "Total packed payload = ";
        print_support_memory(os, static_cast<long double>(total_payload_bytes()));
        os << "\n";

        os << "Total metadata       = ";
        print_support_memory(os, static_cast<long double>(total_metadata_bytes()));
        os << "\n";
    }
};

inline bool amplitude_half_sjC_block_owns_storage(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    std::size_t logical_block) noexcept
{
    if (logical_block >= support.sjC_map.size() ||
        logical_block >= support.sjC.blocks.size())
    {
        return false;
    }

    const AmplitudeSjCHalfBlockMap& map = support.sjC_map[logical_block];
    return !map.empty()
        && !map.transpose_from_storage
        && static_cast<std::size_t>(map.storage_block) == logical_block
        && !support.sjC.blocks[logical_block].empty();
}

inline const Range2D& amplitude_half_sjC_logical_range(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    std::size_t logical_block) noexcept
{
    return support.sjC_map[logical_block].logical;
}

inline double packed_load_sjC_half(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const double* values,
    std::size_t logical_block,
    std::size_t ik3,
    std::size_t ik4)
{
    const AmplitudeSjCHalfBlockMap& map = support.sjC_map[logical_block];

#ifndef NDEBUG
    if (map.empty()) {
        std::cerr << "packed_load_sjC_half: empty logical block\n"
                  << "block = " << logical_block << "\n";
        std::abort();
    }
    if (!(ik3 >= map.logical.i_begin && ik3 < map.logical.i_end &&
          ik4 >= map.logical.j_begin && ik4 < map.logical.j_end))
    {
        std::cerr << "packed_load_sjC_half: logical index outside support\n"
                  << "block = " << logical_block
                  << ", ik3 = " << ik3
                  << ", ik4 = " << ik4 << "\n";
        std::abort();
    }
#endif

    std::size_t storage_ik3 = ik3;
    std::size_t storage_ik4 = ik4;

    if (map.transpose_from_storage) {
#ifndef NDEBUG
        if (ik3 >= support.sjC_k3_to_k4.size() ||
            ik4 >= support.sjC_k4_to_k3.size() ||
            support.sjC_k3_to_k4[ik3] < 0 ||
            support.sjC_k4_to_k3[ik4] < 0)
        {
            std::cerr << "packed_load_sjC_half: missing swapped b-axis map\n"
                      << "block = " << logical_block
                      << ", ik3 = " << ik3
                      << ", ik4 = " << ik4 << "\n";
            std::abort();
        }
#endif
        storage_ik3 = static_cast<std::size_t>(support.sjC_k4_to_k3[ik4]);
        storage_ik4 = static_cast<std::size_t>(support.sjC_k3_to_k4[ik3]);
    }

    const std::size_t storage_block = static_cast<std::size_t>(map.storage_block);
    const Range2D& r = support.sjC.blocks[storage_block];

#ifndef NDEBUG
    if (r.empty() ||
        !(storage_ik3 >= r.i_begin && storage_ik3 < r.i_end &&
          storage_ik4 >= r.j_begin && storage_ik4 < r.j_end))
    {
        std::cerr << "packed_load_sjC_half: storage index outside support\n"
                  << "logical block = " << logical_block
                  << ", storage block = " << storage_block
                  << ", storage ik3 = " << storage_ik3
                  << ", storage ik4 = " << storage_ik4 << "\n";
        std::abort();
    }
#endif

    const std::size_t ni = static_cast<std::size_t>(r.i_end - r.i_begin);
    const std::size_t local_i = storage_ik3 - r.i_begin;
    const std::size_t local_j = storage_ik4 - r.j_begin;

    return values[r.offset + local_i + ni * local_j];
}

AmplitudeInputSupportPlan_HalfSjC build_amplitude_input_support_plan_half_sjC(
    const B4Spins& s,
    int Dl,
    const AmplitudeChannels& AmC);


// ============================================================================
// Static / generated x-slab / q-slab support helpers
// ============================================================================
// These helpers avoid materializing global sjX support.  They intentionally
// reuse the tested one-x support builder for metadata generation: one x-slice of
// support metadata is cheap, while rewriting all triangle/range logic by hand is
// a fashionable way to manufacture wrong answers.

inline void amplitude_half_sjC_clear_physical_storage(AmplitudeInputSupportPlan_HalfSjC& plan)
{
    for (Range2D& r : plan.sjC.blocks) {
        r = empty_range_2d();
    }
    for (AmplitudeSjCHalfBlockMap& m : plan.sjC_map) {
        m.logical = empty_range_2d();
        m.storage_block = 0u;
        m.transpose_from_storage = false;
    }
    finalize_offsets(plan.sjC);
}


// Build the fixed-boundary intertwiner-shaped channel view needed by the existing packed
// coherent booster code from a full tensor channel object.
//
// This is intentionally *not* a single-amplitude channel: i2..i5 have already
// been summed into coherent B2..B5, and the support needed by those projected
// boosters is the full virtual intertwiner/ell layout.  The scalar `AmplitudeChannels::i`
// field is therefore only a harmless compatibility placeholder for older helper
// signatures that still carry the single-amplitude type around.  Using a user's
// `fixed_intertwiners` here would be conceptually wrong and can make the x support depend
// on a boundary intertwiner that has already been contracted away.  Very C++:
// the type is lying, so we make the lie explicit and boxed in.
inline AmplitudeChannels make_amplitude_channels_for_coherent_boosters(
    const FullTensorChannels& FTC)
{
    AmplitudeChannels ch;

    ch.i[0] = FTC.i1.empty() ? 0 : FTC.i1.front();
    ch.i[1] = FTC.i2.empty() ? 0 : FTC.i2.front();
    ch.i[2] = FTC.i3.empty() ? 0 : FTC.i3.front();
    ch.i[3] = FTC.i4.empty() ? 0 : FTC.i4.front();
    ch.i[4] = FTC.i5.empty() ? 0 : FTC.i5.front();

    ch.k2 = FTC.k2;
    ch.k3 = FTC.k3;
    ch.k4 = FTC.k4;
    ch.k5 = FTC.k5;
    ch.xvals = FTC.xvals;

    return ch;
}

// Static coherent-booster support from full tensor channels.
//
// This is the support builder that should be used before
// fill_packed_amplitude_coherent_boosters_via_canonical(...).  It builds only
// B2..B5 packed supports over the full virtual intertwiner/ell ranges and leaves sjX storage
// empty, because sjA..sjE are generated later by the shared coherent C1
// contraction from its own C1-dependent support.  In other words: this support
// describes the already-projected coherent booster tensors, not a fixed-boundary intertwiner
// single amplitude.
AmplitudeInputSupportPlan_HalfSjC build_amplitude_static_booster_support_plan_half_sjC(
    const B4Spins& s,
    int Dl,
    const FullTensorChannels& FTC);

AmplitudeInputSupportPlan_HalfSjC build_amplitude_static_booster_support_plan_half_sjC(
    const B4Spins& s,
    int Dl,
    const AmplitudeChannels& AmC);

inline int amplitude_half_sjC_infer_Dl_from_static_support(
    const AmplitudeInputSupportPlan_HalfSjC& static_support) noexcept
{
    if (!static_support.l23.empty()) {
        return static_cast<int>(static_support.l23.size()) - 1;
    }
    if (!static_support.l24.empty()) {
        return static_cast<int>(static_support.l24.size()) - 1;
    }
    return 0;
}

inline AmplitudeInputSupportPlan_HalfSjC build_amplitude_xslice_full_support_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const B4Spins& s,
    std::size_t ix_global)
{
    if (ix_global >= static_support.channels.xvals.size()) {
        throw std::out_of_range("build_amplitude_xslice_full_support_half_sjC: ix out of range");
    }

    AmplitudeChannels one_x = static_support.channels;
    one_x.xvals.assign(1u, static_support.channels.xvals[ix_global]);

    const int Dl = amplitude_half_sjC_infer_Dl_from_static_support(static_support);
    AmplitudeInputSupportPlan_HalfSjC out =
        build_amplitude_input_support_plan_half_sjC(s, Dl, one_x);

    // Force B supports to be the global/static ones so packed_B offsets remain
    // valid.  sj supports are one-x local.
    out.B2 = static_support.B2;
    out.B3 = static_support.B3;
    out.B4 = static_support.B4;
    out.B5 = static_support.B5;
    return out;
}


AmplitudeInputSupportPlan_HalfSjC build_amplitude_sj_xslab_support_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const B4Spins& s,
    std::size_t ix_begin,
    std::size_t ix_end);

inline std::size_t amplitude_q_count_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& xslice_full_support) noexcept
{
    return xslice_full_support.l24.size()
         * xslice_full_support.l34.size()
         * xslice_full_support.l35.size();
}

inline void amplitude_decode_q_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    std::size_t q,
    std::size_t& il24,
    std::size_t& il34,
    std::size_t& il35) noexcept
{
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    il35 = q % L35;
    q /= L35;
    il34 = q % L34;
    il24 = q / L34;
}

AmplitudeInputSupportPlan_HalfSjC make_amplitude_xslice_qslab_support_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& xslice_full_support,
    std::size_t q_begin,
    std::size_t q_end);


} // namespace b4support
