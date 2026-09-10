// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/AmplitudeInputSupport_HalfSjC.hpp"

namespace b4support {

AmplitudeInputSupportPlan_HalfSjC build_amplitude_input_support_plan_half_sjC(
    const B4Spins& s,
    int Dl,
    const AmplitudeChannels& AmC)
{
    AmplitudeInputSupportPlan_HalfSjC plan;

    // Reuse the tested full logical support builder, then compact only sjC.
    // Support construction is already tiny; duplicating that logic here would
    // mostly create a new habitat for bugs.
    static_cast<AmplitudeInputSupportPlan&>(plan) =
        build_amplitude_input_support_plan(s, Dl, AmC);

    const std::vector<int>& k3 = plan.channels.k3;
    const std::vector<int>& k4 = plan.channels.k4;
    const std::vector<int>& l24 = plan.l24;
    const std::vector<int>& l35 = plan.l35;

    const std::size_t X   = plan.channels.xvals.size();
    const std::size_t K3d = k3.size();
    const std::size_t K4d = k4.size();
    const std::size_t L24 = l24.size();
    const std::size_t L34 = plan.l34.size();
    const std::size_t L35 = l35.size();

    plan.sjC.name = "sjC support: half-stored block (x,l24,l34,l35) -> k3 x k4";
    plan.sjC_map.assign(plan.sjC.blocks.size(), AmplitudeSjCHalfBlockMap{});
    plan.sjC_k3_to_k4.assign(K3d, -1);
    plan.sjC_k4_to_k3.assign(K4d, -1);

    int max_map_value = 0;
    max_map_value = std::max(max_map_value, amplitude_half_sjC_max_value_in_vector(k3));
    max_map_value = std::max(max_map_value, amplitude_half_sjC_max_value_in_vector(k4));
    max_map_value = std::max(max_map_value, amplitude_half_sjC_max_value_in_vector(l24));
    max_map_value = std::max(max_map_value, amplitude_half_sjC_max_value_in_vector(l35));

    const std::vector<int> val_to_ik3 =
        amplitude_half_sjC_make_value_to_index_map(k3, max_map_value);
    const std::vector<int> val_to_ik4 =
        amplitude_half_sjC_make_value_to_index_map(k4, max_map_value);
    const std::vector<int> val_to_il24 =
        amplitude_half_sjC_make_value_to_index_map(l24, max_map_value);
    const std::vector<int> val_to_il35 =
        amplitude_half_sjC_make_value_to_index_map(l35, max_map_value);

    for (std::size_t ik3 = 0; ik3 < K3d; ++ik3) {
        const int v = k3[ik3];
        if (v >= 0 && v <= max_map_value) {
            plan.sjC_k3_to_k4[ik3] = val_to_ik4[static_cast<std::size_t>(v)];
        }
    }

    for (std::size_t ik4 = 0; ik4 < K4d; ++ik4) {
        const int v = k4[ik4];
        if (v >= 0 && v <= max_map_value) {
            plan.sjC_k4_to_k3[ik4] = val_to_ik3[static_cast<std::size_t>(v)];
        }
    }

    if (X == 0 || K3d == 0 || K4d == 0 || L24 == 0 || L34 == 0 || L35 == 0) {
        finalize_offsets(plan.sjC);
        return plan;
    }

#ifdef _OPENMP
#pragma omp parallel for collapse(4) schedule(static)
#endif
    for (std::ptrdiff_t ix_s = 0; ix_s < static_cast<std::ptrdiff_t>(X); ++ix_s) {
        for (std::ptrdiff_t il24_s = 0; il24_s < static_cast<std::ptrdiff_t>(L24); ++il24_s) {
            for (std::ptrdiff_t il34_s = 0; il34_s < static_cast<std::ptrdiff_t>(L34); ++il34_s) {
                for (std::ptrdiff_t il35_s = 0; il35_s < static_cast<std::ptrdiff_t>(L35); ++il35_s) {
                    const std::size_t ix   = static_cast<std::size_t>(ix_s);
                    const std::size_t il24 = static_cast<std::size_t>(il24_s);
                    const std::size_t il34 = static_cast<std::size_t>(il34_s);
                    const std::size_t il35 = static_cast<std::size_t>(il35_s);

                    const std::size_t block = idx4(ix, il24, il34, il35, L24, L34, L35);
                    const Range2D logical = plan.sjC.blocks[block];

                    AmplitudeSjCHalfBlockMap map;
                    map.logical = logical;
                    map.storage_block = static_cast<std::uint32_t>(block);
                    map.transpose_from_storage = false;

                    if (logical.empty()) {
                        plan.sjC_map[block] = map;
                        plan.sjC.blocks[block] = empty_range_2d();
                        continue;
                    }

                    const int Dl24 = l24[il24];
                    const int Dl35 = l35[il35];

                    const int sym_il24 = (Dl35 >= 0 && Dl35 <= max_map_value)
                        ? val_to_il24[static_cast<std::size_t>(Dl35)]
                        : -1;
                    const int sym_il35 = (Dl24 >= 0 && Dl24 <= max_map_value)
                        ? val_to_il35[static_cast<std::size_t>(Dl24)]
                        : -1;

                    bool virtual_ranges_map = true;
                    for (std::uint32_t ik3 = logical.i_begin; ik3 < logical.i_end; ++ik3) {
                        if (ik3 >= plan.sjC_k3_to_k4.size() || plan.sjC_k3_to_k4[ik3] < 0) {
                            virtual_ranges_map = false;
                            break;
                        }
                    }
                    if (virtual_ranges_map) {
                        for (std::uint32_t ik4 = logical.j_begin; ik4 < logical.j_end; ++ik4) {
                            if (ik4 >= plan.sjC_k4_to_k3.size() || plan.sjC_k4_to_k3[ik4] < 0) {
                                virtual_ranges_map = false;
                                break;
                            }
                        }
                    }

                    const bool can_use_swapped_block =
                        sym_il24 >= 0 && sym_il35 >= 0 && virtual_ranges_map;

                    bool store_here = true;
                    if (can_use_swapped_block && Dl24 > Dl35) {
                        store_here = false;
                        map.transpose_from_storage = true;
                        map.storage_block = static_cast<std::uint32_t>(
                            idx4(
                                ix,
                                static_cast<std::size_t>(sym_il24),
                                il34,
                                static_cast<std::size_t>(sym_il35),
                                L24,
                                L34,
                                L35
                            )
                        );
                    }

                    plan.sjC_map[block] = map;
                    plan.sjC.blocks[block] = store_here ? logical : empty_range_2d();
                }
            }
        }
    }

    // Recompute physical offsets after discarding reflected storage blocks.
    finalize_offsets(plan.sjC);

    return plan;
}

AmplitudeInputSupportPlan_HalfSjC build_amplitude_static_booster_support_plan_half_sjC(
    const B4Spins& s,
    int Dl,
    const FullTensorChannels& FTC)
{
    AmplitudeInputSupportPlan_HalfSjC out;
    out.channels = make_amplitude_channels_for_coherent_boosters(FTC);

    out.l23 = make_l_axis(s.Dj23, Dl);
    out.l24 = make_l_axis(s.Dj24, Dl);
    out.l25 = make_l_axis(s.Dj25, Dl);
    out.l34 = make_l_axis(s.Dj34, Dl);
    out.l35 = make_l_axis(s.Dj35, Dl);
    out.l45 = make_l_axis(s.Dj45, Dl);

    const std::size_t K2d = out.channels.k2.size();
    const std::size_t K3d = out.channels.k3.size();
    const std::size_t K4d = out.channels.k4.size();
    const std::size_t K5d = out.channels.k5.size();

    const std::size_t L23 = out.l23.size();
    const std::size_t L24 = out.l24.size();
    const std::size_t L25 = out.l25.size();
    const std::size_t L34 = out.l34.size();
    const std::size_t L35 = out.l35.size();
    const std::size_t L45 = out.l45.size();

    // sj supports intentionally empty in the static coherent-booster plan.
    out.sjA = make_support_1d("sjA static empty support", {0u, L25}, K5d);
    out.sjB = make_support_1d("sjB static empty support", {0u, K4d, L35, L45}, K5d);
    out.sjC = make_support_2d("sjC static empty support", {0u, L24, L34, L35}, K3d, K4d);
    out.sjD = make_support_1d("sjD static empty support", {0u, K3d, L23, L24}, K2d);
    out.sjE = make_support_1d("sjE static empty support", {0u, L25}, K2d);
    finalize_offsets(out.sjA);
    finalize_offsets(out.sjB);
    finalize_offsets(out.sjC);
    finalize_offsets(out.sjD);
    finalize_offsets(out.sjE);

    out.sjC_map.assign(out.sjC.blocks.size(), AmplitudeSjCHalfBlockMap{});

    int max_map_value = 0;
    max_map_value = std::max(max_map_value, amplitude_half_sjC_max_value_in_vector(out.channels.k3));
    max_map_value = std::max(max_map_value, amplitude_half_sjC_max_value_in_vector(out.channels.k4));
    out.sjC_k3_to_k4.assign(K3d, -1);
    out.sjC_k4_to_k3.assign(K4d, -1);
    const std::vector<int> val_to_ik3 = amplitude_half_sjC_make_value_to_index_map(out.channels.k3, max_map_value);
    const std::vector<int> val_to_ik4 = amplitude_half_sjC_make_value_to_index_map(out.channels.k4, max_map_value);
    for (std::size_t ik3 = 0; ik3 < K3d; ++ik3) {
        const int v = out.channels.k3[ik3];
        if (v >= 0 && v <= max_map_value) out.sjC_k3_to_k4[ik3] = val_to_ik4[static_cast<std::size_t>(v)];
    }
    for (std::size_t ik4 = 0; ik4 < K4d; ++ik4) {
        const int v = out.channels.k4[ik4];
        if (v >= 0 && v <= max_map_value) out.sjC_k4_to_k3[ik4] = val_to_ik3[static_cast<std::size_t>(v)];
    }

    out.B2 = make_support_1d("B2 support: block (l23,l24,l25) -> k2", {L23, L24, L25}, K2d);
    out.B3 = make_support_1d("B3 support: block (k3,l34,l35) -> l23", {K3d, L34, L35}, L23);
    out.B4 = make_support_1d("B4 support: block (k4,l24,l34) -> l45", {K4d, L24, L34}, L45);
    out.B5 = make_support_1d("B5 support: block (l25,l35,l45) -> k5", {L25, L35, L45}, K5d);

    if (K2d == 0 || K3d == 0 || K4d == 0 || K5d == 0) {
        finalize_offsets(out.B2);
        finalize_offsets(out.B3);
        finalize_offsets(out.B4);
        finalize_offsets(out.B5);
        return out;
    }

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (std::ptrdiff_t il23_s = 0; il23_s < static_cast<std::ptrdiff_t>(L23); ++il23_s) {
        for (std::ptrdiff_t il24_s = 0; il24_s < static_cast<std::ptrdiff_t>(L24); ++il24_s) {
            for (std::ptrdiff_t il25_s = 0; il25_s < static_cast<std::ptrdiff_t>(L25); ++il25_s) {
                const std::size_t il23 = static_cast<std::size_t>(il23_s);
                const std::size_t il24 = static_cast<std::size_t>(il24_s);
                const std::size_t il25 = static_cast<std::size_t>(il25_s);
                const std::size_t block = idx3(il23, il24, il25, L24, L25);
                out.B2.blocks[block] = axis_range_for_two_triangles(
                    out.channels.k2,
                    out.l23[il23], out.l24[il24],
                    out.l25[il25], s.Dj12);
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (std::ptrdiff_t ik3_s = 0; ik3_s < static_cast<std::ptrdiff_t>(K3d); ++ik3_s) {
        for (std::ptrdiff_t il34_s = 0; il34_s < static_cast<std::ptrdiff_t>(L34); ++il34_s) {
            for (std::ptrdiff_t il35_s = 0; il35_s < static_cast<std::ptrdiff_t>(L35); ++il35_s) {
                const std::size_t ik3 = static_cast<std::size_t>(ik3_s);
                const std::size_t il34 = static_cast<std::size_t>(il34_s);
                const std::size_t il35 = static_cast<std::size_t>(il35_s);
                const std::size_t block = idx3(ik3, il34, il35, L34, L35);
                const int Dk3 = out.channels.k3[ik3];
                if (!triangle_condition(Dk3, out.l34[il34], out.l35[il35])) {
                    out.B3.blocks[block] = empty_range_1d();
                } else {
                    out.B3.blocks[block] = axis_range_for_triangle(out.l23, Dk3, s.Dj13);
                }
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (std::ptrdiff_t ik4_s = 0; ik4_s < static_cast<std::ptrdiff_t>(K4d); ++ik4_s) {
        for (std::ptrdiff_t il24_s = 0; il24_s < static_cast<std::ptrdiff_t>(L24); ++il24_s) {
            for (std::ptrdiff_t il34_s = 0; il34_s < static_cast<std::ptrdiff_t>(L34); ++il34_s) {
                const std::size_t ik4 = static_cast<std::size_t>(ik4_s);
                const std::size_t il24 = static_cast<std::size_t>(il24_s);
                const std::size_t il34 = static_cast<std::size_t>(il34_s);
                const std::size_t block = idx3(ik4, il24, il34, L24, L34);
                const int Dk4 = out.channels.k4[ik4];
                if (!triangle_condition(Dk4, out.l24[il24], out.l34[il34])) {
                    out.B4.blocks[block] = empty_range_1d();
                } else {
                    out.B4.blocks[block] = axis_range_for_triangle(out.l45, Dk4, s.Dj14);
                }
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (std::ptrdiff_t il25_s = 0; il25_s < static_cast<std::ptrdiff_t>(L25); ++il25_s) {
        for (std::ptrdiff_t il35_s = 0; il35_s < static_cast<std::ptrdiff_t>(L35); ++il35_s) {
            for (std::ptrdiff_t il45_s = 0; il45_s < static_cast<std::ptrdiff_t>(L45); ++il45_s) {
                const std::size_t il25 = static_cast<std::size_t>(il25_s);
                const std::size_t il35 = static_cast<std::size_t>(il35_s);
                const std::size_t il45 = static_cast<std::size_t>(il45_s);
                const std::size_t block = idx3(il25, il35, il45, L35, L45);
                out.B5.blocks[block] = axis_range_for_two_triangles(
                    out.channels.k5,
                    s.Dj15, out.l25[il25],
                    out.l35[il35], out.l45[il45]);
            }
        }
    }

    finalize_offsets(out.B2);
    finalize_offsets(out.B3);
    finalize_offsets(out.B4);
    finalize_offsets(out.B5);

    return out;
}

AmplitudeInputSupportPlan_HalfSjC build_amplitude_static_booster_support_plan_half_sjC(
    const B4Spins& s,
    int Dl,
    const AmplitudeChannels& AmC)
{
    AmplitudeInputSupportPlan_HalfSjC out;

    if (AmC.xvals.empty()) {
        out.channels = AmC;
        out.l23 = make_l_axis(s.Dj23, Dl);
        out.l24 = make_l_axis(s.Dj24, Dl);
        out.l25 = make_l_axis(s.Dj25, Dl);
        out.l34 = make_l_axis(s.Dj34, Dl);
        out.l35 = make_l_axis(s.Dj35, Dl);
        out.l45 = make_l_axis(s.Dj45, Dl);
        return out;
    }

    // Build only a single-x support to get B2/B3/B4/B5 and all axes.  This does
    // not allocate sj values, and its sjC support metadata is only O(L^3), not
    // O(X L^3).  The static plan keeps the full x-axis in channels.
    AmplitudeChannels one_x = AmC;
    one_x.xvals.assign(1u, AmC.xvals.front());

    AmplitudeInputSupportPlan_HalfSjC tmp =
        build_amplitude_input_support_plan_half_sjC(s, Dl, one_x);

    out.channels = AmC;
    out.l23 = tmp.l23;
    out.l24 = tmp.l24;
    out.l25 = tmp.l25;
    out.l34 = tmp.l34;
    out.l35 = tmp.l35;
    out.l45 = tmp.l45;

    out.B2 = std::move(tmp.B2);
    out.B3 = std::move(tmp.B3);
    out.B4 = std::move(tmp.B4);
    out.B5 = std::move(tmp.B5);

    const std::size_t K2d = AmC.k2.size();
    const std::size_t K3d = AmC.k3.size();
    const std::size_t K4d = AmC.k4.size();
    const std::size_t K5d = AmC.k5.size();
    const std::size_t L23 = out.l23.size();
    const std::size_t L24 = out.l24.size();
    const std::size_t L25 = out.l25.size();
    const std::size_t L34 = out.l34.size();
    const std::size_t L35 = out.l35.size();
    const std::size_t L45 = out.l45.size();

    out.sjA = make_support_1d("sjA static empty support", {0u, L25}, K5d);
    out.sjB = make_support_1d("sjB static empty support", {0u, K4d, L35, L45}, K5d);
    out.sjC = make_support_2d("sjC static empty support", {0u, L24, L34, L35}, K3d, K4d);
    out.sjD = make_support_1d("sjD static empty support", {0u, K3d, L23, L24}, K2d);
    out.sjE = make_support_1d("sjE static empty support", {0u, L25}, K2d);
    finalize_offsets(out.sjA);
    finalize_offsets(out.sjB);
    finalize_offsets(out.sjC);
    finalize_offsets(out.sjD);
    finalize_offsets(out.sjE);

    // Reuse virtual intertwiner-axis maps. They do not depend on x.
    out.sjC_k3_to_k4 = tmp.sjC_k3_to_k4;
    out.sjC_k4_to_k3 = tmp.sjC_k4_to_k3;

    return out;
}

AmplitudeInputSupportPlan_HalfSjC build_amplitude_sj_xslab_support_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const B4Spins& s,
    std::size_t ix_begin,
    std::size_t ix_end)
{
    const std::size_t X = static_support.channels.xvals.size();
    ix_begin = std::min(ix_begin, X);
    ix_end = std::min(ix_end, X);
    if (ix_end < ix_begin) {
        ix_end = ix_begin;
    }

    AmplitudeChannels slab_channels = static_support.channels;
    slab_channels.xvals.assign(static_support.channels.xvals.begin() + static_cast<std::ptrdiff_t>(ix_begin),
                               static_support.channels.xvals.begin() + static_cast<std::ptrdiff_t>(ix_end));

    const int Dl = amplitude_half_sjC_infer_Dl_from_static_support(static_support);
    AmplitudeInputSupportPlan_HalfSjC out =
        build_amplitude_input_support_plan_half_sjC(s, Dl, slab_channels);

    // Keep the global/static booster supports and offsets.  packed_B is filled
    // from static_support, so slab-local B supports must reference the same
    // packed payload layout even though sjA/sjB/sjC/sjD/sjE are local to this
    // x-slab.
    out.B2 = static_support.B2;
    out.B3 = static_support.B3;
    out.B4 = static_support.B4;
    out.B5 = static_support.B5;

    // The virtual intertwiner-axis reflection maps do not depend on x.  Reuse the static
    // maps explicitly so generated x-slabs and full/static plans agree.
    out.sjC_k3_to_k4 = static_support.sjC_k3_to_k4;
    out.sjC_k4_to_k3 = static_support.sjC_k4_to_k3;

    return out;
}

AmplitudeInputSupportPlan_HalfSjC make_amplitude_xslice_qslab_support_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& xslice_full_support,
    std::size_t q_begin,
    std::size_t q_end)
{
    const std::size_t L24 = xslice_full_support.l24.size();
    const std::size_t L34 = xslice_full_support.l34.size();
    const std::size_t L35 = xslice_full_support.l35.size();
    const std::size_t Q = L24 * L34 * L35;

    q_begin = std::min(q_begin, Q);
    q_end = std::min(q_end, Q);
    if (q_end < q_begin) q_end = q_begin;

    AmplitudeInputSupportPlan_HalfSjC out = xslice_full_support;

    for (Range2D& r : out.sjC.blocks) {
        r = empty_range_2d();
    }
    for (AmplitudeSjCHalfBlockMap& m : out.sjC_map) {
        m.logical = empty_range_2d();
        m.storage_block = 0u;
        m.transpose_from_storage = false;
    }

    for (std::size_t q = q_begin; q < q_end; ++q) {
        std::size_t il24 = 0, il34 = 0, il35 = 0;
        amplitude_decode_q_half_sjC(xslice_full_support, q, il24, il34, il35);
        const std::size_t logical_block = idx4(0u, il24, il34, il35, L24, L34, L35);

        const AmplitudeSjCHalfBlockMap& src_map = xslice_full_support.sjC_map[logical_block];
        if (src_map.empty()) {
            continue;
        }

        out.sjC_map[logical_block] = src_map;
        const std::size_t storage_block = static_cast<std::size_t>(src_map.storage_block);
        if (storage_block < xslice_full_support.sjC.blocks.size()) {
            out.sjC.blocks[storage_block] = xslice_full_support.sjC.blocks[storage_block];
        }
    }

    finalize_offsets(out.sjC);
    return out;
}

} // namespace b4support
