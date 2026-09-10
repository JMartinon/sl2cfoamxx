// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "full_tensor/FullTensorPackedSJTensors_HalfSjC.hpp"

#include "amplitude/AmplitudePackedSJTensors.hpp"
#include "wigner/SixJTensors.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

namespace b4support {


void fill_packed_full_tensor_sj_tensors_rec(
    const B4Spins& s,
    const FullTensorInputSupportPlan& support,
    PackedFullTensorSJTensors& out,
    std::uint8_t components)
{
    const FullTensorChannels& ch = support.channels;

    const std::vector<int>& i1 = ch.i1;
    const std::vector<int>& xvals = ch.xvals;

    const std::vector<int>& k2 = ch.k2;
    const std::vector<int>& k3 = ch.k3;
    const std::vector<int>& k4 = ch.k4;
    const std::vector<int>& k5 = ch.k5;

    const std::vector<int>& l23 = support.l23;
    const std::vector<int>& l24 = support.l24;
    const std::vector<int>& l25 = support.l25;
    const std::vector<int>& l34 = support.l34;
    const std::vector<int>& l35 = support.l35;
    const std::vector<int>& l45 = support.l45;

    const std::size_t Dim_i1 = i1.size();
    const std::size_t Dim_x  = xvals.size();

    const std::size_t Dim_k2 = k2.size();
    const std::size_t Dim_k3 = k3.size();
    const std::size_t Dim_k4 = k4.size();
    const std::size_t Dim_k5 = k5.size();

    const std::size_t L23 = l23.size();
    const std::size_t L24 = l24.size();
    const std::size_t L25 = l25.size();
    const std::size_t L34 = l34.size();
    const std::size_t L35 = l35.size();
    const std::size_t L45 = l45.size();

    out.clear();
    if (components & FullTensorSjA) out.sjA.assign(static_cast<std::size_t>(support.sjA.packed_values), 0.0);
    if (components & FullTensorSjB) out.sjB.assign(static_cast<std::size_t>(support.sjB.packed_values), 0.0);
    if (components & FullTensorSjC) out.sjC.assign(static_cast<std::size_t>(support.sjC.packed_values), 0.0);
    if (components & FullTensorSjD) out.sjD.assign(static_cast<std::size_t>(support.sjD.packed_values), 0.0);
    if (components & FullTensorSjE) out.sjE.assign(static_cast<std::size_t>(support.sjE.packed_values), 0.0);

    if (Dim_x == 0 || Dim_k2 == 0 || Dim_k3 == 0 || Dim_k4 == 0 || Dim_k5 == 0)
    {
        return;
    }

    const int Dj12 = s.Dj12;
    const int Dj13 = s.Dj13;
    const int Dj14 = s.Dj14;
    const int Dj15 = s.Dj15;

    const int Dj24 = s.Dj24;
    const int Dj34 = s.Dj34;
    const int Dj35 = s.Dj35;

    const int Dx0 = xvals.front();
    const int x_parity = Dx0 & 1;

    std::vector<double> sqrt_i1(Dim_i1);
    std::vector<double> sqrt_k2(Dim_k2);
    std::vector<double> sqrt_k3(Dim_k3);
    std::vector<double> sqrt_k4(Dim_k4);
    std::vector<double> sqrt_k5(Dim_k5);
    std::vector<double> x_weight(Dim_x);

    for (std::size_t i = 0; i < Dim_i1; ++i) {
        sqrt_i1[i] = std::sqrt(static_cast<double>(i1[i] + 1));
    }
    for (std::size_t i = 0; i < Dim_k2; ++i) {
        sqrt_k2[i] = std::sqrt(static_cast<double>(k2[i] + 1));
    }
    for (std::size_t i = 0; i < Dim_k3; ++i) {
        sqrt_k3[i] = std::sqrt(static_cast<double>(k3[i] + 1));
    }
    for (std::size_t i = 0; i < Dim_k4; ++i) {
        sqrt_k4[i] = std::sqrt(static_cast<double>(k4[i] + 1));
    }
    for (std::size_t i = 0; i < Dim_k5; ++i) {
        sqrt_k5[i] = std::sqrt(static_cast<double>(k5[i] + 1));
    }
    for (std::size_t ix = 0; ix < Dim_x; ++ix) {
        x_weight[ix] = static_cast<double>(xvals[ix] + 1);
    }

    const Wigner15jRealPhaseSplit phase_split =
        wigner15j_real_phase_split(s);
    const int PrefE = phase_split.e_correction * pm1_from_even_Dsum(
        Dj12 - Dj35 - Dj34 - Dj24 - phase_split.e_parity);

    int max_map_value = 0;
    max_map_value = std::max(max_map_value, max_value_in_vector(k3));
    max_map_value = std::max(max_map_value, max_value_in_vector(k4));
    max_map_value = std::max(max_map_value, max_value_in_vector(l24));
    max_map_value = std::max(max_map_value, max_value_in_vector(l35));

    const std::vector<int> val_to_ik3 = make_value_to_index_map(k3, max_map_value);
    const std::vector<int> val_to_ik4 = make_value_to_index_map(k4, max_map_value);
    const std::vector<int> val_to_il24 = make_value_to_index_map(l24, max_map_value);
    const std::vector<int> val_to_il35 = make_value_to_index_map(l35, max_map_value);

    constexpr int chunk_size = 16;

#pragma omp parallel
    {
        std::vector<double> rec_values;

        if (components & FullTensorSjA) {
        // ================================================================
        // sjA block (i1,x,l25) -> k5
        // ================================================================
#pragma omp for collapse(3) schedule(dynamic, chunk_size)
        for (std::size_t ii1 = 0; ii1 < Dim_i1; ++ii1) {
            for (std::size_t il25 = 0; il25 < L25; ++il25) {
                for (std::size_t ik5 = 0; ik5 < Dim_k5; ++ik5) {
                    const int Di1  = i1[ii1];
                    const int Dl25 = l25[il25];
                    const int Dk5  = k5[ik5];

                    if (!triangle_condition(Dk5, Dl25, Dj15) ||
                        !triangle_condition(Di1, Dj14, Dj15))
                    {
                        continue;
                    }

                    if (((Di1 ^ Dl25 ^ x_parity) |
                         (Dk5 ^ Dj14 ^ x_parity)) & 1)
                    {
                        continue;
                    }

                    const int Dx_min = std::max(
                        iabs_int(Di1 - Dl25),
                        iabs_int(Dk5 - Dj14)
                    );
                    const int Dx_max = std::min(
                        Di1 + Dl25,
                        Dk5 + Dj14
                    );

                    if (Dx_min > Dx_max || Dx_max < Dx0) {
                        continue;
                    }

                    const Range1D rx = axis_range_from_interval(
                        xvals,
                        Dx_min,
                        Dx_max,
                        x_parity
                    );

                    if (rx.empty()) {
                        continue;
                    }

                    const bool ok = sjrec_detail::compute_wigner6j_x_series(
                        Dx_min,
                        Dx_max,
                        Di1,
                        Dl25,
                        Dj15,
                        Dk5,
                        Dj14,
                        rec_values
                    );

                    if (!ok) {
                        continue;
                    }

                    const double PrefA =
                        static_cast<double>(pm1_from_even_Dsum(
                            Di1 + Dk5 + Dl25 + Dj15 - phase_split.a_parity))
                        * sqrt_i1[ii1]
                        * sqrt_k5[ik5];

                    for (std::size_t ix = rx.begin; ix < rx.end; ++ix) {
                        const int Dx = xvals[ix];
                        const std::size_t off = sjrec_detail::x_offset_from_Dx(Dx, Dx_min);
                        const std::size_t block = idx3(ii1, ix, il25, Dim_x, L25);

                        packed_store_1d(
                            support.sjA,
                            block,
                            ik5,
                            out.sjA.data(),
                            PrefA * x_weight[ix] * rec_values[off]
                        );
                    }
                }
            }
        }
        }

        if (components & FullTensorSjB) {
        // ================================================================
        // sjB block (x,k4,l35,l45) -> k5
        // ================================================================
#pragma omp for collapse(4) schedule(dynamic, chunk_size)
        for (std::size_t il45 = 0; il45 < L45; ++il45) {
            for (std::size_t il35 = 0; il35 < L35; ++il35) {
                for (std::size_t ik4 = 0; ik4 < Dim_k4; ++ik4) {
                    for (std::size_t ik5 = 0; ik5 < Dim_k5; ++ik5) {
                        const int Dl45 = l45[il45];
                        const int Dl35 = l35[il35];
                        const int Dk4  = k4[ik4];
                        const int Dk5  = k5[ik5];

                        if (!triangle_condition(Dk5, Dl35, Dl45) ||
                            !triangle_condition(Dk4, Dj14, Dl45))
                        {
                            continue;
                        }

                        if (((Dk5 ^ Dj14 ^ x_parity) |
                             (Dk4 ^ Dl35 ^ x_parity)) & 1)
                        {
                            continue;
                        }

                        const int Dx_min = std::max(
                            iabs_int(Dk4 - Dl35),
                            iabs_int(Dk5 - Dj14)
                        );
                        const int Dx_max = std::min(
                            Dk4 + Dl35,
                            Dk5 + Dj14
                        );

                        if (Dx_min > Dx_max || Dx_max < Dx0) {
                            continue;
                        }

                        const Range1D rx = axis_range_from_interval(
                            xvals,
                            Dx_min,
                            Dx_max,
                            x_parity
                        );

                        if (rx.empty()) {
                            continue;
                        }

                        const bool ok = sjrec_detail::compute_wigner6j_x_series(
                            Dx_min,
                            Dx_max,
                            Dk5,
                            Dj14,
                            Dl45,
                            Dk4,
                            Dl35,
                            rec_values
                        );

                        if (!ok) {
                            continue;
                        }

                        const double PrefB =
                            static_cast<double>(pm1_from_even_Dsum(Dk4 + Dl45 + Dj14 + Dl35 + Dj35))
                            * sqrt_k4[ik4];

                        for (std::size_t ix = rx.begin; ix < rx.end; ++ix) {
                            const int Dx = xvals[ix];
                            const std::size_t off = sjrec_detail::x_offset_from_Dx(Dx, Dx_min);
                            const std::size_t block = idx4(ix, ik4, il35, il45, Dim_k4, L35, L45);

                            packed_store_1d(
                                support.sjB,
                                block,
                                ik5,
                                out.sjB.data(),
                                PrefB * rec_values[off]
                            );
                        }
                    }
                }
            }
        }
        }

        if (components & FullTensorSjC) {
        // ================================================================
        // sjC half-stored blocks.
        // ================================================================
        // We fill only blocks that own storage in support.sjC.blocks.  The
        // logical non-canonical blocks are represented by support.sjC_map and
        // are reconstructed/transposed by the contraction accessor.
#pragma omp for collapse(5) schedule(dynamic, chunk_size)
        for (std::size_t il35 = 0; il35 < L35; ++il35) {
            for (std::size_t il34 = 0; il34 < L34; ++il34) {
                for (std::size_t il24 = 0; il24 < L24; ++il24) {
                    for (std::size_t ik3 = 0; ik3 < Dim_k3; ++ik3) {
                        for (std::size_t ik4 = 0; ik4 < Dim_k4; ++ik4) {
                            const int Dl35 = l35[il35];
                            const int Dl34 = l34[il34];
                            const int Dl24 = l24[il24];
                            const int Dk3  = k3[ik3];
                            const int Dk4  = k4[ik4];

                            if (!triangle_condition(Dk3, Dl35, Dl34) ||
                                !triangle_condition(Dk4, Dl24, Dl34))
                            {
                                continue;
                            }

                            if (((Dk4 ^ Dl35 ^ x_parity) |
                                 (Dk3 ^ Dl24 ^ x_parity)) & 1)
                            {
                                continue;
                            }

                            const int Dx_min = std::max(
                                iabs_int(Dk4 - Dl35),
                                iabs_int(Dk3 - Dl24)
                            );
                            const int Dx_max = std::min(
                                Dk4 + Dl35,
                                Dk3 + Dl24
                            );

                            if (Dx_min > Dx_max || Dx_max < Dx0) {
                                continue;
                            }

                            const Range1D rx = axis_range_from_interval(
                                xvals,
                                Dx_min,
                                Dx_max,
                                x_parity
                            );

                            if (rx.empty()) {
                                continue;
                            }

                            bool owns_any_storage_block = false;
                            for (std::size_t ix = rx.begin; ix < rx.end; ++ix) {
                                const std::size_t block = idx4(ix, il24, il34, il35, L24, L34, L35);
                                const FullTensorSjCHalfBlockMap& map = support.sjC_map[block];
                                if (!map.empty() &&
                                    !map.transpose_from_storage &&
                                    static_cast<std::size_t>(map.storage_block) == block)
                                {
                                    owns_any_storage_block = true;
                                    break;
                                }
                            }
                            if (!owns_any_storage_block) {
                                continue;
                            }

                            const bool ok = sjrec_detail::compute_wigner6j_x_series(
                                Dx_min,
                                Dx_max,
                                Dk4,
                                Dl35,
                                Dl34,
                                Dk3,
                                Dl24,
                                rec_values
                            );

                            if (!ok) {
                                continue;
                            }

                            const int PrefC = pm1_from_even_Dsum(Dl34 + Dj34);

                            for (std::size_t ix = rx.begin; ix < rx.end; ++ix) {
                                const std::size_t block = idx4(ix, il24, il34, il35, L24, L34, L35);
                                const FullTensorSjCHalfBlockMap& map = support.sjC_map[block];

                                // Only this canonical logical block owns payload.
                                if (map.empty() || map.transpose_from_storage ||
                                    static_cast<std::size_t>(map.storage_block) != block)
                                {
                                    continue;
                                }

                                const Range2D& r_store = support.sjC.blocks[block];
                                if (r_store.empty()) {
                                    continue;
                                }

                                const int Dx = xvals[ix];
                                const std::size_t off = sjrec_detail::x_offset_from_Dx(Dx, Dx_min);
                                const double out_val = static_cast<double>(PrefC) * rec_values[off];

                                packed_store_2d(
                                    support.sjC,
                                    block,
                                    ik3,
                                    ik4,
                                    out.sjC.data(),
                                    out_val
                                );
                            }
                        }
                    }
                }
            }
        }
        }

        if (components & FullTensorSjD) {
        // ================================================================
        // sjD block (x,k3,l23,l24) -> k2
        // ================================================================
#pragma omp for collapse(4) schedule(dynamic, chunk_size)
        for (std::size_t il24 = 0; il24 < L24; ++il24) {
            for (std::size_t il23 = 0; il23 < L23; ++il23) {
                for (std::size_t ik2 = 0; ik2 < Dim_k2; ++ik2) {
                    for (std::size_t ik3 = 0; ik3 < Dim_k3; ++ik3) {
                        const int Dl24 = l24[il24];
                        const int Dl23 = l23[il23];
                        const int Dk2  = k2[ik2];
                        const int Dk3  = k3[ik3];

                        if (!triangle_condition(Dk2, Dl24, Dl23) ||
                            !triangle_condition(Dk3, Dj13, Dl23))
                        {
                            continue;
                        }

                        if (((Dk3 ^ Dl24 ^ x_parity) |
                             (Dk2 ^ Dj13 ^ x_parity)) & 1)
                        {
                            continue;
                        }

                        const int Dx_min = std::max(
                            iabs_int(Dk3 - Dl24),
                            iabs_int(Dk2 - Dj13)
                        );
                        const int Dx_max = std::min(
                            Dk3 + Dl24,
                            Dk2 + Dj13
                        );

                        if (Dx_min > Dx_max || Dx_max < Dx0) {
                            continue;
                        }

                        const Range1D rx = axis_range_from_interval(
                            xvals,
                            Dx_min,
                            Dx_max,
                            x_parity
                        );

                        if (rx.empty()) {
                            continue;
                        }

                        const bool ok = sjrec_detail::compute_wigner6j_x_series(
                            Dx_min,
                            Dx_max,
                            Dk3,
                            Dl24,
                            Dl23,
                            Dk2,
                            Dj13,
                            rec_values
                        );

                        if (!ok) {
                            continue;
                        }

                        const double PrefD =
                            static_cast<double>(pm1_from_even_Dsum(
                                Dk2 + Dk3 + Dl23 + Dj13 + Dl24 + Dj24 -
                                phase_split.d_parity))
                            * sqrt_k2[ik2]
                            * sqrt_k3[ik3];

                        for (std::size_t ix = rx.begin; ix < rx.end; ++ix) {
                            const int Dx = xvals[ix];
                            const std::size_t off = sjrec_detail::x_offset_from_Dx(Dx, Dx_min);
                            const std::size_t block = idx4(ix, ik3, il23, il24, Dim_k3, L23, L24);

                            packed_store_1d(
                                support.sjD,
                                block,
                                ik2,
                                out.sjD.data(),
                                PrefD * rec_values[off]
                            );
                        }
                    }
                }
            }
        }
        }

        if (components & FullTensorSjE) {
        // ================================================================
        // sjE block (i1,x,l25) -> k2
        // ================================================================
#pragma omp for collapse(3) schedule(dynamic, chunk_size)
        for (std::size_t ii1 = 0; ii1 < Dim_i1; ++ii1) {
            for (std::size_t il25 = 0; il25 < L25; ++il25) {
                for (std::size_t ik2 = 0; ik2 < Dim_k2; ++ik2) {
                    const int Di1  = i1[ii1];
                    const int Dl25 = l25[il25];
                    const int Dk2  = k2[ik2];

                    if (!triangle_condition(Dk2, Dl25, Dj12) ||
                        !triangle_condition(Di1, Dj13, Dj12))
                    {
                        continue;
                    }

                    if (((Di1 ^ Dl25 ^ x_parity) |
                         (Dk2 ^ Dj13 ^ x_parity)) & 1)
                    {
                        continue;
                    }

                    const int Dx_min = std::max(
                        iabs_int(Di1 - Dl25),
                        iabs_int(Dk2 - Dj13)
                    );
                    const int Dx_max = std::min(
                        Di1 + Dl25,
                        Dk2 + Dj13
                    );

                    if (Dx_min > Dx_max || Dx_max < Dx0) {
                        continue;
                    }

                    const Range1D rx = axis_range_from_interval(
                        xvals,
                        Dx_min,
                        Dx_max,
                        x_parity
                    );

                    if (rx.empty()) {
                        continue;
                    }

                    const bool ok = sjrec_detail::compute_wigner6j_x_series(
                        Dx_min,
                        Dx_max,
                        Di1,
                        Dl25,
                        Dj12,
                        Dk2,
                        Dj13,
                        rec_values
                    );

                    if (!ok) {
                        continue;
                    }

                    for (std::size_t ix = rx.begin; ix < rx.end; ++ix) {
                        const int Dx = xvals[ix];
                        const std::size_t off = sjrec_detail::x_offset_from_Dx(Dx, Dx_min);
                        const std::size_t block = idx3(ii1, ix, il25, Dim_x, L25);

                        packed_store_1d(
                            support.sjE,
                            block,
                            ik2,
                            out.sjE.data(),
                            static_cast<double>(PrefE) * rec_values[off]
                        );
                    }
                }
            }
        }
        }
    }
}

} // namespace b4support
