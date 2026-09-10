// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/AmplitudePackedSJTensors_HalfSjC.hpp"

namespace b4support {

PackedAmplitudeSJFillChunks choose_packed_amplitude_sj_fill_chunks_local(
                                                                                const AmplitudeInputSupportPlan_HalfSjC& support,
                                                                                int requested_threads) noexcept
{
    const int nt = packed_sj_fill_num_threads(requested_threads);

    const std::size_t X   = support.channels.xvals.size();
    const std::size_t B3  = support.channels.k3.size();
    const std::size_t B4  = support.channels.k4.size();
    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    PackedAmplitudeSJFillChunks chunks;

    // sjA/sjE loop space is (l25,x); each iteration computes a contiguous
    // k5/k2 recurrence segment.  The previous heuristic used L25*B5/L25*B2,
    // which only happened to match in symmetric tests.  Computers do love
    // coincidences until they don't.
    chunks.sjA = choose_dynamic_chunk(packed_sj_mul_sat(L25, X),
                                      nt, 8, 128, 8);
    chunks.sjE = choose_dynamic_chunk(packed_sj_mul_sat(L25, X),
                                      nt, 8, 128, 8);

    // New stride-1 sjB loop space: (l45,l35,k4,x).
    // Each iteration may compute and store a contiguous k5 recurrence segment.
    const std::size_t nB = packed_sj_mul_sat(packed_sj_mul_sat(L45, L35),
                                             packed_sj_mul_sat(B4, X));
    chunks.sjB = choose_dynamic_chunk(nB, nt, 64, 1024, 512);

    // New stride-1 sjC task loop.  We parallelize over stored physical blocks;
    // each task loops over its stored k4 range and writes contiguous k3 segments.
    // The actual task count is not available until after task construction, so
    // this is only a fallback value.  The fill routine refines it using task size.
    const std::size_t nC = packed_sj_mul_sat(packed_sj_mul_sat(packed_sj_mul_sat(L35, L34), L24),
                                             packed_sj_mul_sat(X, B4));
    chunks.sjC = choose_dynamic_chunk(nC, nt, 64, 1024, 512);

    // New stride-1 sjD loop space: (l24,l23,k3,x).
    // Each iteration may compute and store a contiguous k2 recurrence segment.
    const std::size_t nD = packed_sj_mul_sat(packed_sj_mul_sat(L24, L23),
                                             packed_sj_mul_sat(B3, X));
    chunks.sjD = choose_dynamic_chunk(nD, nt, 64, 1024, 512);

    return chunks;
}

std::vector<SjCStoredBlockTask>
make_sjC_stored_block_tasks(const AmplitudeInputSupportPlan_HalfSjC& support)
{
    const std::size_t Dim_x = support.channels.xvals.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();

    std::vector<SjCStoredBlockTask> tasks;
    tasks.reserve(support.sjC.blocks.size());

    for (std::size_t ix = 0; ix < Dim_x; ++ix) {
        for (std::size_t il24 = 0; il24 < L24; ++il24) {
            for (std::size_t il34 = 0; il34 < L34; ++il34) {
                for (std::size_t il35 = 0; il35 < L35; ++il35) {
                    const std::size_t block =
                    idx4(ix, il24, il34, il35, L24, L34, L35);

                    const Range2D& r = support.sjC.blocks[block];
                    if (r.empty()) {
                        continue;
                    }

                    SjCStoredBlockTask task;
                    task.ix = ix;
                    task.il24 = il24;
                    task.il34 = il34;
                    task.il35 = il35;
                    task.i_begin = r.i_begin;
                    task.i_end = r.i_end;
                    task.j_begin = r.j_begin;
                    task.j_end = r.j_end;
                    task.offset = r.offset;
                    tasks.push_back(task);
                }
            }
        }
    }

    return tasks;
}

void fill_packed_amplitude_sj_tensors_rec_half_sjC(
                                                          const B4Spins& s,
                                                          const AmplitudeInputSupportPlan_HalfSjC& support,
                                                          PackedAmplitudeSJTensors_HalfSjC& out)
{
    const AmplitudeChannels& AmC = support.channels;

    const std::vector<int>& xvals = AmC.xvals;

    const std::vector<int>& k2 = AmC.k2;
    const std::vector<int>& k3 = AmC.k3;
    const std::vector<int>& k4 = AmC.k4;
    const std::vector<int>& k5 = AmC.k5;

    const std::vector<int>& l23 = support.l23;
    const std::vector<int>& l24 = support.l24;
    const std::vector<int>& l25 = support.l25;
    const std::vector<int>& l34 = support.l34;
    const std::vector<int>& l35 = support.l35;
    const std::vector<int>& l45 = support.l45;

    const std::size_t Dim_x  = xvals.size();

    const std::size_t Dim_k2 = k2.size();
    const std::size_t Dim_k3 = k3.size();
    const std::size_t Dim_k4 = k4.size();
    const std::size_t Dim_k5 = k5.size();

    const std::size_t L23 = l23.size();
    const std::size_t L24 = l24.size();
    const std::size_t L25 = l25.size();
    const std::size_t L35 = l35.size();
    const std::size_t L45 = l45.size();

    if (Dim_x == 0) {
        return;
    }

#ifndef NDEBUG
    assert_regular_step2_axis(xvals, "x");
    assert_regular_step2_axis(k2, "k2");
    assert_regular_step2_axis(k3, "k3");
    assert_regular_step2_axis(k4, "k4");
    assert_regular_step2_axis(k5, "k5");
#endif

    const int Di1 = AmC.i[0];

    const int Dj12 = s.Dj12;
    const int Dj13 = s.Dj13;
    const int Dj14 = s.Dj14;
    const int Dj15 = s.Dj15;

    const int Dj24 = s.Dj24;
    const int Dj34 = s.Dj34;
    const int Dj35 = s.Dj35;

#if B4PACK_SJ_FILL_TIMING
    auto tic = std::chrono::high_resolution_clock::now();
#endif

    out.allocate_from_support(support);

#if B4PACK_SJ_FILL_TIMING
    auto tac = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> timer = tac - tic;
    std::cout
    << "Time allocate_from_support: "
    << std::fixed << std::setprecision(6)
    << timer.count()
    << " seconds\n";
    tic = std::chrono::high_resolution_clock::now();
#endif

    std::vector<double> sqrt_k2(Dim_k2);
    std::vector<double> sqrt_k3(Dim_k3);
    std::vector<double> sqrt_k4(Dim_k4);
    std::vector<double> sqrt_k5(Dim_k5);
    std::vector<double> x_weight(Dim_x);

    const double sqrt_i1 = std::sqrt(static_cast<double>(Di1 + 1));

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

    PackedAmplitudeSJFillChunks chunks =
    choose_packed_amplitude_sj_fill_chunks_local(support);

#if B4PACK_SJ_FILL_TIMING
    tac = std::chrono::high_resolution_clock::now();
    timer = tac - tic;
    std::cout
    << "Time choose_packed_amplitude_sj_fill_chunks_local: "
    << std::fixed << std::setprecision(6)
    << timer.count()
    << " seconds\n";
    tic = std::chrono::high_resolution_clock::now();
#endif

    const std::vector<SjCStoredBlockTask> sjC_tasks =
    make_sjC_stored_block_tasks(support);

#if B4PACK_SJ_FILL_TIMING
    tac = std::chrono::high_resolution_clock::now();
    timer = tac - tic;
    std::cout
    << "Time make_sjC_stored_block_tasks: "
    << std::fixed << std::setprecision(6)
    << timer.count()
    << " seconds\n";
    tic = std::chrono::high_resolution_clock::now();
#endif

    // The sjC task loop is now over stored blocks, not the old 5D logical grid.
    // Refine the chunk from actual task count.  This is deliberately smaller
    // than the old x-recurrence chunk because each task is heavier.
    chunks.sjC = choose_dynamic_chunk(sjC_tasks.size(),
                                      packed_sj_fill_num_threads(),
                                      16,
                                      256,
                                      64);
#if B4PACK_SJ_FILL_TIMING
    tac = std::chrono::high_resolution_clock::now();
    timer = tac - tic;
    std::cout
    << "Time choose_dynamic_chunk: "
    << std::fixed << std::setprecision(6)
    << timer.count()
    << " seconds\n";
    tic = std::chrono::high_resolution_clock::now();
#endif


#if B4PACK_SJ_FILL_TIMING
    std::chrono::high_resolution_clock::time_point sj_stage_tic;
#endif

#pragma omp parallel
    {
        std::vector<double> rec_values;
        rec_values.reserve(std::max<std::size_t>({Dim_x, Dim_k2, Dim_k3, Dim_k4, Dim_k5}) + 8u);

        sjrec_detail::Wigner6jPreparedVariableJ4 sjB_j4_prep;
        sjrec_detail::Wigner6jPreparedVariableJ4Workspace sjB_j4_ws;

        sjrec_detail::Wigner6jPreparedVariableJ4 sjC_j4_prep;
        sjrec_detail::Wigner6jPreparedVariableJ4Workspace sjC_j4_ws;

        sjrec_detail::Wigner6jPreparedVariableJ4 sjD_j4_prep;
        sjrec_detail::Wigner6jPreparedVariableJ4Workspace sjD_j4_ws;

#if B4PACK_SJ_FILL_TIMING
#pragma omp barrier
#pragma omp master
        {
            sj_stage_tic = std::chrono::high_resolution_clock::now();
        }
#pragma omp barrier
#endif

        // ================================================================
        // sjA stride-1 fill.
        //
        // Old logical 6j:
        //   { x    i1   l25
        //     j15  k5   j14 }
        //
        // Tetrahedral symmetry, recurrence over k5:
        //   { k5   j15  l25
        //     i1   x    j14 }
        //
        // Packed layout:
        //   block = (x,l25), inner = k5
        //
        // Unlike the old x-recurrence path, this writes the packed 1D payload
        // contiguously for each stored (x,l25) block.
        // ================================================================
#pragma omp for collapse(2) schedule(dynamic, chunks.sjA) nowait
        for (std::size_t il25 = 0; il25 < L25; ++il25) {
            for (std::size_t ix = 0; ix < Dim_x; ++ix) {
                const std::size_t block = idx2(ix, il25, L25);

                const Range1D& r = support.sjA.blocks[block];
                if (r.empty()) {
                    continue;
                }

                const int Dx   = xvals[ix];
                const int Dl25 = l25[il25];

                // Fixed triangles for { k5 j15 l25 ; i1 x j14 }.
                if (!triangle_condition(Di1, Dj15, Dj14) ||
                    !triangle_condition(Di1, Dx,   Dl25)) {
                    continue;
                }

                const int k5_parity = (Dj15 + Dl25) & 1;
                if (((Dx + Dj14) & 1) != k5_parity) {
                    continue;
                }

                const int Dk5_min =
                std::max(iabs_int(Dj15 - Dl25),
                         iabs_int(Dx   - Dj14));

                const int Dk5_max =
                std::min(Dj15 + Dl25,
                         Dx   + Dj14);

                if (Dk5_min > Dk5_max) {
                    continue;
                }

                const Range1D rk5 =
                axis_range_from_interval_regular2(k5, Dk5_min, Dk5_max, k5_parity);

                if (rk5.empty()) {
                    continue;
                }

                const std::size_t ik5_begin =
                std::max<std::size_t>(static_cast<std::size_t>(rk5.begin),
                                      static_cast<std::size_t>(r.begin));

                const std::size_t ik5_end =
                std::min<std::size_t>(static_cast<std::size_t>(rk5.end),
                                      static_cast<std::size_t>(r.end));

                if (ik5_begin >= ik5_end) {
                    continue;
                }

                const bool ok =
                sjrec_detail::compute_wigner6j_x_series(
                                                        Dk5_min,
                                                        Dk5_max,
                                                        Dj15,
                                                        Dl25,
                                                        Di1,
                                                        Dx,
                                                        Dj14,
                                                        rec_values);

                if (!ok) {
                    continue;
                }

                double* const dst =
                out.sjA.data() + static_cast<std::size_t>(r.offset);

                const std::size_t local0 =
                ik5_begin - static_cast<std::size_t>(r.begin);

                const std::size_t n_store = ik5_end - ik5_begin;

                const std::size_t rec_off0 =
                sjrec_detail::x_offset_from_Dx(k5[ik5_begin], Dk5_min);

                const double xw = x_weight[ix];

#pragma omp simd
                for (std::size_t t = 0; t < n_store; ++t) {
                    const std::size_t ik5 = ik5_begin + t;
                    const int Dk5 = k5[ik5];

                    const double PrefA =
                    static_cast<double>(
                                        pm1_from_even_Dsum(
                                            Di1 + Dk5 + Dl25 + Dj15 -
                                            phase_split.a_parity))
                    * sqrt_i1
                    * sqrt_k5[ik5];

                    dst[local0 + t] =
                    PrefA * xw * rec_values[rec_off0 + t];
                }
            }
        }


#if B4PACK_SJ_FILL_TIMING
#pragma omp barrier
#pragma omp master
        {
            const auto sj_stage_tac = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> sj_stage_timer = sj_stage_tac - sj_stage_tic;
            std::cout << "Time fill sjA: "
                      << std::fixed << std::setprecision(6)
                      << sj_stage_timer.count()
                      << " seconds\n";
            sj_stage_tic = std::chrono::high_resolution_clock::now();
        }
#pragma omp barrier
#endif

        // ================================================================
        // sjB stride-1 fill.
        //
        // Old logical 6j:
        //   { x    k5   j14
        //     l45  k4   l35 }
        //
        // Tetrahedral symmetry lets us recur over k5 instead:
        //   { k5   x    j14
        //     k4   l45  l35 }
        //
        // Prepared-j4 path:
        //   fixed  = (x,l45,l35), recurrence variable = k5, j4 = k4.
        //   This hoists X/Z/Y_base/Y_f4 outside the k4 sweep.  Reusing work:
        //   an outrageously simple idea, somehow still worth writing down.
        // ================================================================
#pragma omp for collapse(3) schedule(dynamic, chunks.sjB) nowait
        for (std::size_t il45 = 0; il45 < L45; ++il45) {
            for (std::size_t il35 = 0; il35 < L35; ++il35) {
                for (std::size_t ix = 0; ix < Dim_x; ++ix) {
                    const int Dx   = xvals[ix];
                    const int Dl45 = l45[il45];
                    const int Dl35 = l35[il35];

                    const int k5_parity = (Dx + Dj14) & 1;
                    if (((Dl45 + Dl35) & 1) != k5_parity) {
                        continue;
                    }

                    const int Dk5_min =
                    std::max(iabs_int(Dx - Dj14),
                             iabs_int(Dl45 - Dl35));

                    const int Dk5_max =
                    std::min(Dx + Dj14,
                             Dl45 + Dl35);

                    if (Dk5_min > Dk5_max) {
                        continue;
                    }

                    const Range1D rk5 =
                    axis_range_from_interval_regular2(k5, Dk5_min, Dk5_max, k5_parity);

                    if (rk5.empty()) {
                        continue;
                    }

                    const int k4_parity = (Dx + Dl35) & 1;
                    if (((Dj14 + Dl45) & 1) != k4_parity) {
                        continue;
                    }

                    const int Dk4_min =
                    std::max(iabs_int(Dx   - Dl35),
                             iabs_int(Dj14 - Dl45));

                    const int Dk4_max =
                    std::min(Dx   + Dl35,
                             Dj14 + Dl45);

                    if (Dk4_min > Dk4_max) {
                        continue;
                    }

                    const Range1D rk4 =
                    axis_range_from_interval_regular2(k4, Dk4_min, Dk4_max, k4_parity);

                    if (rk4.empty()) {
                        continue;
                    }

                    const bool prep_ok =
                    sjrec_detail::prepare_wigner6j_x_series_variable_j4(
                                                                         Dk5_min,
                                                                         Dk5_max,
                                                                         Dx,
                                                                         Dj14,
                                                                         Dl45,
                                                                         Dl35,
                                                                         sjB_j4_prep);
                    if (!prep_ok) {
                        continue;
                    }

                    for (std::uint32_t ik4_u = rk4.begin;
                         ik4_u < rk4.end;
                         ++ik4_u) {
                        const std::size_t ik4 = static_cast<std::size_t>(ik4_u);
                        const int Dk4 = k4[ik4];

                        const std::size_t block =
                        idx4(ix, ik4, il35, il45,
                             Dim_k4, L35, L45);

                        const Range1D& r = support.sjB.blocks[block];
                        if (r.empty()) {
                            continue;
                        }

                        const std::size_t ik5_begin =
                        std::max<std::size_t>(static_cast<std::size_t>(rk5.begin),
                                              static_cast<std::size_t>(r.begin));

                        const std::size_t ik5_end =
                        std::min<std::size_t>(static_cast<std::size_t>(rk5.end),
                                              static_cast<std::size_t>(r.end));

                        if (ik5_begin >= ik5_end) {
                            continue;
                        }

                        double* const dst =
                        out.sjB.data() + static_cast<std::size_t>(r.offset);

                        const std::size_t local0 =
                        ik5_begin - static_cast<std::size_t>(r.begin);

                        const std::size_t n_store = ik5_end - ik5_begin;

                        const std::size_t rec_off0 =
                        sjrec_detail::x_offset_from_Dx(k5[ik5_begin], Dk5_min);

                        const double PrefB =
                        static_cast<double>(
                                            pm1_from_even_Dsum(Dk4 + Dl45 + Dj14 + Dl35 + Dj35))
                        * sqrt_k4[ik4];

                        const bool ok =
                        sjrec_detail::compute_wigner6j_x_series_prepared_j4_store_scaled(
                                                                                          sjB_j4_prep,
                                                                                          Dk4,
                                                                                          PrefB,
                                                                                          rec_off0,
                                                                                          n_store,
                                                                                          dst + local0,
                                                                                          sjB_j4_ws);
                        if (!ok) {
                            continue;
                        }
                    }
                }
            }
        }

#if B4PACK_SJ_FILL_TIMING
#pragma omp barrier
#pragma omp master
        {
            const auto sj_stage_tac = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> sj_stage_timer = sj_stage_tac - sj_stage_tic;
            std::cout << "Time fill sjB: "
                      << std::fixed << std::setprecision(6)
                      << sj_stage_timer.count()
                      << " seconds\n";
            sj_stage_tic = std::chrono::high_resolution_clock::now();
        }
#pragma omp barrier
#endif

        // ================================================================
        // sjC stride-1 fill over stored physical blocks.
        //
        // Old logical 6j:
        //   { x    k4   l35
        //     l34  k3   l24 }
        //
        // Tetrahedral symmetry, recurrence over k3:
        //   { k3   l34  l35
        //     k4   x    l24 }
        //
        // Packed layout:
        //   block = (x,l24,l34,l35), inner = k3 x k4
        // ================================================================

#pragma omp for schedule(dynamic, chunks.sjC) nowait
        for (std::size_t itask = 0; itask < sjC_tasks.size(); ++itask) {
            const SjCStoredBlockTask& task = sjC_tasks[itask];

            const std::size_t ix   = task.ix;
            const std::size_t il24 = task.il24;
            const std::size_t il34 = task.il34;
            const std::size_t il35 = task.il35;

            const int Dx   = xvals[ix];
            const int Dl24 = l24[il24];
            const int Dl34 = l34[il34];
            const int Dl35 = l35[il35];

            const int PrefC = pm1_from_even_Dsum(Dl34 + Dj34);

            const std::size_t nk3 =
            static_cast<std::size_t>(task.i_end - task.i_begin);

            // The recurrence variable is k3.  For fixed (x,l24,l34,l35), the
            // k3 recurrence range is constant across the whole k4 sweep.  The
            // previous code recomputed this per k4 column, because apparently
            // integers get lonely if not recalculated thousands of times.
            const int k3_parity = (Dl34 + Dl35) & 1;
            if (((Dx + Dl24) & 1) != k3_parity) {
                continue;
            }

            const int Dk3_min =
            std::max(iabs_int(Dl34 - Dl35),
                     iabs_int(Dx   - Dl24));

            const int Dk3_max =
            std::min(Dl34 + Dl35,
                     Dx   + Dl24);

            if (Dk3_min > Dk3_max) {
                continue;
            }

            const Range1D rk3 =
            axis_range_from_interval_regular2(k3, Dk3_min, Dk3_max, k3_parity);

            if (rk3.empty()) {
                continue;
            }

            const std::size_t ik3_begin =
            std::max<std::size_t>(static_cast<std::size_t>(rk3.begin),
                                  static_cast<std::size_t>(task.i_begin));

            const std::size_t ik3_end =
            std::min<std::size_t>(static_cast<std::size_t>(rk3.end),
                                  static_cast<std::size_t>(task.i_end));

            if (ik3_begin >= ik3_end) {
                continue;
            }

            const std::size_t local0 =
            ik3_begin - static_cast<std::size_t>(task.i_begin);

            const std::size_t n_store = ik3_end - ik3_begin;

            const std::size_t rec_off0 =
            sjrec_detail::x_offset_from_Dx(k3[ik3_begin], Dk3_min);

            // k4 is the varying non-recurrence leg.  Its valid interval is also
            // fixed for this stored block, so intersect it once with the stored
            // physical k4 range instead of testing both triangles per column.
            const int k4_parity = (Dl34 + Dl24) & 1;
            if (((Dx + Dl35) & 1) != k4_parity) {
                continue;
            }

            const int Dk4_min =
            std::max(iabs_int(Dl34 - Dl24),
                     iabs_int(Dx   - Dl35));

            const int Dk4_max =
            std::min(Dl34 + Dl24,
                     Dx   + Dl35);

            if (Dk4_min > Dk4_max) {
                continue;
            }

            const Range1D rk4 =
            axis_range_from_interval_regular2(k4, Dk4_min, Dk4_max, k4_parity);

            if (rk4.empty()) {
                continue;
            }

            const std::uint32_t ik4_begin =
            std::max<std::uint32_t>(rk4.begin, task.j_begin);

            const std::uint32_t ik4_end =
            std::min<std::uint32_t>(rk4.end, task.j_end);

            if (ik4_begin >= ik4_end) {
                continue;
            }

            const bool prep_ok =
            sjrec_detail::prepare_wigner6j_x_series_variable_j4(
                                                                 Dk3_min,
                                                                 Dk3_max,
                                                                 Dl34,
                                                                 Dl35,
                                                                 Dx,
                                                                 Dl24,
                                                                 sjC_j4_prep);
            if (!prep_ok) {
                continue;
            }

            for (std::uint32_t ik4_u = ik4_begin;
                 ik4_u < ik4_end;
                 ++ik4_u) {
                const std::size_t ik4 = static_cast<std::size_t>(ik4_u);
                const int Dk4 = k4[ik4];

                double* const col =
                out.sjC.data()
                + static_cast<std::size_t>(task.offset)
                + static_cast<std::size_t>(ik4_u - task.j_begin) * nk3;

                const bool ok =
                sjrec_detail::compute_wigner6j_x_series_prepared_j4_store_scaled(
                                                                                  sjC_j4_prep,
                                                                                  Dk4,
                                                                                  static_cast<double>(PrefC),
                                                                                  rec_off0,
                                                                                  n_store,
                                                                                  col + local0,
                                                                                  sjC_j4_ws);

                if (!ok) {
                    continue;
                }
            }
        }



#if B4PACK_SJ_FILL_TIMING
#pragma omp barrier
#pragma omp master
        {
            const auto sj_stage_tac = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> sj_stage_timer = sj_stage_tac - sj_stage_tic;
            std::cout << "Time fill sjC: "
                      << std::fixed << std::setprecision(6)
                      << sj_stage_timer.count()
                      << " seconds\n";
            sj_stage_tic = std::chrono::high_resolution_clock::now();
        }
#pragma omp barrier
#endif

        // ================================================================
        // sjD stride-1 fill.
        //
        // Old logical 6j:
        //   { x    k3   l24
        //     l23  k2   j13 }
        //
        // Tetrahedral symmetry, recurrence over k2:
        //   { k2   l23  l24
        //     k3   x    j13 }
        //
        // Prepared-j4 path:
        //   fixed = (x,l23,l24), recurrence variable = k2, j4 = k3.
        // ================================================================
#pragma omp for collapse(3) schedule(dynamic, chunks.sjD) nowait
        for (std::size_t il24 = 0; il24 < L24; ++il24) {
            for (std::size_t il23 = 0; il23 < L23; ++il23) {
                for (std::size_t ix = 0; ix < Dim_x; ++ix) {
                    const int Dx   = xvals[ix];
                    const int Dl24 = l24[il24];
                    const int Dl23 = l23[il23];

                    const int k2_parity = (Dl23 + Dl24) & 1;
                    if (((Dx + Dj13) & 1) != k2_parity) {
                        continue;
                    }

                    const int Dk2_min =
                    std::max(iabs_int(Dl23 - Dl24),
                             iabs_int(Dx - Dj13));

                    const int Dk2_max =
                    std::min(Dl23 + Dl24,
                             Dx + Dj13);

                    if (Dk2_min > Dk2_max) {
                        continue;
                    }

                    const Range1D rk2 =
                    axis_range_from_interval_regular2(k2, Dk2_min, Dk2_max, k2_parity);

                    if (rk2.empty()) {
                        continue;
                    }

                    const int k3_parity = (Dl23 + Dj13) & 1;
                    if (((Dl24 + Dx) & 1) != k3_parity) {
                        continue;
                    }

                    const int Dk3_min =
                    std::max(iabs_int(Dl23 - Dj13),
                             iabs_int(Dl24 - Dx));

                    const int Dk3_max =
                    std::min(Dl23 + Dj13,
                             Dl24 + Dx);

                    if (Dk3_min > Dk3_max) {
                        continue;
                    }

                    const Range1D rk3 =
                    axis_range_from_interval_regular2(k3, Dk3_min, Dk3_max, k3_parity);

                    if (rk3.empty()) {
                        continue;
                    }

                    const bool prep_ok =
                    sjrec_detail::prepare_wigner6j_x_series_variable_j4(
                                                                         Dk2_min,
                                                                         Dk2_max,
                                                                         Dl23,
                                                                         Dl24,
                                                                         Dx,
                                                                         Dj13,
                                                                         sjD_j4_prep);
                    if (!prep_ok) {
                        continue;
                    }

                    for (std::uint32_t ik3_u = rk3.begin;
                         ik3_u < rk3.end;
                         ++ik3_u) {
                        const std::size_t ik3 = static_cast<std::size_t>(ik3_u);
                        const int Dk3 = k3[ik3];

                        const std::size_t block =
                        idx4(ix, ik3, il23, il24,
                             Dim_k3, L23, L24);

                        const Range1D& r = support.sjD.blocks[block];
                        if (r.empty()) {
                            continue;
                        }

                        const std::size_t ik2_begin =
                        std::max<std::size_t>(static_cast<std::size_t>(rk2.begin),
                                              static_cast<std::size_t>(r.begin));

                        const std::size_t ik2_end =
                        std::min<std::size_t>(static_cast<std::size_t>(rk2.end),
                                              static_cast<std::size_t>(r.end));

                        if (ik2_begin >= ik2_end) {
                            continue;
                        }

                        const bool ok =
                        sjrec_detail::compute_wigner6j_x_series_prepared_j4(
                                                                             sjD_j4_prep,
                                                                             Dk3,
                                                                             rec_values,
                                                                             sjD_j4_ws);

                        if (!ok) {
                            continue;
                        }

                        double* const dst =
                        out.sjD.data() + static_cast<std::size_t>(r.offset);

                        const std::size_t local0 =
                        ik2_begin - static_cast<std::size_t>(r.begin);

                        const std::size_t n_store = ik2_end - ik2_begin;

                        const std::size_t rec_off0 =
                        sjrec_detail::x_offset_from_Dx(k2[ik2_begin], Dk2_min);

#pragma omp simd
                        for (std::size_t t = 0; t < n_store; ++t) {
                            const std::size_t ik2 = ik2_begin + t;
                            const int Dk2 = k2[ik2];
                            const double PrefD =
                            static_cast<double>(
                                                pm1_from_even_Dsum(
                                                    Dk2 + Dk3 + Dl23 + Dj13 + Dl24 + Dj24 -
                                                    phase_split.d_parity))
                            * sqrt_k2[ik2]
                            * sqrt_k3[ik3];

                            dst[local0 + t] = PrefD * rec_values[rec_off0 + t];
                        }
                    }
                }
            }
        }

#if B4PACK_SJ_FILL_TIMING
#pragma omp barrier
#pragma omp master
        {
            const auto sj_stage_tac = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> sj_stage_timer = sj_stage_tac - sj_stage_tic;
            std::cout << "Time fill sjD: "
                      << std::fixed << std::setprecision(6)
                      << sj_stage_timer.count()
                      << " seconds\n";
            sj_stage_tic = std::chrono::high_resolution_clock::now();
        }
#pragma omp barrier
#endif

        // ================================================================
        // sjE stride-1 fill.
        //
        // Old logical 6j:
        //   { x    i1   l25
        //     j12  k2   j13 }
        //
        // Tetrahedral symmetry, recurrence over k2:
        //   { k2   j12  l25
        //     i1   x    j13 }
        //
        // Packed layout:
        //   block = (x,l25), inner = k2
        //
        // This tensor is tiny, so the performance impact is modest, but this
        // makes all packed sjX fills consistent: recurrence along the physical
        // inner dimension whenever possible.
        // ================================================================
#pragma omp for collapse(2) schedule(dynamic, chunks.sjE) nowait
        for (std::size_t il25 = 0; il25 < L25; ++il25) {
            for (std::size_t ix = 0; ix < Dim_x; ++ix) {
                const std::size_t block = idx2(ix, il25, L25);

                const Range1D& r = support.sjE.blocks[block];
                if (r.empty()) {
                    continue;
                }

                const int Dx   = xvals[ix];
                const int Dl25 = l25[il25];

                // Fixed triangles for { k2 j12 l25 ; i1 x j13 }.
                if (!triangle_condition(Di1, Dj12, Dj13) ||
                    !triangle_condition(Di1, Dx,   Dl25)) {
                    continue;
                }

                const int k2_parity = (Dj12 + Dl25) & 1;
                if (((Dx + Dj13) & 1) != k2_parity) {
                    continue;
                }

                const int Dk2_min =
                std::max(iabs_int(Dj12 - Dl25),
                         iabs_int(Dx   - Dj13));

                const int Dk2_max =
                std::min(Dj12 + Dl25,
                         Dx   + Dj13);

                if (Dk2_min > Dk2_max) {
                    continue;
                }

                const Range1D rk2 =
                axis_range_from_interval_regular2(k2, Dk2_min, Dk2_max, k2_parity);

                if (rk2.empty()) {
                    continue;
                }

                const std::size_t ik2_begin =
                std::max<std::size_t>(static_cast<std::size_t>(rk2.begin),
                                      static_cast<std::size_t>(r.begin));

                const std::size_t ik2_end =
                std::min<std::size_t>(static_cast<std::size_t>(rk2.end),
                                      static_cast<std::size_t>(r.end));

                if (ik2_begin >= ik2_end) {
                    continue;
                }

                const bool ok =
                sjrec_detail::compute_wigner6j_x_series(
                                                        Dk2_min,
                                                        Dk2_max,
                                                        Dj12,
                                                        Dl25,
                                                        Di1,
                                                        Dx,
                                                        Dj13,
                                                        rec_values);

                if (!ok) {
                    continue;
                }

                double* const dst =
                out.sjE.data() + static_cast<std::size_t>(r.offset);

                const std::size_t local0 =
                ik2_begin - static_cast<std::size_t>(r.begin);

                const std::size_t n_store = ik2_end - ik2_begin;

                const std::size_t rec_off0 =
                sjrec_detail::x_offset_from_Dx(k2[ik2_begin], Dk2_min);

#pragma omp simd
                for (std::size_t t = 0; t < n_store; ++t) {
                    dst[local0 + t] =
                    static_cast<double>(PrefE)
                    * rec_values[rec_off0 + t];
                }
            }
        }

#if B4PACK_SJ_FILL_TIMING
#pragma omp barrier
#pragma omp master
        {
            const auto sj_stage_tac = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> sj_stage_timer = sj_stage_tac - sj_stage_tic;
            std::cout << "Time fill sjE: "
                      << std::fixed << std::setprecision(6)
                      << sj_stage_timer.count()
                      << " seconds\n";
            sj_stage_tic = std::chrono::high_resolution_clock::now();
        }
#pragma omp barrier
#endif

    }

#if B4PACK_SJ_FILL_TIMING
    tac = std::chrono::high_resolution_clock::now();
    timer = tac - tic;
    std::cout
    << "Time actual filling: "
    << std::fixed << std::setprecision(6)
    << timer.count()
    << " seconds\n";
    tic = std::chrono::high_resolution_clock::now();
#endif

}

} // namespace b4support
