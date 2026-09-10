// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/AmplitudePackedContraction.hpp"

// The header keeps this helper macro local to its declarations and inline
// kernels. Apply the same pointer contract to the out-of-line implementations.
#if defined(__GNUC__) || defined(__clang__)
#  define B4PACK_RESTRICT __restrict__
#elif defined(_MSC_VER)
#  define B4PACK_RESTRICT __restrict
#else
#  define B4PACK_RESTRICT
#endif

namespace b4support {

void fill_T1_values_packed_slice(
                                        const AmplitudeInputSupportPlan& support,
                                        const PackedAmplitudeSJTensors& sj,
                                        const PackedAmplitudeBoosters& B,
                                        const RangeBlockSupport1D& T1_support,
                                        std::size_t ix,
                                        double* B4PACK_RESTRICT T1_values,
                                        PackedContractionOptions opt)
{
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t K5d = support.channels.k5.size();

    const std::size_t L25 = support.l25.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    assert(ix < support.channels.xvals.size());
    assert(T1_values != nullptr);

    // New pair-product layout:
    //
    //   P5[k5, l25] = sjA(x,l25;k5) * B5(l25,l35,l45;k5)
    //
    // for a fixed (l35,l45), then reuse P5 for every k4:
    //
    //   T1(l45,k4,l35;l25) = sum_k5 sjB(x,k4,l35,l45;k5) * P5[k5,l25].
    //
    // The old scalar-triple-dot layout recomputed sjA*B5 for every k4. The CPU
    // was very brave about it, but bravery is not a performance strategy.
    const std::size_t p5_size = K5d * L25;

    // Use an operation-scale threshold, not merely the number of outer blocks.
    // For the symmetric benchmark L45*B4*L35 is only 13431, below the default
    // threshold, even though the actual dot/axpy work is much larger.
    const std::size_t parallel_work = L45 * L35 * K4d * K5d * L25;

    auto process_l35_l45 = [&](std::size_t il35,
                               std::size_t il45,
                               std::vector<double>& P5) {
        std::fill(P5.begin(), P5.end(), 0.0);

        // Build reusable P5[k5,l25].
        for (std::size_t il25 = 0; il25 < L25; ++il25) {
            const std::size_t sjA_block = idx2(ix, il25, L25);
            const std::size_t B5_block  = idx3(il25, il35, il45, L35, L45);

            const Range1D& r_sjA = support.sjA.blocks[sjA_block];
            const Range1D& r_B5  = support.B5.blocks[B5_block];
            const Range1D r = intersect_2_ranges(r_sjA, r_B5);

            if (r.empty()) {
                continue;
            }

            const double* B4PACK_RESTRICT sjA_base = sj.sjA.data() + r_sjA.offset;
            const double* B4PACK_RESTRICT B5_base  = B.B5.data()  + r_B5.offset;

            for (std::size_t ik5 = r.begin; ik5 < r.end; ++ik5) {
                P5[ik5 * L25 + il25] =
                    sjA_base[ik5 - r_sjA.begin] *
                    B5_base [ik5 - r_B5.begin];
            }
        }

        // Reuse P5 for all k4 rows.
        for (std::size_t ik4 = 0; ik4 < K4d; ++ik4) {
            const std::size_t T1_block = idx3(il45, ik4, il35, K4d, L35);
            const Range1D& r_out = T1_support.blocks[T1_block];

            if (r_out.empty()) {
                continue;
            }

            double* B4PACK_RESTRICT out = T1_values + r_out.offset;
            const std::size_t n_out = static_cast<std::size_t>(r_out.count());
            zero_range(out, n_out);

            const std::size_t sjB_block = idx4(ix, ik4, il35, il45,
                                               K4d, L35, L45);
            const Range1D& r_sjB = support.sjB.blocks[sjB_block];

            if (r_sjB.empty()) {
                continue;
            }

            const double* B4PACK_RESTRICT sjB_base = sj.sjB.data() + r_sjB.offset;

            for (std::size_t ik5 = r_sjB.begin; ik5 < r_sjB.end; ++ik5) {
                const double coeff = sjB_base[ik5 - r_sjB.begin];
                const double* B4PACK_RESTRICT p = P5.data() + ik5 * L25 + r_out.begin;
                axpy_range(coeff, p, out, n_out);
            }
        }
    };

#ifdef _OPENMP
#pragma omp parallel if(opt.use_openmp && parallel_work >= opt.parallel_threshold)
    {
        std::vector<double> P5(p5_size);

#pragma omp for collapse(2) schedule(dynamic, 1)
        for (std::ptrdiff_t il45_s = 0;
             il45_s < static_cast<std::ptrdiff_t>(L45);
             ++il45_s)
        {
            for (std::size_t il35 = 0; il35 < L35; ++il35) {
                process_l35_l45(il35, static_cast<std::size_t>(il45_s), P5);
            }
        }
    }
#else
    std::vector<double> P5(p5_size);
    for (std::size_t il45 = 0; il45 < L45; ++il45) {
        for (std::size_t il35 = 0; il35 < L35; ++il35) {
            process_l35_l45(il35, il45, P5);
        }
    }
#endif
}

void fill_T3_values_packed_slice(
                                        const AmplitudeInputSupportPlan& support,
                                        const PackedAmplitudeSJTensors& sj,
                                        const PackedAmplitudeBoosters& B,
                                        const RangeBlockSupport1D& T3_support,
                                        std::size_t ix,
                                        double* B4PACK_RESTRICT T3_values,
                                        PackedContractionOptions opt)
{
    const std::size_t K2d = support.channels.k2.size();
    const std::size_t K3d = support.channels.k3.size();

    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();

    assert(ix < support.channels.xvals.size());
    assert(T3_values != nullptr);

    // New pair-product layout:
    //
    //   P2[k2, l25] = sjE(x,l25;k2) * B2(l23,l24,l25;k2)
    //
    // for a fixed (l23,l24), then reuse P2 for every k3:
    //
    //   T3(l23,k3,l24;l25) = sum_k2 sjD(x,k3,l23,l24;k2) * P2[k2,l25].
    const std::size_t p2_size = K2d * L25;
    const std::size_t parallel_work = L23 * L24 * K3d * K2d * L25;

    auto process_l23_l24 = [&](std::size_t il23,
                               std::size_t il24,
                               std::vector<double>& P2) {
        std::fill(P2.begin(), P2.end(), 0.0);

        // Build reusable P2[k2,l25].
        for (std::size_t il25 = 0; il25 < L25; ++il25) {
            const std::size_t sjE_block = idx2(ix, il25, L25);
            const std::size_t B2_block  = idx3(il23, il24, il25, L24, L25);

            const Range1D& r_sjE = support.sjE.blocks[sjE_block];
            const Range1D& r_B2  = support.B2.blocks[B2_block];
            const Range1D r = intersect_2_ranges(r_sjE, r_B2);

            if (r.empty()) {
                continue;
            }

            const double* B4PACK_RESTRICT sjE_base = sj.sjE.data() + r_sjE.offset;
            const double* B4PACK_RESTRICT B2_base  = B.B2.data()  + r_B2.offset;

            for (std::size_t ik2 = r.begin; ik2 < r.end; ++ik2) {
                P2[ik2 * L25 + il25] =
                    sjE_base[ik2 - r_sjE.begin] *
                    B2_base [ik2 - r_B2.begin];
            }
        }

        // Reuse P2 for all k3 rows.
        for (std::size_t ik3 = 0; ik3 < K3d; ++ik3) {
            const std::size_t T3_block = idx3(il23, ik3, il24, K3d, L24);
            const Range1D& r_out = T3_support.blocks[T3_block];

            if (r_out.empty()) {
                continue;
            }

            double* B4PACK_RESTRICT out = T3_values + r_out.offset;
            const std::size_t n_out = static_cast<std::size_t>(r_out.count());
            zero_range(out, n_out);

            const std::size_t sjD_block = idx4(ix, ik3, il23, il24,
                                               K3d, L23, L24);
            const Range1D& r_sjD = support.sjD.blocks[sjD_block];

            if (r_sjD.empty()) {
                continue;
            }

            const double* B4PACK_RESTRICT sjD_base = sj.sjD.data() + r_sjD.offset;

            for (std::size_t ik2 = r_sjD.begin; ik2 < r_sjD.end; ++ik2) {
                const double coeff = sjD_base[ik2 - r_sjD.begin];
                const double* B4PACK_RESTRICT p = P2.data() + ik2 * L25 + r_out.begin;
                axpy_range(coeff, p, out, n_out);
            }
        }
    };

#ifdef _OPENMP
#pragma omp parallel if(opt.use_openmp && parallel_work >= opt.parallel_threshold)
    {
        std::vector<double> P2(p2_size);

#pragma omp for collapse(2) schedule(dynamic, 1)
        for (std::ptrdiff_t il23_s = 0;
             il23_s < static_cast<std::ptrdiff_t>(L23);
             ++il23_s)
        {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                process_l23_l24(static_cast<std::size_t>(il23_s), il24, P2);
            }
        }
    }
#else
    std::vector<double> P2(p2_size);
    for (std::size_t il23 = 0; il23 < L23; ++il23) {
        for (std::size_t il24 = 0; il24 < L24; ++il24) {
            process_l23_l24(il23, il24, P2);
        }
    }
#endif
}

} // namespace b4support

#undef B4PACK_RESTRICT
