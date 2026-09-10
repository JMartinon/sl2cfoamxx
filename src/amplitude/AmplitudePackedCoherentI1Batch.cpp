// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/AmplitudePackedCoherentI1Batch.hpp"

namespace b4support {

void fill_T1_values_packed_slice_coherent(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    std::size_t ix,
    b4utils::ComplexBuffer<>& T1_values,
    PackedContractionOptions opt)
{
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t K5d = support.channels.k5.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    assert(ix < support.channels.xvals.size());
    T1_values.resize_discard(std::max<std::size_t>(1, static_cast<std::size_t>(T1_support.packed_values)));

    const std::size_t p5_size = K5d * L25;
    const std::size_t parallel_work = L45 * L35 * K4d * K5d * L25;

    auto process_l35_l45 = [&](std::size_t il35, std::size_t il45,
                               std::vector<double>& P5_re,
                               std::vector<double>& P5_im) {
        std::fill(P5_re.begin(), P5_re.end(), 0.0);
        std::fill(P5_im.begin(), P5_im.end(), 0.0);

        for (std::size_t il25 = 0; il25 < L25; ++il25) {
            const std::size_t sjA_block = idx2(ix, il25, L25);
            const std::size_t B5_block  = idx3(il25, il35, il45, L35, L45);
            const Range1D& r_sjA = support.sjA.blocks[sjA_block];
            const Range1D& r_B5  = support.B5.blocks[B5_block];
            const Range1D r = intersect_2_ranges(r_sjA, r_B5);
            if (r.empty()) continue;

            const double* B4PACK_RESTRICT sjA_base = sj.sjA.data() + r_sjA.offset;
            const double* B4PACK_RESTRICT B5_re = B.B5.re_data() + r_B5.offset;
            const double* B4PACK_RESTRICT B5_im = B.B5.im_data() + r_B5.offset;

            for (std::size_t ik5 = r.begin; ik5 < r.end; ++ik5) {
                const double s = sjA_base[ik5 - r_sjA.begin];
                const std::size_t p = ik5 * L25 + il25;
                P5_re[p] = s * B5_re[ik5 - r_B5.begin];
                P5_im[p] = s * B5_im[ik5 - r_B5.begin];
            }
        }

        for (std::size_t ik4 = 0; ik4 < K4d; ++ik4) {
            const std::size_t T1_block = idx3(il45, ik4, il35, K4d, L35);
            const Range1D& r_out = T1_support.blocks[T1_block];
            if (r_out.empty()) continue;

            double* B4PACK_RESTRICT out_re = T1_values.re_data() + r_out.offset;
            double* B4PACK_RESTRICT out_im = T1_values.im_data() + r_out.offset;
            const std::size_t n_out = r_out.count();
            qlocal_zero_complex_row(out_re, out_im, n_out);

            const std::size_t sjB_block = idx4(ix, ik4, il35, il45, K4d, L35, L45);
            const Range1D& r_sjB = support.sjB.blocks[sjB_block];
            if (r_sjB.empty()) continue;

            const double* B4PACK_RESTRICT sjB_base = sj.sjB.data() + r_sjB.offset;
            for (std::size_t ik5 = r_sjB.begin; ik5 < r_sjB.end; ++ik5) {
                const double coeff = sjB_base[ik5 - r_sjB.begin];
                const double* B4PACK_RESTRICT p_re = P5_re.data() + ik5 * L25 + r_out.begin;
                const double* B4PACK_RESTRICT p_im = P5_im.data() + ik5 * L25 + r_out.begin;
                qlocal_axpy_real_to_complex(coeff, p_re, p_im, out_re, out_im, n_out);
            }
        }
    };

#ifdef _OPENMP
#pragma omp parallel if(opt.use_openmp && parallel_work >= opt.parallel_threshold)
    {
        std::vector<double> P5_re(p5_size), P5_im(p5_size);
#pragma omp for collapse(2) schedule(dynamic, 1)
        for (std::ptrdiff_t il45_s = 0; il45_s < static_cast<std::ptrdiff_t>(L45); ++il45_s) {
            for (std::size_t il35 = 0; il35 < L35; ++il35) {
                process_l35_l45(il35, static_cast<std::size_t>(il45_s), P5_re, P5_im);
            }
        }
    }
#else
    std::vector<double> P5_re(p5_size), P5_im(p5_size);
    for (std::size_t il45 = 0; il45 < L45; ++il45) {
        for (std::size_t il35 = 0; il35 < L35; ++il35) process_l35_l45(il35, il45, P5_re, P5_im);
    }
#endif
}

void fill_T3_values_packed_slice_coherent(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T3_support,
    std::size_t ix,
    b4utils::ComplexBuffer<>& T3_values,
    PackedContractionOptions opt)
{
    const std::size_t K2d = support.channels.k2.size();
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();

    assert(ix < support.channels.xvals.size());
    T3_values.resize_discard(std::max<std::size_t>(1, static_cast<std::size_t>(T3_support.packed_values)));

    const std::size_t p2_size = K2d * L25;
    const std::size_t parallel_work = L23 * L24 * K3d * K2d * L25;

    auto process_l23_l24 = [&](std::size_t il23, std::size_t il24,
                               std::vector<double>& P2_re,
                               std::vector<double>& P2_im) {
        std::fill(P2_re.begin(), P2_re.end(), 0.0);
        std::fill(P2_im.begin(), P2_im.end(), 0.0);

        for (std::size_t il25 = 0; il25 < L25; ++il25) {
            const std::size_t sjE_block = idx2(ix, il25, L25);
            const std::size_t B2_block  = idx3(il23, il24, il25, L24, L25);
            const Range1D& r_sjE = support.sjE.blocks[sjE_block];
            const Range1D& r_B2  = support.B2.blocks[B2_block];
            const Range1D r = intersect_2_ranges(r_sjE, r_B2);
            if (r.empty()) continue;

            const double* B4PACK_RESTRICT sjE_base = sj.sjE.data() + r_sjE.offset;
            const double* B4PACK_RESTRICT B2_re = B.B2.re_data() + r_B2.offset;
            const double* B4PACK_RESTRICT B2_im = B.B2.im_data() + r_B2.offset;

            for (std::size_t ik2 = r.begin; ik2 < r.end; ++ik2) {
                const double s = sjE_base[ik2 - r_sjE.begin];
                const std::size_t p = ik2 * L25 + il25;
                P2_re[p] = s * B2_re[ik2 - r_B2.begin];
                P2_im[p] = s * B2_im[ik2 - r_B2.begin];
            }
        }

        for (std::size_t ik3 = 0; ik3 < K3d; ++ik3) {
            const std::size_t T3_block = idx3(il23, ik3, il24, K3d, L24);
            const Range1D& r_out = T3_support.blocks[T3_block];
            if (r_out.empty()) continue;

            double* B4PACK_RESTRICT out_re = T3_values.re_data() + r_out.offset;
            double* B4PACK_RESTRICT out_im = T3_values.im_data() + r_out.offset;
            const std::size_t n_out = r_out.count();
            qlocal_zero_complex_row(out_re, out_im, n_out);

            const std::size_t sjD_block = idx4(ix, ik3, il23, il24, K3d, L23, L24);
            const Range1D& r_sjD = support.sjD.blocks[sjD_block];
            if (r_sjD.empty()) continue;

            const double* B4PACK_RESTRICT sjD_base = sj.sjD.data() + r_sjD.offset;
            for (std::size_t ik2 = r_sjD.begin; ik2 < r_sjD.end; ++ik2) {
                const double coeff = sjD_base[ik2 - r_sjD.begin];
                const double* B4PACK_RESTRICT p_re = P2_re.data() + ik2 * L25 + r_out.begin;
                const double* B4PACK_RESTRICT p_im = P2_im.data() + ik2 * L25 + r_out.begin;
                qlocal_axpy_real_to_complex(coeff, p_re, p_im, out_re, out_im, n_out);
            }
        }
    };

#ifdef _OPENMP
#pragma omp parallel if(opt.use_openmp && parallel_work >= opt.parallel_threshold)
    {
        std::vector<double> P2_re(p2_size), P2_im(p2_size);
#pragma omp for collapse(2) schedule(dynamic, 1)
        for (std::ptrdiff_t il23_s = 0; il23_s < static_cast<std::ptrdiff_t>(L23); ++il23_s) {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                process_l23_l24(static_cast<std::size_t>(il23_s), il24, P2_re, P2_im);
            }
        }
    }
#else
    std::vector<double> P2_re(p2_size), P2_im(p2_size);
    for (std::size_t il23 = 0; il23 < L23; ++il23) {
        for (std::size_t il24 = 0; il24 < L24; ++il24) process_l23_l24(il23, il24, P2_re, P2_im);
    }
#endif
}

void build_T1_support_qlocal_slice_coherent_lane_into(
    const AmplitudeInputSupportPlan& common,
    const RangeBlockSupport1D& sjA,
    std::size_t ix,
    RangeBlockSupport1D& T1,
    PackedContractionOptions opt)
{
    const std::size_t K4d = common.channels.k4.size();

    const std::size_t L25 = common.l25.size();
    const std::size_t L35 = common.l35.size();
    const std::size_t L45 = common.l45.size();

    assert(ix < common.channels.xvals.size());
    assert(T1.blocks.size() == L45 * K4d * L35);

    const std::size_t nblocks = L45 * K4d * L35;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) \
    if(opt.use_openmp && nblocks >= opt.parallel_threshold)
#endif
    for (std::ptrdiff_t il45_s = 0;
         il45_s < static_cast<std::ptrdiff_t>(L45);
         ++il45_s)
    {
        for (std::size_t ik4 = 0; ik4 < K4d; ++ik4) {
            for (std::size_t il35 = 0; il35 < L35; ++il35) {
                const std::size_t il45 = static_cast<std::size_t>(il45_s);

                const std::size_t T1_block = idx3(il45, ik4, il35, K4d, L35);
                const std::size_t sjB_block = idx4(ix, ik4, il35, il45, K4d, L35, L45);
                const Range1D& r_sjB = common.sjB.blocks[sjB_block];

                if (r_sjB.empty()) {
                    T1.blocks[T1_block] = empty_range_1d();
                    continue;
                }

                CoverRangeBuilder cover;

                for (std::size_t il25 = 0; il25 < L25; ++il25) {
                    const std::size_t sjA_block = idx2(ix, il25, L25);
                    const std::size_t B5_block  = idx3(il25, il35, il45, L35, L45);
                    const Range1D& r_sjA = sjA.blocks[sjA_block];
                    const Range1D& r_B5  = common.B5.blocks[B5_block];

                    if (ranges_intersect(r_sjA, r_sjB, r_B5)) {
                        cover.add(il25);
                    }
                }

                T1.blocks[T1_block] = cover.finish();
            }
        }
    }

    finalize_offsets(T1);
}

void build_T3_support_qlocal_slice_coherent_lane_into(
    const AmplitudeInputSupportPlan& common,
    const RangeBlockSupport1D& sjE,
    std::size_t ix,
    RangeBlockSupport1D& T3,
    PackedContractionOptions opt)
{
    const std::size_t K3d = common.channels.k3.size();

    const std::size_t L23 = common.l23.size();
    const std::size_t L24 = common.l24.size();
    const std::size_t L25 = common.l25.size();

    assert(ix < common.channels.xvals.size());
    assert(T3.blocks.size() == L23 * K3d * L24);

    const std::size_t nblocks = L23 * K3d * L24;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) \
    if(opt.use_openmp && nblocks >= opt.parallel_threshold)
#endif
    for (std::ptrdiff_t il23_s = 0;
         il23_s < static_cast<std::ptrdiff_t>(L23);
         ++il23_s)
    {
        for (std::size_t ik3 = 0; ik3 < K3d; ++ik3) {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                const std::size_t il23 = static_cast<std::size_t>(il23_s);

                const std::size_t T3_block = idx3(il23, ik3, il24, K3d, L24);
                const std::size_t sjD_block = idx4(ix, ik3, il23, il24, K3d, L23, L24);
                const Range1D& r_sjD = common.sjD.blocks[sjD_block];

                if (r_sjD.empty()) {
                    T3.blocks[T3_block] = empty_range_1d();
                    continue;
                }

                CoverRangeBuilder cover;

                for (std::size_t il25 = 0; il25 < L25; ++il25) {
                    const std::size_t sjE_block = idx2(ix, il25, L25);
                    const std::size_t B2_block  = idx3(il23, il24, il25, L24, L25);
                    const Range1D& r_sjE = sjE.blocks[sjE_block];
                    const Range1D& r_B2  = common.B2.blocks[B2_block];

                    if (ranges_intersect(r_sjE, r_sjD, r_B2)) {
                        cover.add(il25);
                    }
                }

                T3.blocks[T3_block] = cover.finish();
            }
        }
    }

    finalize_offsets(T3);
}

void fill_T1_values_packed_slice_coherent_lane(
    const AmplitudeInputSupportPlan& common,
    const RangeBlockSupport1D& sjA,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    std::size_t ix,
    b4utils::ComplexBuffer<>& T1_values,
    PackedContractionOptions opt)
{
    const std::size_t K4d = common.channels.k4.size();
    const std::size_t K5d = common.channels.k5.size();
    const std::size_t L25 = common.l25.size();
    const std::size_t L35 = common.l35.size();
    const std::size_t L45 = common.l45.size();

    assert(ix < common.channels.xvals.size());
    T1_values.resize_discard(std::max<std::size_t>(1, static_cast<std::size_t>(T1_support.packed_values)));

    const std::size_t p5_size = K5d * L25;
    const std::size_t parallel_work = L45 * L35 * K4d * K5d * L25;

    auto process_l35_l45 = [&](std::size_t il35, std::size_t il45,
                               std::vector<double>& P5_re,
                               std::vector<double>& P5_im) {
        std::fill(P5_re.begin(), P5_re.end(), 0.0);
        std::fill(P5_im.begin(), P5_im.end(), 0.0);

        for (std::size_t il25 = 0; il25 < L25; ++il25) {
            const std::size_t sjA_block = idx2(ix, il25, L25);
            const std::size_t B5_block  = idx3(il25, il35, il45, L35, L45);
            const Range1D& r_sjA = sjA.blocks[sjA_block];
            const Range1D& r_B5  = common.B5.blocks[B5_block];
            const Range1D r = intersect_2_ranges(r_sjA, r_B5);
            if (r.empty()) continue;

            const double* B4PACK_RESTRICT sjA_base = sj.sjA.data() + r_sjA.offset;
            const double* B4PACK_RESTRICT B5_re = B.B5.re_data() + r_B5.offset;
            const double* B4PACK_RESTRICT B5_im = B.B5.im_data() + r_B5.offset;

            for (std::size_t ik5 = r.begin; ik5 < r.end; ++ik5) {
                const double s = sjA_base[ik5 - r_sjA.begin];
                const std::size_t p = ik5 * L25 + il25;
                P5_re[p] = s * B5_re[ik5 - r_B5.begin];
                P5_im[p] = s * B5_im[ik5 - r_B5.begin];
            }
        }

        for (std::size_t ik4 = 0; ik4 < K4d; ++ik4) {
            const std::size_t T1_block = idx3(il45, ik4, il35, K4d, L35);
            const Range1D& r_out = T1_support.blocks[T1_block];
            if (r_out.empty()) continue;

            double* B4PACK_RESTRICT out_re = T1_values.re_data() + r_out.offset;
            double* B4PACK_RESTRICT out_im = T1_values.im_data() + r_out.offset;
            const std::size_t n_out = r_out.count();
            qlocal_zero_complex_row(out_re, out_im, n_out);

            const std::size_t sjB_block = idx4(ix, ik4, il35, il45, K4d, L35, L45);
            const Range1D& r_sjB = common.sjB.blocks[sjB_block];
            if (r_sjB.empty()) continue;

            const double* B4PACK_RESTRICT sjB_base = sj.sjB.data() + r_sjB.offset;
            for (std::size_t ik5 = r_sjB.begin; ik5 < r_sjB.end; ++ik5) {
                const double coeff = sjB_base[ik5 - r_sjB.begin];
                const double* B4PACK_RESTRICT p_re = P5_re.data() + ik5 * L25 + r_out.begin;
                const double* B4PACK_RESTRICT p_im = P5_im.data() + ik5 * L25 + r_out.begin;
                qlocal_axpy_real_to_complex(coeff, p_re, p_im, out_re, out_im, n_out);
            }
        }
    };

#ifdef _OPENMP
#pragma omp parallel if(opt.use_openmp && parallel_work >= opt.parallel_threshold)
    {
        std::vector<double> P5_re(p5_size), P5_im(p5_size);
#pragma omp for collapse(2) schedule(dynamic, 1)
        for (std::ptrdiff_t il45_s = 0; il45_s < static_cast<std::ptrdiff_t>(L45); ++il45_s) {
            for (std::size_t il35 = 0; il35 < L35; ++il35) {
                process_l35_l45(il35, static_cast<std::size_t>(il45_s), P5_re, P5_im);
            }
        }
    }
#else
    std::vector<double> P5_re(p5_size), P5_im(p5_size);
    for (std::size_t il45 = 0; il45 < L45; ++il45) {
        for (std::size_t il35 = 0; il35 < L35; ++il35) process_l35_l45(il35, il45, P5_re, P5_im);
    }
#endif
}

void fill_T3_values_packed_slice_coherent_lane(
    const AmplitudeInputSupportPlan& common,
    const RangeBlockSupport1D& sjE,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T3_support,
    std::size_t ix,
    b4utils::ComplexBuffer<>& T3_values,
    PackedContractionOptions opt)
{
    const std::size_t K2d = common.channels.k2.size();
    const std::size_t K3d = common.channels.k3.size();
    const std::size_t L23 = common.l23.size();
    const std::size_t L24 = common.l24.size();
    const std::size_t L25 = common.l25.size();

    assert(ix < common.channels.xvals.size());
    T3_values.resize_discard(std::max<std::size_t>(1, static_cast<std::size_t>(T3_support.packed_values)));

    const std::size_t p2_size = K2d * L25;
    const std::size_t parallel_work = L23 * L24 * K3d * K2d * L25;

    auto process_l23_l24 = [&](std::size_t il23, std::size_t il24,
                               std::vector<double>& P2_re,
                               std::vector<double>& P2_im) {
        std::fill(P2_re.begin(), P2_re.end(), 0.0);
        std::fill(P2_im.begin(), P2_im.end(), 0.0);

        for (std::size_t il25 = 0; il25 < L25; ++il25) {
            const std::size_t sjE_block = idx2(ix, il25, L25);
            const std::size_t B2_block  = idx3(il23, il24, il25, L24, L25);
            const Range1D& r_sjE = sjE.blocks[sjE_block];
            const Range1D& r_B2  = common.B2.blocks[B2_block];
            const Range1D r = intersect_2_ranges(r_sjE, r_B2);
            if (r.empty()) continue;

            const double* B4PACK_RESTRICT sjE_base = sj.sjE.data() + r_sjE.offset;
            const double* B4PACK_RESTRICT B2_re = B.B2.re_data() + r_B2.offset;
            const double* B4PACK_RESTRICT B2_im = B.B2.im_data() + r_B2.offset;

            for (std::size_t ik2 = r.begin; ik2 < r.end; ++ik2) {
                const double s = sjE_base[ik2 - r_sjE.begin];
                const std::size_t p = ik2 * L25 + il25;
                P2_re[p] = s * B2_re[ik2 - r_B2.begin];
                P2_im[p] = s * B2_im[ik2 - r_B2.begin];
            }
        }

        for (std::size_t ik3 = 0; ik3 < K3d; ++ik3) {
            const std::size_t T3_block = idx3(il23, ik3, il24, K3d, L24);
            const Range1D& r_out = T3_support.blocks[T3_block];
            if (r_out.empty()) continue;

            double* B4PACK_RESTRICT out_re = T3_values.re_data() + r_out.offset;
            double* B4PACK_RESTRICT out_im = T3_values.im_data() + r_out.offset;
            const std::size_t n_out = r_out.count();
            qlocal_zero_complex_row(out_re, out_im, n_out);

            const std::size_t sjD_block = idx4(ix, ik3, il23, il24, K3d, L23, L24);
            const Range1D& r_sjD = common.sjD.blocks[sjD_block];
            if (r_sjD.empty()) continue;

            const double* B4PACK_RESTRICT sjD_base = sj.sjD.data() + r_sjD.offset;
            for (std::size_t ik2 = r_sjD.begin; ik2 < r_sjD.end; ++ik2) {
                const double coeff = sjD_base[ik2 - r_sjD.begin];
                const double* B4PACK_RESTRICT p_re = P2_re.data() + ik2 * L25 + r_out.begin;
                const double* B4PACK_RESTRICT p_im = P2_im.data() + ik2 * L25 + r_out.begin;
                qlocal_axpy_real_to_complex(coeff, p_re, p_im, out_re, out_im, n_out);
            }
        }
    };

#ifdef _OPENMP
#pragma omp parallel if(opt.use_openmp && parallel_work >= opt.parallel_threshold)
    {
        std::vector<double> P2_re(p2_size), P2_im(p2_size);
#pragma omp for collapse(2) schedule(dynamic, 1)
        for (std::ptrdiff_t il23_s = 0; il23_s < static_cast<std::ptrdiff_t>(L23); ++il23_s) {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                process_l23_l24(static_cast<std::size_t>(il23_s), il24, P2_re, P2_im);
            }
        }
    }
#else
    std::vector<double> P2_re(p2_size), P2_im(p2_size);
    for (std::size_t il23 = 0; il23 < L23; ++il23) {
        for (std::size_t il24 = 0; il24 < L24; ++il24) process_l23_l24(il23, il24, P2_re, P2_im);
    }
#endif
}

Range1D build_T2_panel_for_q_coherent(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const PackedAmplitudeCoherentBoosters& B,
    std::size_t ik4_begin,
    std::size_t nk4,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4PACK_RESTRICT T2_re,
    double* B4PACK_RESTRICT T2_im,
    Range1D* B4PACK_RESTRICT T2_ranges,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();
    Range1D panel_cover = empty_range_1d();

    for (std::size_t jb = 0; jb < nk4; ++jb) {
        const std::size_t ik4 = ik4_begin + jb;
        double* row_re = T2_re + jb * L25;
        double* row_im = T2_im + jb * L25;
        qlocal_zero_complex_row(row_re, row_im, L25);
        Range1D row_range = empty_range_1d();

        const std::size_t B4_block = idx3(ik4, il24, il34, L24, L34);
        const Range1D& r_B4 = support.B4.blocks[B4_block];
        if (!r_B4.empty()) {
            const double* B4_re = B.B4.re_data() + r_B4.offset;
            const double* B4_im = B.B4.im_data() + r_B4.offset;
            for (std::size_t il45 = r_B4.begin; il45 < r_B4.end; ++il45) {
                const std::size_t T1_block = idx3(il45, ik4, il35, K4d, L35);
                const Range1D& r_T1 = T1_support.blocks[T1_block];
                if (r_T1.empty()) continue;
                const double ar = B4_re[il45 - r_B4.begin];
                const double ai = B4_im[il45 - r_B4.begin];
                const double* src_re = T1_values.re_data() + r_T1.offset;
                const double* src_im = T1_values.im_data() + r_T1.offset;
                qlocal_axpy_complex_to_complex(ar, ai, src_re, src_im,
                                               row_re + r_T1.begin,
                                               row_im + r_T1.begin,
                                               r_T1.count());
                qlocal_cover_range_with_range(row_range, r_T1);
            }
        }
        T2_ranges[jb] = row_range;
        if (!row_range.empty()) {
            qlocal_cover_range_with_range(panel_cover, row_range);
            if (counters) ++counters->T2_rows_built;
        }
    }
    return panel_cover;
}

Range1D build_T4_panel_for_q_coherent(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    const PackedAmplitudeCoherentBoosters& B,
    std::size_t ik3_begin,
    std::size_t nk3,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4PACK_RESTRICT T4_re,
    double* B4PACK_RESTRICT T4_im,
    Range1D* B4PACK_RESTRICT T4_ranges,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    Range1D panel_cover = empty_range_1d();

    for (std::size_t ib = 0; ib < nk3; ++ib) {
        const std::size_t ik3 = ik3_begin + ib;
        double* row_re = T4_re + ib * L25;
        double* row_im = T4_im + ib * L25;
        qlocal_zero_complex_row(row_re, row_im, L25);
        Range1D row_range = empty_range_1d();

        const std::size_t B3_block = idx3(ik3, il34, il35, L34, L35);
        const Range1D& r_B3 = support.B3.blocks[B3_block];
        if (!r_B3.empty()) {
            const double* B3_re = B.B3.re_data() + r_B3.offset;
            const double* B3_im = B.B3.im_data() + r_B3.offset;
            for (std::size_t il23 = r_B3.begin; il23 < r_B3.end; ++il23) {
                const std::size_t T3_block = idx3(il23, ik3, il24, K3d, L24);
                const Range1D& r_T3 = T3_support.blocks[T3_block];
                if (r_T3.empty()) continue;
                const double ar = B3_re[il23 - r_B3.begin];
                const double ai = B3_im[il23 - r_B3.begin];
                const double* src_re = T3_values.re_data() + r_T3.offset;
                const double* src_im = T3_values.im_data() + r_T3.offset;
                qlocal_axpy_complex_to_complex(ar, ai, src_re, src_im,
                                               row_re + r_T3.begin,
                                               row_im + r_T3.begin,
                                               r_T3.count());
                qlocal_cover_range_with_range(row_range, r_T3);
            }
        }
        T4_ranges[ib] = row_range;
        if (!row_range.empty()) {
            qlocal_cover_range_with_range(panel_cover, row_range);
            if (counters) ++counters->T4_rows_built;
        }
    }
    return panel_cover;
}

b4utils::Complex64 contract_q_block_project_C_times_T2_streamed_coherent(
    const AmplitudeInputSupportPlan& support,
    const double* C_base,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalComplexThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);
    (void)idx4(ix, il24, il34, il35, L24, L34, L35);

    double* T2_re = ws.T2_re.data(); double* T2_im = ws.T2_im.data();
    double* T4_re = ws.T4_re.data(); double* T4_im = ws.T4_im.data();
    double* W_re  = ws.work_re.data(); double* W_im  = ws.work_im.data();

    Range1D* T2_ranges = ws.T2_ranges.data();
    Range1D* T4_ranges = ws.T4_ranges.data();

    const Range1D T2_cover = build_T2_panel_for_q_coherent(
        support, T1_support, T1_values, B, rC.j_begin, nk4,
        il24, il34, il35, T2_re, T2_im, T2_ranges, counters);
    if (T2_cover.empty()) return {0.0, 0.0};

    const Range1D T4_cover = build_T4_panel_for_q_coherent(
        support, T3_support, T3_values, B, rC.i_begin, nk3,
        il24, il34, il35, T4_re, T4_im, T4_ranges, counters);
    if (T4_cover.empty()) return {0.0, 0.0};

    qlocal_zero_complex_row(W_re, W_im, nk3 * L25);

    for (std::size_t jb = 0; jb < nk4; ++jb) {
        const Range1D& r_T2 = T2_ranges[jb];
        if (r_T2.empty()) continue;
        const double* row_re = T2_re + jb * L25;
        const double* row_im = T2_im + jb * L25;
        const double* C_col = C_base + nk3 * jb;
        for (std::size_t ib = 0; ib < nk3; ++ib) {
            const double c = C_col[ib];
            qlocal_axpy_real_to_complex(c,
                                        row_re + r_T2.begin,
                                        row_im + r_T2.begin,
                                        W_re + ib * L25 + r_T2.begin,
                                        W_im + ib * L25 + r_T2.begin,
                                        r_T2.count());
        }
    }

    const Range1D active_cover = intersect_2_ranges(T2_cover, T4_cover);
    if (active_cover.empty()) return {0.0, 0.0};

    b4utils::Complex64 local{0.0, 0.0};
    for (std::size_t ib = 0; ib < nk3; ++ib) {
        const Range1D& r_T4 = T4_ranges[ib];
        if (r_T4.empty()) continue;
        const Range1D r = intersect_2_ranges(active_cover, r_T4);
        if (r.empty()) continue;
        const b4utils::Complex64 z = qlocal_dot_complex_no_conj(
            W_re + ib * L25 + r.begin, W_im + ib * L25 + r.begin,
            T4_re + ib * L25 + r.begin, T4_im + ib * L25 + r.begin,
            r.count());
        local.re += z.re;
        local.im += z.im;
    }
    return local;
}

b4utils::Complex64 contract_q_block_project_Ct_times_T4_streamed_coherent(
    const AmplitudeInputSupportPlan& support,
    const double* C_base,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalComplexThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);
    (void)idx4(ix, il24, il34, il35, L24, L34, L35);

    double* T2_re = ws.T2_re.data(); double* T2_im = ws.T2_im.data();
    double* T4_re = ws.T4_re.data(); double* T4_im = ws.T4_im.data();
    double* V_re  = ws.work_re.data(); double* V_im  = ws.work_im.data();
    Range1D* T2_ranges = ws.T2_ranges.data();
    Range1D* T4_ranges = ws.T4_ranges.data();

    const Range1D T4_cover = build_T4_panel_for_q_coherent(
        support, T3_support, T3_values, B, rC.i_begin, nk3,
        il24, il34, il35, T4_re, T4_im, T4_ranges, counters);
    if (T4_cover.empty()) return {0.0, 0.0};

    const Range1D T2_cover = build_T2_panel_for_q_coherent(
        support, T1_support, T1_values, B, rC.j_begin, nk4,
        il24, il34, il35, T2_re, T2_im, T2_ranges, counters);
    if (T2_cover.empty()) return {0.0, 0.0};

    qlocal_zero_complex_row(V_re, V_im, nk4 * L25);

    for (std::size_t ib = 0; ib < nk3; ++ib) {
        const Range1D& r_T4 = T4_ranges[ib];
        if (r_T4.empty()) continue;
        const double* row_re = T4_re + ib * L25;
        const double* row_im = T4_im + ib * L25;
        for (std::size_t jb = 0; jb < nk4; ++jb) {
            const double c = C_base[ib + nk3 * jb];
            qlocal_axpy_real_to_complex(c,
                                        row_re + r_T4.begin,
                                        row_im + r_T4.begin,
                                        V_re + jb * L25 + r_T4.begin,
                                        V_im + jb * L25 + r_T4.begin,
                                        r_T4.count());
        }
    }

    const Range1D active_cover = intersect_2_ranges(T4_cover, T2_cover);
    if (active_cover.empty()) return {0.0, 0.0};

    b4utils::Complex64 local{0.0, 0.0};
    for (std::size_t jb = 0; jb < nk4; ++jb) {
        const Range1D& r_T2 = T2_ranges[jb];
        if (r_T2.empty()) continue;
        const Range1D r = intersect_2_ranges(active_cover, r_T2);
        if (r.empty()) continue;
        const b4utils::Complex64 z = qlocal_dot_complex_no_conj(
            T2_re + jb * L25 + r.begin, T2_im + jb * L25 + r.begin,
            V_re + jb * L25 + r.begin,  V_im + jb * L25 + r.begin,
            r.count());
        local.re += z.re;
        local.im += z.im;
    }
    return local;
}

b4utils::Complex64 contract_qlocal_streamed_slice_allq_half_sjC_coherent(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    std::size_t ix,
    PackedContractionOptions opt,
    PackedQLocalContractionStats* stats)
{
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();
    assert(ix < support.channels.xvals.size());
    const std::size_t nq = L24 * L34 * L35;
    double slice_re = 0.0;
    double slice_im = 0.0;
    std::mutex stats_mutex;

#ifdef _OPENMP
#pragma omp parallel reduction(+:slice_re,slice_im) if(opt.use_openmp && nq * K3d * K4d >= opt.parallel_threshold)
#endif
    {
        PackedQLocalComplexThreadWorkspace ws;
        ws.ensure(K3d, K4d, L25);
        std::vector<double> Ctmp;
        PackedQLocalThreadCounters local_counters;

#ifdef _OPENMP
#pragma omp for collapse(2) schedule(dynamic, 1)
#endif
        for (std::ptrdiff_t il24_s = 0; il24_s < static_cast<std::ptrdiff_t>(L24); ++il24_s) {
            for (std::size_t il34 = 0; il34 < L34; ++il34) {
                const std::size_t il24 = static_cast<std::size_t>(il24_s);
                for (std::size_t il35 = 0; il35 < L35; ++il35) {
                    const std::size_t C_block = idx4(ix, il24, il34, il35, L24, L34, L35);
                    const AmplitudeSjCHalfBlockMap& map = support.sjC_map[C_block];
                    if (map.empty()) continue;
                    const Range2D& logical = map.logical;
                    if (logical.empty()) continue;

                    // Q-slabs own logical blocks. Their real 6j panels may
                    // share reflected storage, but the projected coherent
                    // boosters still require each logical contraction.
                    const double* C_base = nullptr;
                    const std::size_t storage = static_cast<std::size_t>(map.storage_block);
                    if (!map.transpose_from_storage && storage < support.sjC.blocks.size()) {
                        const Range2D& stored = support.sjC.blocks[storage];
                        if (!stored.empty() && stored.i_begin == logical.i_begin &&
                            stored.i_end == logical.i_end && stored.j_begin == logical.j_begin &&
                            stored.j_end == logical.j_end) {
                            C_base = sj.sjC.data() + stored.offset;
                        }
                    }
                    if (C_base == nullptr) {
                        // Reserve once on first use. The low-memory planner
                        // admits this bounded panel for each possible worker.
                        if (Ctmp.capacity() == 0) Ctmp.reserve(K3d * K4d);
                        materialize_logical_sjC_panel_half(support, sj, C_block, logical, Ctmp);
                        C_base = Ctmp.data();
                    }
                    const b4utils::Complex64 q = contract_q_block_streamed_adaptive_coherent(
                        support, C_base, B, T1_support, T1_values, T3_support, T3_values,
                        ix, il24, il34, il35, logical, ws,
                        stats ? &local_counters : nullptr);
                    slice_re += q.re;
                    slice_im += q.im;
                }
            }
        }

        if (stats != nullptr) {
            const std::lock_guard lock(stats_mutex);
            local_counters.add_to(*stats);
        }
    }
    return {slice_re, slice_im};
}

std::vector<PackedAmplitudeQSlab> choose_q_slabs_for_xslice_half_sjC_coherent(
    const AmplitudeInputSupportPlan_HalfSjC& xslice_full_support,
    const AmplitudeInputSupportPlan_HalfSjC& wing_support,
    const RangeBlockSupport1D& T1,
    const RangeBlockSupport1D& T3,
    const PackedAmplitudeXQSlabOptions& opt,
    std::uint64_t fixed_bytes)
{
    const std::size_t Q = amplitude_q_count_half_sjC(xslice_full_support);
    const std::uint64_t budget = xqslab_memory_budget(opt);
    const std::uint64_t wing_bytes = xqslab_wing_payload_bytes(wing_support);
    // Coherent T1/T3 payload is split-complex, so double the real estimate.
    const std::uint64_t T_bytes = xqslab_sat_mul_u64(2ull, xqslab_T1T3_payload_bytes(T1, T3));
    const std::uint64_t q_metadata = xslice_full_support.sjC.metadata_bytes()
        + static_cast<std::uint64_t>(xslice_full_support.sjC_map.size() * sizeof(AmplitudeSjCHalfBlockMap));
    std::uint64_t base_variable = 0;
    base_variable = xqslab_sat_add_u64(base_variable, wing_bytes);
    base_variable = xqslab_sat_add_u64(base_variable, T_bytes);
    base_variable = xqslab_sat_add_u64(base_variable, q_metadata);

    std::vector<PackedAmplitudeQSlab> slabs;
    slabs.reserve(Q == 0 ? 0 : std::min<std::size_t>(Q, 1024));
    std::vector<int> seen_epoch(Q, 0);
    int epoch = 1;
    std::size_t q = 0;
    while (q < Q) {
        if (epoch == std::numeric_limits<int>::max()) {
            std::fill(seen_epoch.begin(), seen_epoch.end(), 0);
            epoch = 1;
        }
        std::size_t q_end = q;
        std::uint64_t c_values = 0;
        bool added_any = false;
        while (q_end < Q && (q_end - q) < opt.max_q_per_slab) {
            std::size_t il24 = 0, il34 = 0, il35 = 0;
            amplitude_decode_q_half_sjC(xslice_full_support, q_end, il24, il34, il35);
            const std::size_t logical_block = idx4(0u, il24, il34, il35,
                                                   xslice_full_support.l24.size(),
                                                   xslice_full_support.l34.size(),
                                                   xslice_full_support.l35.size());
            const AmplitudeSjCHalfBlockMap& map = xslice_full_support.sjC_map[logical_block];
            std::uint64_t add_values = 0;
            if (!map.empty()) {
                const std::size_t storage = static_cast<std::size_t>(map.storage_block);
                if (storage < Q && seen_epoch[storage] != epoch) {
                    add_values = xslice_full_support.sjC.blocks[storage].count();
                }
            }
            const std::uint64_t trial_values = xqslab_sat_add_u64(c_values, add_values);
            std::uint64_t trial_var = xqslab_sat_add_u64(base_variable, xqslab_bytes_for_values(trial_values));
            trial_var = xqslab_add_fractional_overhead(trial_var, opt.variable_overhead_fraction);
            const std::uint64_t trial_peak = xqslab_sat_add_u64(fixed_bytes, trial_var);
            if (added_any && trial_peak > budget) break;
            if (!map.empty()) {
                const std::size_t storage = static_cast<std::size_t>(map.storage_block);
                if (storage < Q && seen_epoch[storage] != epoch) {
                    seen_epoch[storage] = epoch;
                    c_values = trial_values;
                }
            }
            ++q_end;
            added_any = true;
            if (trial_peak > budget) break;
        }
        if (!added_any) q_end = q + 1;
        std::uint64_t var = xqslab_sat_add_u64(base_variable, xqslab_bytes_for_values(c_values));
        var = xqslab_add_fractional_overhead(var, opt.variable_overhead_fraction);
        const std::uint64_t peak = xqslab_sat_add_u64(fixed_bytes, var);
        slabs.push_back(PackedAmplitudeQSlab{q, q_end, xqslab_bytes_for_values(c_values), var, peak, peak <= budget});
        q = q_end;
        ++epoch;
    }
    return slabs;
}

PackedAmplitudeCoherentXQResult contract_packed_amplitude_coherent_xqslab_streamed_generated_support(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const std::array<int, 10>& spins,
    const PackedAmplitudeXQSlabOptions& opt,
    WingExecutionOrder order,
    PackedContractionOptions contraction_options,
    std::ostream* log)
{
    using clock = std::chrono::high_resolution_clock;
    const auto total_t0 = clock::now();
    PackedAmplitudeCoherentXQResult result;
    result.stats.clear();
    result.x_slices = static_support.channels.xvals.size();
    const B4Spins b4spins(spins);
    const std::uint64_t fixed_bytes = xqslab_fixed_bytes_coherent(
        static_support, packed_B, opt, contraction_options);

    b4utils::Complex64 amplitude{0.0, 0.0};
    for (std::size_t ix = 0; ix < static_support.channels.xvals.size(); ++ix) {
        if (log && opt.print_x_timing) {
            *log << "\n[coherent xqslab] x slice " << ix
                 << " (Dx=" << static_support.channels.xvals[ix] << ")\n";
        }
        b4utils::Complex64 x_amp{0.0, 0.0};
        std::size_t x_q_slabs = 0;
        std::size_t x_unfit = 0;
        auto release_t0 = clock::now();
        {
            auto t0 = clock::now();
            AmplitudeInputSupportPlan_HalfSjC xslice_full_support =
                build_amplitude_xslice_full_support_half_sjC(static_support, b4spins, ix);
            AmplitudeInputSupportPlan_HalfSjC wing_support = xslice_full_support;
            amplitude_half_sjC_clear_physical_storage(wing_support);
            result.timings.xsupport += xqslab_seconds_since(t0);

            t0 = clock::now();
            PackedAmplitudeSJTensors_HalfSjC sj;
            fill_packed_amplitude_sj_tensors_rec_half_sjC(b4spins, wing_support, sj);
            result.timings.wing_fill += xqslab_seconds_since(t0);

            PackedQLocalSupportWorkspace sws;
            sws.initialize(wing_support);
            PackedQLocalComplexValueWorkspace vws;

            t0 = clock::now();
            if (order == WingExecutionOrder::LeftThenRight) {
                build_T3_support_qlocal_slice_into(wing_support, 0u, sws.T3, contraction_options);
                fill_T3_values_packed_slice_coherent(wing_support, sj, packed_B, sws.T3, 0u, vws.T3, contraction_options);
                build_T1_support_qlocal_slice_into(wing_support, 0u, sws.T1, contraction_options);
                fill_T1_values_packed_slice_coherent(wing_support, sj, packed_B, sws.T1, 0u, vws.T1, contraction_options);
            } else {
                build_T1_support_qlocal_slice_into(wing_support, 0u, sws.T1, contraction_options);
                fill_T1_values_packed_slice_coherent(wing_support, sj, packed_B, sws.T1, 0u, vws.T1, contraction_options);
                build_T3_support_qlocal_slice_into(wing_support, 0u, sws.T3, contraction_options);
                fill_T3_values_packed_slice_coherent(wing_support, sj, packed_B, sws.T3, 0u, vws.T3, contraction_options);
            }
            result.timings.T1T3 += xqslab_seconds_since(t0);
            result.stats.T1_values_total += sws.T1.packed_values;
            result.stats.T3_values_total += sws.T3.packed_values;

            t0 = clock::now();
            const auto qslabs = choose_q_slabs_for_xslice_half_sjC_coherent(
                xslice_full_support, wing_support, sws.T1, sws.T3, opt, fixed_bytes);
            result.timings.planning += xqslab_seconds_since(t0);

            if (log && opt.print_plan) {
                *log << "[coherent xqslab] q slabs for x=" << ix << ": " << qslabs.size() << "\n";
            }

            for (std::size_t iq = 0; iq < qslabs.size(); ++iq) {
                const PackedAmplitudeQSlab& qslab = qslabs[iq];
                ++x_q_slabs;
                ++result.q_slabs_total;
                result.max_estimated_peak_bytes = std::max(result.max_estimated_peak_bytes, qslab.estimated_peak_bytes);
                if (!qslab.fits_budget) {
                    ++x_unfit;
                    ++result.unfitting_q_slabs;
                    if (opt.require_all_q_slabs_fit) {
                        throw b4utils::ResourceLimitFailure("coherent x+q-slab fallback: one q-slab still exceeds the memory budget");
                    }
                }
                if (log && opt.print_q_timing) {
                    *log << "  coherent q slab " << iq << " q=[" << qslab.q_begin << "," << qslab.q_end << ") peak=";
                    xqslab_print_bytes(*log, qslab.estimated_peak_bytes);
                    *log << (qslab.fits_budget ? "\n" : "  NO\n");
                }

                t0 = clock::now();
                AmplitudeInputSupportPlan_HalfSjC q_support =
                    make_amplitude_xslice_qslab_support_half_sjC(xslice_full_support, qslab.q_begin, qslab.q_end);
                result.timings.qsupport += xqslab_seconds_since(t0);

                t0 = clock::now();
                fill_packed_amplitude_sjC_only_half_sjC(b4spins, q_support, sj);
                result.timings.sjC_fill += xqslab_seconds_since(t0);

                t0 = clock::now();
                const b4utils::Complex64 qamp = contract_qlocal_streamed_slice_allq_half_sjC_coherent(
                    q_support, sj, packed_B, sws.T1, vws.T1, sws.T3, vws.T3, 0u,
                    contraction_options, opt.print_contraction_stats ? &result.stats : nullptr);
                x_amp.re += qamp.re;
                x_amp.im += qamp.im;
                result.timings.qcontract += xqslab_seconds_since(t0);
            }
            release_t0 = clock::now();
        }
        result.timings.release += xqslab_seconds_since(release_t0);
        amplitude.re += x_amp.re;
        amplitude.im += x_amp.im;
        if (log && opt.print_x_timing) {
            *log << "[coherent xqslab] x=" << ix
                 << " q_slabs=" << x_q_slabs
                 << " unfitting=" << x_unfit
                 << " x_amp=(" << std::scientific << std::setprecision(17) << x_amp.re << ", " << x_amp.im << ")"
                 << " cumulative=(" << amplitude.re << ", " << amplitude.im << ")"
                 << std::fixed << std::setprecision(6) << "\n";
        }
    }
    result.amplitude = amplitude;
    result.timings.total = xqslab_seconds_since(total_t0);
    if (log && opt.print_contraction_stats) result.stats.print(*log);
    return result;
}

std::array<int, 5> make_coherent_i1_support_template_boundarys(
    const B4Spins& spins,
    int Di1)
{
    const std::vector<int> valid_i1 =
        get_valid_i(spins.Dj14, spins.Dj15, spins.Dj13, spins.Dj12);
    const std::vector<int> valid_i2 =
        get_valid_i(spins.Dj23, spins.Dj24, spins.Dj25, spins.Dj12);
    const std::vector<int> valid_i3 =
        get_valid_i(spins.Dj34, spins.Dj35, spins.Dj13, spins.Dj23);
    const std::vector<int> valid_i4 =
        get_valid_i(spins.Dj45, spins.Dj14, spins.Dj24, spins.Dj34);
    const std::vector<int> valid_i5 =
        get_valid_i(spins.Dj15, spins.Dj25, spins.Dj35, spins.Dj45);

    if (!b4_contains_int_sorted_local(valid_i1, Di1)) {
        throw b4utils::InternalFailure(
            "coherent C1 contraction: C1 contains an invalid i1 value");
    }

    std::array<int, 5> out{};
    out[0] = Di1;

    // Prefer reflection-friendly placeholders when available.  This preserves
    // the existing q-orbit backend guard for reflection-symmetric data without
    // making the public API ask for meaningless i2..i5 values.
    const int i25 = b4_pick_common_or_first_local(valid_i2, valid_i5, "i2", "i5");
    out[1] = i25;
    out[4] = b4_contains_int_sorted_local(valid_i5, i25)
           ? i25
           : b4_pick_first_valid_or_throw_local(valid_i5, "i5");

    const int i34 = b4_pick_common_or_first_local(valid_i3, valid_i4, "i3", "i4");
    out[2] = i34;
    out[3] = b4_contains_int_sorted_local(valid_i4, i34)
           ? i34
           : b4_pick_first_valid_or_throw_local(valid_i4, "i4");

    return out;
}

PackedCoherentI1BatchResult contract_packed_amplitude_coherent_C1_streamed_xqslab(
    const B4Spins& spins,
    int Dl,
    const std::array<int, 10>& spins_array,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const B4CoherentCoefficientsView& C1,
    const PackedCoherentI1BatchOptions& opt,
    WingExecutionOrder order,
    PackedContractionOptions contraction_options,
    std::ostream* log)
{
    using clock = std::chrono::high_resolution_clock;
    const auto total_t0 = clock::now();
    PackedCoherentI1BatchResult result;

    b4_assert_valid_coherent_coefficients(C1);
    const bool have_progress_plan =
        opt.progress_i1_values != nullptr &&
        opt.progress_units_by_i1 != nullptr &&
        opt.progress_i1_values->size() == opt.progress_units_by_i1->size();
    std::vector<std::uint8_t> progress_i1_reported(
        have_progress_plan ? opt.progress_i1_values->size() : 0u,
        std::uint8_t{0});
    const auto advance_completed_i1 = [&](int Di1, std::size_t fallback_x) {
        if (opt.progress_advance == nullptr) return;

        std::uint64_t completed_units =
            static_cast<std::uint64_t>(fallback_x);
        if (have_progress_plan) {
            const auto position = std::lower_bound(
                opt.progress_i1_values->begin(),
                opt.progress_i1_values->end(),
                Di1);
            if (position != opt.progress_i1_values->end() &&
                *position == Di1) {
                const std::size_t index = static_cast<std::size_t>(
                    position - opt.progress_i1_values->begin());
                if (progress_i1_reported[index] != 0) return;
                progress_i1_reported[index] = 1;
                completed_units = (*opt.progress_units_by_i1)[index];
            }
        }
        if (completed_units != 0) {
            opt.progress_advance(opt.progress_context, completed_units);
        }
    };
    for (std::size_t i = 0; i < C1.size(); ++i) {
        const int Di1 = C1.doubled_iotas[i];
        const b4utils::Complex64 c = C1.coefficients[i];
        if (opt.skip_zero_c1 && c.re == 0.0 && c.im == 0.0) continue;
        ++result.active_i1;

        const std::array<int, 5> fixed_intertwiners_for_support =
            make_coherent_i1_support_template_boundarys(spins, Di1);
        const auto support_t0 = clock::now();
        const AmplitudeChannels AmC = make_AmplitudeChannels(spins, Dl, fixed_intertwiners_for_support);
        const auto support = build_amplitude_static_booster_support_plan_half_sjC(spins, Dl, AmC);
        result.support_seconds += std::chrono::duration<double>(clock::now() - support_t0).count();

        if (AmC.xvals.empty() || AmC.k2.empty() || AmC.k3.empty() || AmC.k4.empty() || AmC.k5.empty()) {
            if (log && opt.print_i1_timing) {
                *log << "\n[coherent C1] i1 " << result.active_i1
                     << "/? Di1=" << Di1
                     << " C1=(" << c.re << ", " << c.im << ")"
                     << " empty support -> skipped\n";
            }
            advance_completed_i1(Di1, 0);
            continue;
        }

        if (opt.require_same_booster_support) {
            const std::uint64_t need = support.B2.packed_values + support.B3.packed_values + support.B4.packed_values + support.B5.packed_values;
            const std::uint64_t have = static_cast<std::uint64_t>(packed_B.B2.size() + packed_B.B3.size() + packed_B.B4.size() + packed_B.B5.size());
            if (have < need) {
                throw b4utils::InternalFailure(
                    "coherent C1 contraction: supplied projected boosters do not fit this i1 support");
            }
        }

        if (log && opt.print_i1_timing) {
            *log << "\n[coherent C1] i1 " << result.active_i1
                 << "/? Di1=" << Di1
                 << " C1=(" << c.re << ", " << c.im << ")\n";
        }

        const auto run_t0 = clock::now();
        PackedAmplitudeCoherentXQResult ares = contract_packed_amplitude_coherent_xqslab_streamed_generated_support(
            support, packed_B, spins_array, opt.xq, order, contraction_options, log);
        result.sj_contract_seconds += std::chrono::duration<double>(clock::now() - run_t0).count();
        result.sj_fill_seconds +=
            ares.timings.wing_fill + ares.timings.sjC_fill;
        result.max_estimated_peak_bytes = std::max(result.max_estimated_peak_bytes, ares.max_estimated_peak_bytes);

        result.amplitude.re += c.re * ares.amplitude.re - c.im * ares.amplitude.im;
        result.amplitude.im += c.re * ares.amplitude.im + c.im * ares.amplitude.re;
        advance_completed_i1(Di1, AmC.xvals.size());
    }
    result.total_seconds = std::chrono::duration<double>(clock::now() - total_t0).count();
    if (log && opt.print_plan) result.print(*log);
    return result;
}

CoherentC1InputSupportPlan_HalfSjC build_coherent_C1_input_support_plan_half_sjC(
    const B4Spins& s,
    int Dl,
    const CoherentAmplitudeChannels& ch)
{
    CoherentC1InputSupportPlan_HalfSjC plan;
    plan.coherent_channels = ch;

    const AmplitudeChannels base = make_base_channels_from_coherent_C1_channels(ch);
    plan.common = build_amplitude_input_support_plan_half_sjC(s, Dl, base);

    const std::size_t X   = ch.xvals.size();
    const std::size_t I1  = ch.i1.size();
    const std::size_t L25 = plan.common.l25.size();
    const std::size_t K2d = ch.k2.size();
    const std::size_t K5d = ch.k5.size();

    plan.sjA_all = make_support_1d(
        "sjA support: block (x,l25,i1) -> k5",
        {X, L25, I1},
        K5d
    );
    plan.sjE_all = make_support_1d(
        "sjE support: block (x,l25,i1) -> k2",
        {X, L25, I1},
        K2d
    );

    const std::vector<int>& xvals = ch.xvals;
    const std::vector<int>& l25   = plan.common.l25;
    const std::vector<int>& k2    = ch.k2;
    const std::vector<int>& k5    = ch.k5;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) if(X * L25 * I1 >= 32768)
#endif
    for (std::ptrdiff_t ix_s = 0; ix_s < static_cast<std::ptrdiff_t>(X); ++ix_s) {
        for (std::ptrdiff_t il25_s = 0; il25_s < static_cast<std::ptrdiff_t>(L25); ++il25_s) {
            for (std::ptrdiff_t ii1_s = 0; ii1_s < static_cast<std::ptrdiff_t>(I1); ++ii1_s) {
                const std::size_t ix   = static_cast<std::size_t>(ix_s);
                const std::size_t il25 = static_cast<std::size_t>(il25_s);
                const std::size_t ii1  = static_cast<std::size_t>(ii1_s);

                const int Dx   = xvals[ix];
                const int Dl25 = l25[il25];
                const int Di1  = ch.i1[ii1];

                const std::size_t block = idx3(ix, il25, ii1, L25, I1);

                if (!triangle_condition(Di1, s.Dj14, s.Dj15) ||
                    !triangle_condition(Di1, Dl25, Dx))
                {
                    plan.sjA_all.blocks[block] = empty_range_1d();
                } else {
                    plan.sjA_all.blocks[block] = axis_range_for_two_triangles(
                        k5,
                        Dl25, s.Dj15,
                        s.Dj14, Dx
                    );
                }

                if (!triangle_condition(Di1, s.Dj13, s.Dj12) ||
                    !triangle_condition(Di1, Dl25, Dx))
                {
                    plan.sjE_all.blocks[block] = empty_range_1d();
                } else {
                    plan.sjE_all.blocks[block] = axis_range_for_two_triangles(
                        k2,
                        Dl25, s.Dj12,
                        s.Dj13, Dx
                    );
                }
            }
        }
    }

    finalize_offsets(plan.sjA_all);
    finalize_offsets(plan.sjE_all);

    plan.i1_x_valid.assign(I1 * X, std::uint8_t{0});
    plan.i1_x_valid_pairs = 0;
    for (std::size_t ii1 = 0; ii1 < I1; ++ii1) {
        for (std::size_t ix = 0; ix < X; ++ix) {
            bool has_A = false;
            bool has_E = false;
            for (std::size_t il25 = 0; il25 < L25; ++il25) {
                const std::size_t block = idx3(ix, il25, ii1, L25, I1);
                has_A = has_A || !plan.sjA_all.blocks[block].empty();
                has_E = has_E || !plan.sjE_all.blocks[block].empty();
                if (has_A && has_E) break;
            }
            if (has_A && has_E) {
                plan.i1_x_valid[ii1 * X + ix] = 1;
                ++plan.i1_x_valid_pairs;
            }
        }
    }

    return plan;
}

void fill_packed_coherent_C1_sjA_sjE_all_half_sjC(
    const B4Spins& s,
    const CoherentC1InputSupportPlan_HalfSjC& support,
    PackedAmplitudeSJTensors_HalfSjC& out)
{
    const CoherentAmplitudeChannels& ch = support.coherent_channels;
    const std::vector<int>& xvals = ch.xvals;
    const std::vector<int>& i1    = ch.i1;
    const std::vector<int>& k2    = ch.k2;
    const std::vector<int>& k5    = ch.k5;
    const std::vector<int>& l25   = support.common.l25;

    const std::size_t X   = xvals.size();
    const std::size_t I1  = i1.size();
    const std::size_t L25 = l25.size();
    const std::size_t K2d = k2.size();
    const std::size_t K5d = k5.size();

    out.sjA.resize(static_cast<std::size_t>(support.sjA_all.packed_values));
    out.sjE.resize(static_cast<std::size_t>(support.sjE_all.packed_values));
    std::fill(out.sjA.data(), out.sjA.data() + out.sjA.size(), 0.0);
    std::fill(out.sjE.data(), out.sjE.data() + out.sjE.size(), 0.0);

    if (X == 0 || I1 == 0 || L25 == 0) return;

    std::vector<double> sqrt_i1(I1);
    std::vector<double> sqrt_k5(K5d);
    for (std::size_t ia = 0; ia < I1; ++ia) sqrt_i1[ia] = std::sqrt(static_cast<double>(i1[ia] + 1));
    for (std::size_t ib = 0; ib < K5d; ++ib) sqrt_k5[ib] = std::sqrt(static_cast<double>(k5[ib] + 1));

    const Wigner15jRealPhaseSplit phase_split =
        wigner15j_real_phase_split(s);
    const int PrefE = phase_split.e_correction * pm1_from_even_Dsum(
        s.Dj12 - s.Dj35 - s.Dj34 - s.Dj24 - phase_split.e_parity);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<double> rec_values;
        rec_values.reserve(std::max<std::size_t>({X, I1, K2d, K5d}) + 8u);

#ifdef _OPENMP
#pragma omp for collapse(3) schedule(dynamic, 16) nowait
#endif
        for (std::size_t ia = 0; ia < I1; ++ia) {
            for (std::size_t il25 = 0; il25 < L25; ++il25) {
                for (std::size_t ix = 0; ix < X; ++ix) {
                    const std::size_t block = idx3(ix, il25, ia, L25, I1);
                    const Range1D& r = support.sjA_all.blocks[block];
                    if (r.empty()) continue;

                    const int Di1  = i1[ia];
                    const int Dx   = xvals[ix];
                    const int Dl25 = l25[il25];

                    const int k5_parity = (s.Dj15 + Dl25) & 1;
                    if (((Dx + s.Dj14) & 1) != k5_parity) continue;

                    const int Dk5_min = std::max(iabs_int(s.Dj15 - Dl25), iabs_int(Dx - s.Dj14));
                    const int Dk5_max = std::min(s.Dj15 + Dl25, Dx + s.Dj14);
                    if (Dk5_min > Dk5_max) continue;

                    const Range1D rk5 = axis_range_from_interval_regular2(k5, Dk5_min, Dk5_max, k5_parity);
                    if (rk5.empty()) continue;

                    const std::size_t ik5_begin = std::max<std::size_t>(rk5.begin, r.begin);
                    const std::size_t ik5_end   = std::min<std::size_t>(rk5.end,   r.end);
                    if (ik5_begin >= ik5_end) continue;

                    const bool ok = sjrec_detail::compute_wigner6j_x_series(
                        Dk5_min,
                        Dk5_max,
                        s.Dj15,
                        Dl25,
                        Di1,
                        Dx,
                        s.Dj14,
                        rec_values);
                    if (!ok) continue;

                    double* const dst = out.sjA.data() + static_cast<std::size_t>(r.offset);
                    const std::size_t local0 = ik5_begin - static_cast<std::size_t>(r.begin);
                    const std::size_t n_store = ik5_end - ik5_begin;
                    const std::size_t rec_off0 = sjrec_detail::x_offset_from_Dx(k5[ik5_begin], Dk5_min);
                    const double xw = static_cast<double>(Dx + 1);

#ifdef _OPENMP
#pragma omp simd
#endif
                    for (std::size_t t = 0; t < n_store; ++t) {
                        const std::size_t ik5 = ik5_begin + t;
                        const int Dk5 = k5[ik5];
                        const double PrefA = static_cast<double>(pm1_from_even_Dsum(
                                                 Di1 + Dk5 + Dl25 + s.Dj15 -
                                                 phase_split.a_parity))
                                           * sqrt_i1[ia]
                                           * sqrt_k5[ik5];
                        dst[local0 + t] = PrefA * xw * rec_values[rec_off0 + t];
                    }
                }
            }
        }

#ifdef _OPENMP
#pragma omp for collapse(3) schedule(dynamic, 16)
#endif
        for (std::size_t ia = 0; ia < I1; ++ia) {
            for (std::size_t il25 = 0; il25 < L25; ++il25) {
                for (std::size_t ix = 0; ix < X; ++ix) {
                    const std::size_t block = idx3(ix, il25, ia, L25, I1);
                    const Range1D& r = support.sjE_all.blocks[block];
                    if (r.empty()) continue;

                    const int Di1  = i1[ia];
                    const int Dx   = xvals[ix];
                    const int Dl25 = l25[il25];

                    const int k2_parity = (s.Dj12 + Dl25) & 1;
                    if (((Dx + s.Dj13) & 1) != k2_parity) continue;

                    const int Dk2_min = std::max(iabs_int(s.Dj12 - Dl25), iabs_int(Dx - s.Dj13));
                    const int Dk2_max = std::min(s.Dj12 + Dl25, Dx + s.Dj13);
                    if (Dk2_min > Dk2_max) continue;

                    const Range1D rk2 = axis_range_from_interval_regular2(k2, Dk2_min, Dk2_max, k2_parity);
                    if (rk2.empty()) continue;

                    const std::size_t ik2_begin = std::max<std::size_t>(rk2.begin, r.begin);
                    const std::size_t ik2_end   = std::min<std::size_t>(rk2.end,   r.end);
                    if (ik2_begin >= ik2_end) continue;

                    const bool ok = sjrec_detail::compute_wigner6j_x_series(
                        Dk2_min,
                        Dk2_max,
                        s.Dj12,
                        Dl25,
                        Di1,
                        Dx,
                        s.Dj13,
                        rec_values);
                    if (!ok) continue;

                    double* const dst = out.sjE.data() + static_cast<std::size_t>(r.offset);
                    const std::size_t local0 = ik2_begin - static_cast<std::size_t>(r.begin);
                    const std::size_t n_store = ik2_end - ik2_begin;
                    const std::size_t rec_off0 = sjrec_detail::x_offset_from_Dx(k2[ik2_begin], Dk2_min);

#ifdef _OPENMP
#pragma omp simd
#endif
                    for (std::size_t t = 0; t < n_store; ++t) {
                        dst[local0 + t] = static_cast<double>(PrefE) * rec_values[rec_off0 + t];
                    }
                }
            }
        }
    }
}

Range1D build_T2_panel_for_q_coherent_skeleton(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const PackedAmplitudeCoherentBoosters& B,
    const CoherentC1QBlockSkeleton& q,
    double* B4PACK_RESTRICT T2_re,
    double* B4PACK_RESTRICT T2_im,
    Range1D* B4PACK_RESTRICT T2_ranges,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t nk4 = static_cast<std::size_t>(q.rC.j_end - q.rC.j_begin);

    for (std::size_t jb = 0; jb < nk4; ++jb) T2_ranges[jb] = empty_range_1d();

    Range1D panel_cover = empty_range_1d();
    for (const CoherentC1BoosterRowDesc& row : q.B4_rows) {
        double* row_re = T2_re + row.local * L25;
        double* row_im = T2_im + row.local * L25;
        qlocal_zero_complex_row(row_re, row_im, L25);
        Range1D row_range = empty_range_1d();

        const double* B4_re = B.B4.re_data() + row.range.offset;
        const double* B4_im = B.B4.im_data() + row.range.offset;
        for (std::size_t il45 = row.range.begin; il45 < row.range.end; ++il45) {
            const std::size_t T1_block = idx3(il45, row.global, q.il35, K4d, L35);
            const Range1D& r_T1 = T1_support.blocks[T1_block];
            if (r_T1.empty()) continue;
            const double ar = B4_re[il45 - row.range.begin];
            const double ai = B4_im[il45 - row.range.begin];
            const double* src_re = T1_values.re_data() + r_T1.offset;
            const double* src_im = T1_values.im_data() + r_T1.offset;
            qlocal_axpy_complex_to_complex(ar, ai, src_re, src_im,
                                           row_re + r_T1.begin,
                                           row_im + r_T1.begin,
                                           r_T1.count());
            qlocal_cover_range_with_range(row_range, r_T1);
        }
        T2_ranges[row.local] = row_range;
        if (!row_range.empty()) {
            qlocal_cover_range_with_range(panel_cover, row_range);
            if (counters) ++counters->T2_rows_built;
        }
    }
    return panel_cover;
}

Range1D build_T4_panel_for_q_coherent_skeleton(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    const PackedAmplitudeCoherentBoosters& B,
    const CoherentC1QBlockSkeleton& q,
    double* B4PACK_RESTRICT T4_re,
    double* B4PACK_RESTRICT T4_im,
    Range1D* B4PACK_RESTRICT T4_ranges,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t nk3 = static_cast<std::size_t>(q.rC.i_end - q.rC.i_begin);

    for (std::size_t ib = 0; ib < nk3; ++ib) T4_ranges[ib] = empty_range_1d();

    Range1D panel_cover = empty_range_1d();
    for (const CoherentC1BoosterRowDesc& row : q.B3_rows) {
        double* row_re = T4_re + row.local * L25;
        double* row_im = T4_im + row.local * L25;
        qlocal_zero_complex_row(row_re, row_im, L25);
        Range1D row_range = empty_range_1d();

        const double* B3_re = B.B3.re_data() + row.range.offset;
        const double* B3_im = B.B3.im_data() + row.range.offset;
        for (std::size_t il23 = row.range.begin; il23 < row.range.end; ++il23) {
            const std::size_t T3_block = idx3(il23, row.global, q.il24, K3d, L24);
            const Range1D& r_T3 = T3_support.blocks[T3_block];
            if (r_T3.empty()) continue;
            const double ar = B3_re[il23 - row.range.begin];
            const double ai = B3_im[il23 - row.range.begin];
            const double* src_re = T3_values.re_data() + r_T3.offset;
            const double* src_im = T3_values.im_data() + r_T3.offset;
            qlocal_axpy_complex_to_complex(ar, ai, src_re, src_im,
                                           row_re + r_T3.begin,
                                           row_im + r_T3.begin,
                                           r_T3.count());
            qlocal_cover_range_with_range(row_range, r_T3);
        }
        T4_ranges[row.local] = row_range;
        if (!row_range.empty()) {
            qlocal_cover_range_with_range(panel_cover, row_range);
            if (counters) ++counters->T4_rows_built;
        }
    }
    return panel_cover;
}

b4utils::Complex64 contract_q_block_C_times_T2_dense_coherent_Cbase_skeleton(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    const CoherentC1QBlockSkeleton& q,
    const double* B4PACK_RESTRICT C_base,
    PackedQLocalComplexThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t L25 = support.l25.size();
    const std::size_t nk3 = static_cast<std::size_t>(q.rC.i_end - q.rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(q.rC.j_end - q.rC.j_begin);

    double* B4PACK_RESTRICT T2_re = ws.T2_re.data();
    double* B4PACK_RESTRICT T2_im = ws.T2_im.data();
    double* B4PACK_RESTRICT T4_re = ws.T4_re.data();
    double* B4PACK_RESTRICT T4_im = ws.T4_im.data();
    double* B4PACK_RESTRICT W_re  = ws.work_re.data();
    double* B4PACK_RESTRICT W_im  = ws.work_im.data();
    Range1D* B4PACK_RESTRICT T2_ranges = ws.T2_ranges.data();
    Range1D* B4PACK_RESTRICT T4_ranges = ws.T4_ranges.data();

    const Range1D T2_cover = build_T2_panel_for_q_coherent_skeleton(
        support, T1_support, T1_values, B, q, T2_re, T2_im, T2_ranges, counters);
    if (T2_cover.empty()) return {0.0, 0.0};

    const Range1D T4_cover = build_T4_panel_for_q_coherent_skeleton(
        support, T3_support, T3_values, B, q, T4_re, T4_im, T4_ranges, counters);
    if (T4_cover.empty()) return {0.0, 0.0};

    const Range1D active_cover = intersect_2_ranges(T2_cover, T4_cover);
    if (active_cover.empty()) return {0.0, 0.0};

    // Dense C*T2 hot path.  C_base is effectively dense in the measured cases,
    // so do not branch on c==0.  Only the l25 interval that can survive the
    // final T4 dot is projected, which avoids the old "work zero elems" tail.
    for (std::size_t ib = 0; ib < nk3; ++ib) {
        qlocal_zero_complex_row(W_re + ib * L25 + active_cover.begin,
                                W_im + ib * L25 + active_cover.begin,
                                active_cover.count());
    }

    for (std::size_t jb = 0; jb < nk4; ++jb) {
        const Range1D r_T2_active = intersect_2_ranges(T2_ranges[jb], active_cover);
        if (r_T2_active.empty()) continue;

        const double* B4PACK_RESTRICT row_re = T2_re + jb * L25;
        const double* B4PACK_RESTRICT row_im = T2_im + jb * L25;
        const double* B4PACK_RESTRICT C_col  = C_base + nk3 * jb;

        for (std::size_t ib = 0; ib < nk3; ++ib) {
            const double c = C_col[ib];
            qlocal_axpy_real_to_complex(c,
                                        row_re + r_T2_active.begin,
                                        row_im + r_T2_active.begin,
                                        W_re + ib * L25 + r_T2_active.begin,
                                        W_im + ib * L25 + r_T2_active.begin,
                                        r_T2_active.count());
        }
    }

    b4utils::Complex64 local{0.0, 0.0};
    for (std::size_t ib = 0; ib < nk3; ++ib) {
        const Range1D r = intersect_2_ranges(active_cover, T4_ranges[ib]);
        if (r.empty()) continue;
        const b4utils::Complex64 z = qlocal_dot_complex_no_conj(
            W_re  + ib * L25 + r.begin, W_im  + ib * L25 + r.begin,
            T4_re + ib * L25 + r.begin, T4_im + ib * L25 + r.begin,
            r.count());
        local.re += z.re;
        local.im += z.im;
    }
    return local;
}

b4utils::Complex64 contract_q_block_Ct_times_T4_generic_coherent_Cbase_skeleton(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    const CoherentC1QBlockSkeleton& q,
    const double* B4PACK_RESTRICT C_base,
    PackedQLocalComplexThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t L25 = support.l25.size();
    const std::size_t nk3 = static_cast<std::size_t>(q.rC.i_end - q.rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(q.rC.j_end - q.rC.j_begin);

    double* B4PACK_RESTRICT T2_re = ws.T2_re.data();
    double* B4PACK_RESTRICT T2_im = ws.T2_im.data();
    double* B4PACK_RESTRICT T4_re = ws.T4_re.data();
    double* B4PACK_RESTRICT T4_im = ws.T4_im.data();
    double* B4PACK_RESTRICT W_re  = ws.work_re.data();
    double* B4PACK_RESTRICT W_im  = ws.work_im.data();
    Range1D* B4PACK_RESTRICT T2_ranges = ws.T2_ranges.data();
    Range1D* B4PACK_RESTRICT T4_ranges = ws.T4_ranges.data();

    const Range1D T4_cover = build_T4_panel_for_q_coherent_skeleton(
        support, T3_support, T3_values, B, q, T4_re, T4_im, T4_ranges, counters);
    if (T4_cover.empty()) return {0.0, 0.0};

    const Range1D T2_cover = build_T2_panel_for_q_coherent_skeleton(
        support, T1_support, T1_values, B, q, T2_re, T2_im, T2_ranges, counters);
    if (T2_cover.empty()) return {0.0, 0.0};

    const Range1D active_cover = intersect_2_ranges(T4_cover, T2_cover);
    if (active_cover.empty()) return {0.0, 0.0};

    // Rare fallback.  Keep the generic Ct*T4 layout, but still do not branch on
    // c==0.  Also restrict projection to the l25 interval that can survive.
    for (std::size_t jb = 0; jb < nk4; ++jb) {
        qlocal_zero_complex_row(W_re + jb * L25 + active_cover.begin,
                                W_im + jb * L25 + active_cover.begin,
                                active_cover.count());
    }

    for (std::size_t ib = 0; ib < nk3; ++ib) {
        const Range1D r_T4_active = intersect_2_ranges(T4_ranges[ib], active_cover);
        if (r_T4_active.empty()) continue;

        const double* B4PACK_RESTRICT row_re = T4_re + ib * L25;
        const double* B4PACK_RESTRICT row_im = T4_im + ib * L25;

        for (std::size_t jb = 0; jb < nk4; ++jb) {
            const double c = C_base[ib + nk3 * jb];
            qlocal_axpy_real_to_complex(c,
                                        row_re + r_T4_active.begin,
                                        row_im + r_T4_active.begin,
                                        W_re + jb * L25 + r_T4_active.begin,
                                        W_im + jb * L25 + r_T4_active.begin,
                                        r_T4_active.count());
        }
    }

    b4utils::Complex64 local{0.0, 0.0};
    for (std::size_t jb = 0; jb < nk4; ++jb) {
        const Range1D r = intersect_2_ranges(active_cover, T2_ranges[jb]);
        if (r.empty()) continue;
        const b4utils::Complex64 z = qlocal_dot_complex_no_conj(
            T2_re + jb * L25 + r.begin, T2_im + jb * L25 + r.begin,
            W_re  + jb * L25 + r.begin, W_im  + jb * L25 + r.begin,
            r.count());
        local.re += z.re;
        local.im += z.im;
    }
    return local;
}

void fill_T1_values_packed_slice_coherent_i1tile(
    const AmplitudeInputSupportPlan& common,
    std::vector<CoherentC1TileLaneState>& lanes,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    std::size_t ix,
    PackedContractionOptions opt)
{
    const std::size_t K4d = common.channels.k4.size();
    const std::size_t K5d = common.channels.k5.size();
    const std::size_t L25 = common.l25.size();
    const std::size_t L35 = common.l35.size();
    const std::size_t L45 = common.l45.size();
    assert(ix < common.channels.xvals.size());

    std::size_t active = 0;
    for (CoherentC1TileLaneState& lane : lanes) {
        if (!lane.active_for_x) continue;
        lane.vws.T1.resize_discard(std::max<std::size_t>(1, static_cast<std::size_t>(lane.sws.T1.packed_values)));
        ++active;
    }
    if (active == 0) return;

    const std::size_t tile = lanes.size();
    const std::size_t p5_size = K5d * L25;
    const std::size_t parallel_work = L45 * L35 * K4d * K5d * L25 * active;

    auto process_l35_l45 = [&](std::size_t il35, std::size_t il45,
                               std::vector<double>& P5_re,
                               std::vector<double>& P5_im) {
        std::fill(P5_re.begin(), P5_re.end(), 0.0);
        std::fill(P5_im.begin(), P5_im.end(), 0.0);

        for (std::size_t il25 = 0; il25 < L25; ++il25) {
            const std::size_t B5_block  = idx3(il25, il35, il45, L35, L45);
            const Range1D& r_B5 = common.B5.blocks[B5_block];
            if (r_B5.empty()) continue;
            const double* B4PACK_RESTRICT B5_re = B.B5.re_data() + r_B5.offset;
            const double* B4PACK_RESTRICT B5_im = B.B5.im_data() + r_B5.offset;

            const std::size_t sjA_block = idx2(ix, il25, L25);
            for (std::size_t l = 0; l < tile; ++l) {
                CoherentC1TileLaneState& lane = lanes[l];
                if (!lane.active_for_x) continue;
                const Range1D& r_sjA = lane.sjA.blocks[sjA_block];
                const Range1D r = intersect_2_ranges(r_sjA, r_B5);
                if (r.empty()) continue;
                const double* B4PACK_RESTRICT sjA_base = sj.sjA.data() + r_sjA.offset;
                double* B4PACK_RESTRICT p5r = P5_re.data() + l * p5_size;
                double* B4PACK_RESTRICT p5i = P5_im.data() + l * p5_size;
                for (std::size_t ik5 = r.begin; ik5 < r.end; ++ik5) {
                    const double s = sjA_base[ik5 - r_sjA.begin];
                    const std::size_t p = ik5 * L25 + il25;
                    p5r[p] = s * B5_re[ik5 - r_B5.begin];
                    p5i[p] = s * B5_im[ik5 - r_B5.begin];
                }
            }
        }

        for (std::size_t ik4 = 0; ik4 < K4d; ++ik4) {
            const std::size_t sjB_block = idx4(ix, ik4, il35, il45, K4d, L35, L45);
            const Range1D& r_sjB = common.sjB.blocks[sjB_block];
            const double* B4PACK_RESTRICT sjB_base = r_sjB.empty() ? nullptr : (sj.sjB.data() + r_sjB.offset);

            for (std::size_t l = 0; l < tile; ++l) {
                CoherentC1TileLaneState& lane = lanes[l];
                if (!lane.active_for_x) continue;
                const std::size_t T1_block = idx3(il45, ik4, il35, K4d, L35);
                const Range1D& r_out = lane.sws.T1.blocks[T1_block];
                if (r_out.empty()) continue;

                double* B4PACK_RESTRICT out_re = lane.vws.T1.re_data() + r_out.offset;
                double* B4PACK_RESTRICT out_im = lane.vws.T1.im_data() + r_out.offset;
                const std::size_t n_out = r_out.count();
                qlocal_zero_complex_row(out_re, out_im, n_out);
                if (r_sjB.empty()) continue;

                const double* B4PACK_RESTRICT p5r = P5_re.data() + l * p5_size;
                const double* B4PACK_RESTRICT p5i = P5_im.data() + l * p5_size;
                for (std::size_t ik5 = r_sjB.begin; ik5 < r_sjB.end; ++ik5) {
                    const double coeff = sjB_base[ik5 - r_sjB.begin];
                    qlocal_axpy_real_to_complex(coeff,
                                                p5r + ik5 * L25 + r_out.begin,
                                                p5i + ik5 * L25 + r_out.begin,
                                                out_re, out_im, n_out);
                }
            }
        }
    };

#ifdef _OPENMP
#pragma omp parallel if(opt.use_openmp && parallel_work >= opt.parallel_threshold)
    {
        std::vector<double> P5_re(tile * p5_size), P5_im(tile * p5_size);
#pragma omp for collapse(2) schedule(dynamic, 1)
        for (std::ptrdiff_t il45_s = 0; il45_s < static_cast<std::ptrdiff_t>(L45); ++il45_s) {
            for (std::size_t il35 = 0; il35 < L35; ++il35) {
                process_l35_l45(il35, static_cast<std::size_t>(il45_s), P5_re, P5_im);
            }
        }
    }
#else
    std::vector<double> P5_re(tile * p5_size), P5_im(tile * p5_size);
    for (std::size_t il45 = 0; il45 < L45; ++il45) {
        for (std::size_t il35 = 0; il35 < L35; ++il35) process_l35_l45(il35, il45, P5_re, P5_im);
    }
#endif
}

void fill_T3_values_packed_slice_coherent_i1tile(
    const AmplitudeInputSupportPlan& common,
    std::vector<CoherentC1TileLaneState>& lanes,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    std::size_t ix,
    PackedContractionOptions opt)
{
    const std::size_t K2d = common.channels.k2.size();
    const std::size_t K3d = common.channels.k3.size();
    const std::size_t L23 = common.l23.size();
    const std::size_t L24 = common.l24.size();
    const std::size_t L25 = common.l25.size();
    assert(ix < common.channels.xvals.size());

    std::size_t active = 0;
    for (CoherentC1TileLaneState& lane : lanes) {
        if (!lane.active_for_x) continue;
        lane.vws.T3.resize_discard(std::max<std::size_t>(1, static_cast<std::size_t>(lane.sws.T3.packed_values)));
        ++active;
    }
    if (active == 0) return;

    const std::size_t tile = lanes.size();
    const std::size_t p2_size = K2d * L25;
    const std::size_t parallel_work = L23 * L24 * K3d * K2d * L25 * active;

    auto process_l23_l24 = [&](std::size_t il23, std::size_t il24,
                               std::vector<double>& P2_re,
                               std::vector<double>& P2_im) {
        std::fill(P2_re.begin(), P2_re.end(), 0.0);
        std::fill(P2_im.begin(), P2_im.end(), 0.0);

        for (std::size_t il25 = 0; il25 < L25; ++il25) {
            const std::size_t B2_block  = idx3(il23, il24, il25, L24, L25);
            const Range1D& r_B2 = common.B2.blocks[B2_block];
            if (r_B2.empty()) continue;
            const double* B4PACK_RESTRICT B2_re = B.B2.re_data() + r_B2.offset;
            const double* B4PACK_RESTRICT B2_im = B.B2.im_data() + r_B2.offset;

            const std::size_t sjE_block = idx2(ix, il25, L25);
            for (std::size_t l = 0; l < tile; ++l) {
                CoherentC1TileLaneState& lane = lanes[l];
                if (!lane.active_for_x) continue;
                const Range1D& r_sjE = lane.sjE.blocks[sjE_block];
                const Range1D r = intersect_2_ranges(r_sjE, r_B2);
                if (r.empty()) continue;
                const double* B4PACK_RESTRICT sjE_base = sj.sjE.data() + r_sjE.offset;
                double* B4PACK_RESTRICT p2r = P2_re.data() + l * p2_size;
                double* B4PACK_RESTRICT p2i = P2_im.data() + l * p2_size;
                for (std::size_t ik2 = r.begin; ik2 < r.end; ++ik2) {
                    const double s = sjE_base[ik2 - r_sjE.begin];
                    const std::size_t p = ik2 * L25 + il25;
                    p2r[p] = s * B2_re[ik2 - r_B2.begin];
                    p2i[p] = s * B2_im[ik2 - r_B2.begin];
                }
            }
        }

        for (std::size_t ik3 = 0; ik3 < K3d; ++ik3) {
            const std::size_t sjD_block = idx4(ix, ik3, il23, il24, K3d, L23, L24);
            const Range1D& r_sjD = common.sjD.blocks[sjD_block];
            const double* B4PACK_RESTRICT sjD_base = r_sjD.empty() ? nullptr : (sj.sjD.data() + r_sjD.offset);

            for (std::size_t l = 0; l < tile; ++l) {
                CoherentC1TileLaneState& lane = lanes[l];
                if (!lane.active_for_x) continue;
                const std::size_t T3_block = idx3(il23, ik3, il24, K3d, L24);
                const Range1D& r_out = lane.sws.T3.blocks[T3_block];
                if (r_out.empty()) continue;

                double* B4PACK_RESTRICT out_re = lane.vws.T3.re_data() + r_out.offset;
                double* B4PACK_RESTRICT out_im = lane.vws.T3.im_data() + r_out.offset;
                const std::size_t n_out = r_out.count();
                qlocal_zero_complex_row(out_re, out_im, n_out);
                if (r_sjD.empty()) continue;

                const double* B4PACK_RESTRICT p2r = P2_re.data() + l * p2_size;
                const double* B4PACK_RESTRICT p2i = P2_im.data() + l * p2_size;
                for (std::size_t ik2 = r_sjD.begin; ik2 < r_sjD.end; ++ik2) {
                    const double coeff = sjD_base[ik2 - r_sjD.begin];
                    qlocal_axpy_real_to_complex(coeff,
                                                p2r + ik2 * L25 + r_out.begin,
                                                p2i + ik2 * L25 + r_out.begin,
                                                out_re, out_im, n_out);
                }
            }
        }
    };

#ifdef _OPENMP
#pragma omp parallel if(opt.use_openmp && parallel_work >= opt.parallel_threshold)
    {
        std::vector<double> P2_re(tile * p2_size), P2_im(tile * p2_size);
#pragma omp for collapse(2) schedule(dynamic, 1)
        for (std::ptrdiff_t il23_s = 0; il23_s < static_cast<std::ptrdiff_t>(L23); ++il23_s) {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                process_l23_l24(static_cast<std::size_t>(il23_s), il24, P2_re, P2_im);
            }
        }
    }
#else
    std::vector<double> P2_re(tile * p2_size), P2_im(tile * p2_size);
    for (std::size_t il23 = 0; il23 < L23; ++il23) {
        for (std::size_t il24 = 0; il24 < L24; ++il24) process_l23_l24(il23, il24, P2_re, P2_im);
    }
#endif
}

std::size_t coherent_C1_tile_build_T_values_for_x(
    const CoherentC1InputSupportPlan_HalfSjC& support,
    std::vector<CoherentC1TileLaneState>& lanes,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    std::size_t ix,
    WingExecutionOrder order,
    PackedContractionOptions opt)
{
    std::size_t active = 0;
    for (CoherentC1TileLaneState& lane : lanes) {
        lane.active_for_x = coherent_C1_i1_x_valid(support, lane.ia, ix);
        if (lane.active_for_x) {
            if (lane.sws.T1.blocks.empty() || lane.sws.T3.blocks.empty()) {
                initialize_qlocal_support_workspace_from_common(support.common, lane.sws);
            }
            ++active;
        }
    }
    if (active == 0) return 0;

    auto build_supports = [&]() {
        for (CoherentC1TileLaneState& lane : lanes) {
            if (!lane.active_for_x) continue;
            if (order == WingExecutionOrder::LeftThenRight) {
                build_T3_support_qlocal_slice_coherent_lane_into(support.common, lane.sjE, ix, lane.sws.T3, opt);
                build_T1_support_qlocal_slice_coherent_lane_into(support.common, lane.sjA, ix, lane.sws.T1, opt);
            } else {
                build_T1_support_qlocal_slice_coherent_lane_into(support.common, lane.sjA, ix, lane.sws.T1, opt);
                build_T3_support_qlocal_slice_coherent_lane_into(support.common, lane.sjE, ix, lane.sws.T3, opt);
            }
        }
    };

    build_supports();

    if (active > 1) {
        if (order == WingExecutionOrder::LeftThenRight) {
            fill_T3_values_packed_slice_coherent_i1tile(support.common, lanes, sj, B, ix, opt);
            fill_T1_values_packed_slice_coherent_i1tile(support.common, lanes, sj, B, ix, opt);
        } else {
            fill_T1_values_packed_slice_coherent_i1tile(support.common, lanes, sj, B, ix, opt);
            fill_T3_values_packed_slice_coherent_i1tile(support.common, lanes, sj, B, ix, opt);
        }
        return active;
    }

    for (CoherentC1TileLaneState& lane : lanes) {
        if (!lane.active_for_x) continue;
        if (order == WingExecutionOrder::LeftThenRight) {
            fill_T3_values_packed_slice_coherent_lane(support.common, lane.sjE, sj, B, lane.sws.T3, ix, lane.vws.T3, opt);
            fill_T1_values_packed_slice_coherent_lane(support.common, lane.sjA, sj, B, lane.sws.T1, ix, lane.vws.T1, opt);
        } else {
            fill_T1_values_packed_slice_coherent_lane(support.common, lane.sjA, sj, B, lane.sws.T1, ix, lane.vws.T1, opt);
            fill_T3_values_packed_slice_coherent_lane(support.common, lane.sjE, sj, B, lane.sws.T3, ix, lane.vws.T3, opt);
        }
    }
    return active;
}

std::uint64_t coherent_C1_resident_lane_workspace_estimate_bytes(
    const CoherentC1InputSupportPlan_HalfSjC& support,
    std::size_t active_i1)
{
    // Shallow resident lanes no longer deep-copy support.common.  Each lane owns
    // only the two fixed-i1 wing block views sjA/sjE plus T1/T3 support metadata
    // and value workspaces.  The large common supports stay shared in
    // support.common.  This keeps the resident-fast path viable for large cases.
    const std::uint64_t T_support_metadata = coherent_C1_per_lane_T_support_metadata_bytes(support);

    // The resident path retains split-complex T1/T3 values for every i1 lane.
    // They do not exist yet when this preflight runs, so use the exact dense
    // upper bound implied by the block grids and the l25 output extent.
    const auto& ch = support.common.channels;
    const std::uint64_t L23 = static_cast<std::uint64_t>(support.common.l23.size());
    const std::uint64_t L24 = static_cast<std::uint64_t>(support.common.l24.size());
    const std::uint64_t L25 = static_cast<std::uint64_t>(support.common.l25.size());
    const std::uint64_t L35 = static_cast<std::uint64_t>(support.common.l35.size());
    const std::uint64_t L45 = static_cast<std::uint64_t>(support.common.l45.size());
    const std::uint64_t K3d = static_cast<std::uint64_t>(ch.k3.size());
    const std::uint64_t K4d = static_cast<std::uint64_t>(ch.k4.size());
    const std::uint64_t max_T1_values = xqslab_sat_mul_u64(
        xqslab_sat_mul_u64(xqslab_sat_mul_u64(L45, K4d), L35), L25);
    const std::uint64_t max_T3_values = xqslab_sat_mul_u64(
        xqslab_sat_mul_u64(xqslab_sat_mul_u64(L23, K3d), L24), L25);
    const std::uint64_t complex_T_values = xqslab_sat_mul_u64(
        xqslab_sat_add_u64(max_T1_values, max_T3_values),
        2ull * static_cast<std::uint64_t>(sizeof(double)));

    const std::uint64_t X = static_cast<std::uint64_t>(support.coherent_channels.xvals.size());
    const std::uint64_t wing_blocks = xqslab_sat_mul_u64(X, L25);
    const std::uint64_t wing_metadata = xqslab_sat_mul_u64(
        2ull,
        xqslab_sat_mul_u64(wing_blocks, static_cast<std::uint64_t>(sizeof(Range1D))));

    const std::uint64_t per_lane = xqslab_sat_add_u64(
        xqslab_sat_add_u64(T_support_metadata, wing_metadata),
        complex_T_values);
    const std::uint64_t metadata = xqslab_sat_mul_u64(per_lane, static_cast<std::uint64_t>(active_i1));
    const std::uint64_t lane_objects = xqslab_sat_mul_u64(static_cast<std::uint64_t>(active_i1),
                                                         static_cast<std::uint64_t>(sizeof(CoherentC1TileLaneState)));
    return xqslab_sat_add_u64(metadata, lane_objects);
}

void coherent_C1_build_allq_skeletons_for_x(
    const AmplitudeInputSupportPlan_HalfSjC& common,
    std::size_t ix,
    std::vector<CoherentC1QBlockSkeleton>& qs)
{
    // Spin reflection permits sharing real 6j storage, but arbitrary C2..C5
    // projections do not equate the complex reflected summands. Retain each
    // logical q block even when it reads a reflected physical panel.
    qs.clear();
    const std::size_t L24 = common.l24.size();
    const std::size_t L34 = common.l34.size();
    const std::size_t L35 = common.l35.size();
    const std::size_t L25 = common.l25.size();
    assert(ix < common.channels.xvals.size());
    qs.reserve(L24 * L34 * L35);

    for (std::size_t il24 = 0; il24 < L24; ++il24) {
        for (std::size_t il34 = 0; il34 < L34; ++il34) {
            for (std::size_t il35 = 0; il35 < L35; ++il35) {
                const std::size_t C_block = idx4(ix, il24, il34, il35, L24, L34, L35);
                const AmplitudeSjCHalfBlockMap& map = common.sjC_map[C_block];
                if (map.empty()) continue;
                const Range2D& logical = map.logical;
                if (logical.empty()) continue;

                CoherentC1QBlockSkeleton q;
                q.ix = ix;
                q.il24 = il24;
                q.il34 = il34;
                q.il35 = il35;
                q.C_block = C_block;
                q.rC = logical;
                const std::size_t nk3 = static_cast<std::size_t>(q.rC.i_end - q.rC.i_begin);
                const std::size_t nk4 = static_cast<std::size_t>(q.rC.j_end - q.rC.j_begin);
                q.strategy = choose_qlocal_strategy(nk3, nk4, L25);

                const std::size_t storage = static_cast<std::size_t>(map.storage_block);
                q.direct_C = false;
                if (!map.transpose_from_storage && storage < common.sjC.blocks.size()) {
                    const Range2D& sr = common.sjC.blocks[storage];
                    if (!sr.empty() && sr.i_begin == logical.i_begin && sr.i_end == logical.i_end &&
                        sr.j_begin == logical.j_begin && sr.j_end == logical.j_end) {
                        q.direct_C = true;
                        q.C_offset = static_cast<std::size_t>(sr.offset);
                    }
                }
                coherent_C1_populate_qblock_booster_rows(common, q);
                if (q.B3_rows.empty() || q.B4_rows.empty()) continue;
                qs.emplace_back(std::move(q));
            }
        }
    }
}

void coherent_C1_contract_qlocal_skeletons_tile(
    const AmplitudeInputSupportPlan_HalfSjC& common,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    std::vector<CoherentC1TileLaneState>& lanes,
    const std::vector<CoherentC1QBlockSkeleton>& qs,
    PackedContractionOptions opt,
    std::uint64_t* q_block_calls,
    std::uint64_t* q_lane_calls)
{
    if (lanes.empty() || qs.empty()) return;
    const std::size_t tile = lanes.size();
    std::size_t active_tile_lanes = 0;
    for (const auto& lane : lanes) active_tile_lanes += lane.active_for_x ? 1u : 0u;
    if (active_tile_lanes == 0) return;

    const std::size_t K3d = common.channels.k3.size();
    const std::size_t K4d = common.channels.k4.size();
    const std::size_t L25 = common.l25.size();
    std::vector<double> total_re(tile, 0.0), total_im(tile, 0.0);
    std::mutex accumulation_mutex;

#ifdef _OPENMP
#pragma omp parallel if(opt.use_openmp && qs.size() * K3d * K4d * tile >= opt.parallel_threshold)
#endif
    {
        PackedQLocalComplexThreadWorkspace ws_scalar;
        ws_scalar.ensure(K3d, K4d, L25);
        std::vector<double> local_re(tile, 0.0), local_im(tile, 0.0);
        std::vector<double> Ctmp;
        std::uint64_t local_q_blocks = 0;
        std::uint64_t local_q_lanes = 0;

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (std::ptrdiff_t iq_s = 0; iq_s < static_cast<std::ptrdiff_t>(qs.size()); ++iq_s) {
            const CoherentC1QBlockSkeleton& q = qs[static_cast<std::size_t>(iq_s)];
            const double* Cbase = nullptr;
            if (q.direct_C) {
                Cbase = sj.sjC.data() + q.C_offset;
            } else {
                materialize_logical_sjC_panel_half(common, sj, q.C_block, q.rC, Ctmp);
                Cbase = Ctmp.data();
            }
            ++local_q_blocks;
            local_q_lanes += active_tile_lanes;

            for (std::size_t l = 0; l < tile; ++l) {
                if (!lanes[l].active_for_x) continue;
                const b4utils::Complex64 z = contract_q_block_streamed_adaptive_coherent_Cbase_skeleton(
                    common, sj, B,
                    lanes[l].sws.T1, lanes[l].vws.T1,
                    lanes[l].sws.T3, lanes[l].vws.T3,
                    q, Cbase, ws_scalar, nullptr);
                local_re[l] += z.re;
                local_im[l] += z.im;
            }
        }

        {
            const std::lock_guard lock(accumulation_mutex);
            for (std::size_t l = 0; l < tile; ++l) {
                total_re[l] += local_re[l];
                total_im[l] += local_im[l];
            }
            if (q_block_calls) *q_block_calls += local_q_blocks;
            if (q_lane_calls) *q_lane_calls += local_q_lanes;
        }
    }

    for (std::size_t l = 0; l < tile; ++l) {
        lanes[l].amplitude.re += total_re[l];
        lanes[l].amplitude.im += total_im[l];
    }
}

std::uint64_t coherent_C1_estimate_i1_tile_per_thread_scratch_bytes(
    const CoherentC1InputSupportPlan_HalfSjC& support,
    std::size_t tile) noexcept
{
    const std::size_t K2d = support.common.channels.k2.size();
    const std::size_t K3d = support.common.channels.k3.size();
    const std::size_t K4d = support.common.channels.k4.size();
    const std::size_t K5d = support.common.channels.k5.size();
    const std::size_t L25 = support.common.l25.size();

    const std::size_t q_rows = std::max<std::size_t>(1, std::max(K3d, K4d));
    const std::uint64_t q_elems = static_cast<std::uint64_t>(q_rows) *
                                  static_cast<std::uint64_t>(std::max<std::size_t>(1, L25));

    // Even though the stable scalar q path reuses one actual q workspace, the
    // *useful cache footprint* of a larger tile still scales with tile because
    // its T1/T3 lane state, ranges, lane loop metadata, and builder scratch all
    // compete with the same C/T2/T4 panels.  The tile=31 benchmark proved that
    // malloc-size is the wrong model.  Model residency, not mere allocation.
    const std::uint64_t q_lane_bytes =
        6ull * q_elems * static_cast<std::uint64_t>(sizeof(double)) +
        2ull * static_cast<std::uint64_t>(q_rows) * static_cast<std::uint64_t>(sizeof(Range1D));
    const std::uint64_t q_residency_bytes =
        xqslab_sat_mul_u64(static_cast<std::uint64_t>(tile), q_lane_bytes);

    const std::size_t p_rows = std::max<std::size_t>(1, std::max(K2d, K5d));
    const std::uint64_t p_elems = static_cast<std::uint64_t>(p_rows) *
                                  static_cast<std::uint64_t>(std::max<std::size_t>(1, L25));
    const std::uint64_t T_builder_bytes =
        2ull * static_cast<std::uint64_t>(tile) * p_elems * static_cast<std::uint64_t>(sizeof(double));

    const std::uint64_t base = std::max(q_residency_bytes, T_builder_bytes);
    return xqslab_sat_add_u64(base, 16ull * 1024ull);
}

std::size_t coherent_C1_select_i1_tile_size(
    const CoherentC1InputSupportPlan_HalfSjC& support,
    const B4CoherentCoefficientsView& C1,
    const PackedCoherentC1SharedOptions& opt,
    PackedContractionOptions contraction_options)
{
    const std::size_t I1 = support.coherent_channels.i1.size();
    if (I1 == 0) return 1;

    if (!opt.auto_i1_tile_size && opt.i1_tile_size != 0) {
        return std::max<std::size_t>(1, std::min<std::size_t>(I1, opt.i1_tile_size));
    }

    const std::size_t candidates = coherent_C1_count_candidate_i1_lanes(support, C1, opt.skip_zero_c1);
    std::size_t max_tile = std::max<std::size_t>(1, std::min<std::size_t>(I1, candidates == 0 ? I1 : candidates));
    if (opt.i1_tile_hard_max != 0) {
        max_tile = std::max<std::size_t>(1, std::min<std::size_t>(max_tile, opt.i1_tile_hard_max));
    }

    const std::uint64_t total_budget = opt.i1_tile_scratch_budget_bytes == 0
        ? (64ull * 1024ull * 1024ull)
        : opt.i1_tile_scratch_budget_bytes;
    const std::uint64_t cache_budget = opt.i1_tile_cache_budget_bytes == 0
        ? coherent_C1_auto_i1_tile_cache_budget_bytes()
        : opt.i1_tile_cache_budget_bytes;
    const std::size_t threads = coherent_C1_effective_threads_for_scratch(contraction_options);
    std::size_t best = 1;
    for (std::size_t t = 1; t <= max_tile; ) {
        const std::uint64_t per_thread = coherent_C1_estimate_i1_tile_per_thread_scratch_bytes(
            support, t);
        const std::uint64_t total = xqslab_sat_mul_u64(per_thread, static_cast<std::uint64_t>(threads));
        if (per_thread <= cache_budget && total <= total_budget) best = t;
        if (t > max_tile / 2) break;
        t *= 2;
    }
    return std::max<std::size_t>(1, best);
}

b4utils::Complex64 coherent_C1_contract_qlocal_skeletons_flat_scalar_weighted(
    const AmplitudeInputSupportPlan_HalfSjC& common,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    std::vector<std::vector<CoherentC1TileLaneState>>& tile_groups,
    const std::vector<CoherentC1QBlockSkeleton>& qs,
    PackedContractionOptions opt,
    std::size_t requested_chunk,
    std::uint64_t* q_block_calls,
    std::uint64_t* q_lane_calls)
{
    if (tile_groups.empty() || qs.empty()) return {0.0, 0.0};

    std::vector<CoherentC1TileLaneState*> active_lanes;
    active_lanes.reserve(64);
    for (auto& lanes : tile_groups) {
        for (CoherentC1TileLaneState& lane : lanes) {
            if (lane.active_for_x) active_lanes.push_back(&lane);
        }
    }
    if (active_lanes.empty()) return {0.0, 0.0};

    const std::size_t K3d = common.channels.k3.size();
    const std::size_t K4d = common.channels.k4.size();
    const std::size_t L25 = common.l25.size();
    const std::uint64_t ntasks_u64 = xqslab_sat_mul_u64(
        static_cast<std::uint64_t>(active_lanes.size()),
        static_cast<std::uint64_t>(qs.size()));
    const std::size_t ntasks = static_cast<std::size_t>(std::min<std::uint64_t>(
        ntasks_u64, static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())));
    if (ntasks == 0) return {0.0, 0.0};

    const std::size_t chunk = requested_chunk == 0
        ? (ntasks >= 4096 ? 16u : 4u)
        : std::max<std::size_t>(1, requested_chunk);

    double total_re = 0.0;
    double total_im = 0.0;
    std::mutex accumulation_mutex;

#ifdef _OPENMP
#pragma omp parallel if(opt.use_openmp && ntasks * K3d * K4d >= opt.parallel_threshold)
#endif
    {
        PackedQLocalComplexThreadWorkspace ws;
        ws.ensure(K3d, K4d, L25);
        std::vector<double> Ctmp;
        double local_re_sum = 0.0;
        double local_im_sum = 0.0;
        std::uint64_t local_q_blocks = 0;
        std::uint64_t local_q_lanes = 0;

#ifdef _OPENMP
#pragma omp for schedule(dynamic, chunk)
#endif
        for (std::ptrdiff_t it_s = 0; it_s < static_cast<std::ptrdiff_t>(ntasks); ++it_s) {
            const std::size_t it = static_cast<std::size_t>(it_s);
            const std::size_t iq = it % qs.size();
            const std::size_t ilane = it / qs.size();
            CoherentC1TileLaneState& lane = *active_lanes[ilane];
            const CoherentC1QBlockSkeleton& q = qs[iq];

            const double* Cbase = nullptr;
            if (q.direct_C) {
                Cbase = sj.sjC.data() + q.C_offset;
            } else {
                materialize_logical_sjC_panel_half(common, sj, q.C_block, q.rC, Ctmp);
                Cbase = Ctmp.data();
            }

            const b4utils::Complex64 z = contract_q_block_streamed_adaptive_coherent_Cbase_skeleton(
                common, sj, B,
                lane.sws.T1, lane.vws.T1,
                lane.sws.T3, lane.vws.T3,
                q, Cbase, ws, nullptr);

            const b4utils::Complex64 c = lane.coeff;
            const double ar = z.re;
            const double ai = z.im;
            local_re_sum += c.re * ar - c.im * ai;
            local_im_sum += c.re * ai + c.im * ar;
            ++local_q_blocks;
            ++local_q_lanes;
        }

        {
            const std::lock_guard lock(accumulation_mutex);
            total_re += local_re_sum;
            total_im += local_im_sum;
            if (q_block_calls) *q_block_calls += local_q_blocks;
            if (q_lane_calls) *q_lane_calls += local_q_lanes;
        }
    }

    return {total_re, total_im};
}

b4utils::Complex64 coherent_C1_contract_qlocal_skeletons_tile_weighted(
    const AmplitudeInputSupportPlan_HalfSjC& common,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    std::vector<CoherentC1TileLaneState>& lanes,
    const std::vector<CoherentC1QBlockSkeleton>& qs,
    PackedContractionOptions opt,
    std::uint64_t* q_block_calls,
    std::uint64_t* q_lane_calls)
{
    if (lanes.empty() || qs.empty()) return {0.0, 0.0};
    const std::size_t tile = lanes.size();
    std::size_t active_tile_lanes = 0;
    for (const auto& lane : lanes) active_tile_lanes += lane.active_for_x ? 1u : 0u;
    if (active_tile_lanes == 0) return {0.0, 0.0};

    const std::size_t K3d = common.channels.k3.size();
    const std::size_t K4d = common.channels.k4.size();
    const std::size_t L25 = common.l25.size();
    double total_re = 0.0;
    double total_im = 0.0;
    std::mutex accumulation_mutex;

#ifdef _OPENMP
#pragma omp parallel if(opt.use_openmp && qs.size() * K3d * K4d * tile >= opt.parallel_threshold)
#endif
    {
        PackedQLocalComplexThreadWorkspace ws_scalar;
        ws_scalar.ensure(K3d, K4d, L25);
        std::vector<double> Ctmp;
        double local_re_sum = 0.0;
        double local_im_sum = 0.0;
        std::uint64_t local_q_blocks = 0;
        std::uint64_t local_q_lanes = 0;

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (std::ptrdiff_t iq_s = 0; iq_s < static_cast<std::ptrdiff_t>(qs.size()); ++iq_s) {
            const CoherentC1QBlockSkeleton& q = qs[static_cast<std::size_t>(iq_s)];
            const double* Cbase = nullptr;
            if (q.direct_C) {
                Cbase = sj.sjC.data() + q.C_offset;
            } else {
                materialize_logical_sjC_panel_half(common, sj, q.C_block, q.rC, Ctmp);
                Cbase = Ctmp.data();
            }
            ++local_q_blocks;
            local_q_lanes += active_tile_lanes;

            for (std::size_t l = 0; l < tile; ++l) {
                if (!lanes[l].active_for_x) continue;
                const b4utils::Complex64 z = contract_q_block_streamed_adaptive_coherent_Cbase_skeleton(
                    common, sj, B,
                    lanes[l].sws.T1, lanes[l].vws.T1,
                    lanes[l].sws.T3, lanes[l].vws.T3,
                    q, Cbase, ws_scalar, nullptr);
                const b4utils::Complex64 c = lanes[l].coeff;
                const double ar = z.re;
                const double ai = z.im;
                local_re_sum += c.re * ar - c.im * ai;
                local_im_sum += c.re * ai + c.im * ar;
            }
        }

        {
            const std::lock_guard lock(accumulation_mutex);
            total_re += local_re_sum;
            total_im += local_im_sum;
            if (q_block_calls) *q_block_calls += local_q_blocks;
            if (q_lane_calls) *q_lane_calls += local_q_lanes;
        }
    }

    return {total_re, total_im};
}

void coherent_C1_print_tile_group_memory(std::ostream* log,
                                                const char* tag,
                                                const std::vector<std::vector<CoherentC1TileLaneState>>& tile_groups)
{
    if (!log) return;
    std::size_t tiles = tile_groups.size();
    std::size_t lanes_total = 0;
    std::size_t lanes_active = 0;
    std::uint64_t vec_capacity_bytes = 0;
    std::uint64_t T_values_capacity = 0;
    std::uint64_t T_support_model = 0;

    for (const auto& lanes : tile_groups) {
        lanes_total += lanes.size();
        vec_capacity_bytes = xqslab_sat_add_u64(
            vec_capacity_bytes,
            static_cast<std::uint64_t>(lanes.capacity() * sizeof(CoherentC1TileLaneState)));
        for (const CoherentC1TileLaneState& lane : lanes) {
            if (lane.active_for_x) ++lanes_active;
            if (!lane.active_for_x) continue;
            T_values_capacity = xqslab_sat_add_u64(T_values_capacity, static_cast<std::uint64_t>(lane.vws.T1.capacity_bytes()));
            T_values_capacity = xqslab_sat_add_u64(T_values_capacity, static_cast<std::uint64_t>(lane.vws.T3.capacity_bytes()));
            T_support_model = xqslab_sat_add_u64(T_support_model, lane.sws.T1.payload_bytes());
            T_support_model = xqslab_sat_add_u64(T_support_model, lane.sws.T3.payload_bytes());
            T_support_model = xqslab_sat_add_u64(T_support_model, lane.sws.T1.metadata_bytes());
            T_support_model = xqslab_sat_add_u64(T_support_model, lane.sws.T3.metadata_bytes());
        }
    }

    *log << "[coherent C1 tile memory] " << tag
         << " tiles=" << tiles
         << " lanes_total=" << lanes_total
         << " active_for_x=" << lanes_active
         << " vector_capacity="; xqslab_print_bytes(*log, vec_capacity_bytes);
    *log << " T_support_model="; xqslab_print_bytes(*log, T_support_model);
    *log << " T_values_capacity="; xqslab_print_bytes(*log, T_values_capacity);
    *log << "\n";
}

void coherent_C1_accumulate_shared_support_contraction_streamed_i1_tiles(
    const B4Spins& spins,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const B4CoherentCoefficientsView& C1,
    const CoherentC1InputSupportPlan_HalfSjC& support,
    const PackedCoherentC1SharedOptions& opt,
    WingExecutionOrder order,
    PackedContractionOptions contraction_options,
    PackedCoherentC1SharedResult& result,
    std::ostream* log)
{
    using clock = std::chrono::high_resolution_clock;
    if (log && opt.print_memory && opt.print_rss) {
        coherent_C1_print_memory_snapshot(log, "streamed-i1 shared contraction entry");
        coherent_C1_print_support_memory_breakdown(log, "streamed-i1 support", support);
    }

    const auto fill_t0 = clock::now();
    PackedAmplitudeSJTensors_HalfSjC sj;
    fill_packed_coherent_C1_sj_tensors_rec_half_sjC(spins, support, sj);
    result.sj_fill_seconds += std::chrono::duration<double>(clock::now() - fill_t0).count();
    if (log && opt.print_memory && opt.print_rss) {
        coherent_C1_print_sj_actual_buffers(log, "after streamed-i1 sj fill", sj);
        coherent_C1_print_memory_snapshot(log, "after streamed-i1 sj fill");
    }

    ++result.generic_allq_slabs;

    const std::size_t I1 = support.coherent_channels.i1.size();
    const std::size_t X = support.coherent_channels.xvals.size();
    const std::size_t tile_size = coherent_C1_select_i1_tile_size(support, C1, opt, contraction_options);
    const bool tile_auto = opt.auto_i1_tile_size || opt.i1_tile_size == 0;
    const bool accumulate_C1_inside_q = opt.accumulate_C1_inside_q && !opt.print_i1_timing;

    result.selected_i1_tile_size = tile_size;
    result.requested_i1_tile_size = opt.i1_tile_size;
    result.i1_tile_auto_selected = tile_auto;
    result.c1_accumulated_inside_q = accumulate_C1_inside_q;
    result.used_flat_scalar_q_loop = false;

    result.i1_x_pairs_total += static_cast<std::uint64_t>(I1) * static_cast<std::uint64_t>(X);
    result.i1_x_pairs_valid += static_cast<std::uint64_t>(support.i1_x_valid_pairs);
    result.i1_x_pairs_skipped += static_cast<std::uint64_t>(I1) * static_cast<std::uint64_t>(X)
                               - static_cast<std::uint64_t>(support.i1_x_valid_pairs);

    if (log && opt.print_plan) {
        *log << "[coherent C1 shared-streamed-i1] q mode=generic-allq"
             << " i1_tile=" << tile_size << (tile_auto ? "(auto)" : "(manual)")
             << " T-build=tiled"
             << " C1-accum=" << (accumulate_C1_inside_q ? "inside-q" : "per-tile-final")
             << " resident-lane-workspaces=streamed" << "\n";
    }
    if (log && opt.print_x_mask) {
        const double density = (I1 * X) == 0 ? 0.0
            : static_cast<double>(support.i1_x_valid_pairs) / static_cast<double>(I1 * X);
        *log << "[coherent C1 shared-streamed-i1] i1-x mask valid=" << support.i1_x_valid_pairs
             << "/" << (I1 * X)
             << " (" << std::fixed << std::setprecision(2) << (100.0 * density) << "%)\n";
    }

    PackedContractionOptions T_build_options = contraction_options;

    const auto ct0 = clock::now();
    for (std::size_t ix = 0; ix < X; ++ix) {
        const std::uint64_t progress_units = coherent_C1_progress_enabled(opt)
            ? coherent_C1_valid_i1_x_pairs_for_x(support, ix)
            : 0;
        const bool debug_this_x = (log && opt.print_memory && opt.print_rss && ix < opt.rss_debug_first_x);
        if (debug_this_x) coherent_C1_print_memory_snapshot(log, "streamed-i1 before x");

        std::vector<CoherentC1QBlockSkeleton> qs;
        bool qs_built = false;
        bool qs_empty = false;
        std::size_t active_for_x_total = 0;

        for (std::size_t a0 = 0; a0 < I1; a0 += tile_size) {
            const std::size_t i1_end = std::min<std::size_t>(I1, a0 + tile_size);
            std::vector<CoherentC1TileLaneState> lanes;

            const auto lane_t0 = clock::now();
            coherent_C1_build_tile_lanes(support, C1, a0, i1_end, opt.skip_zero_c1, lanes, false);
            result.lane_setup_seconds += std::chrono::duration<double>(clock::now() - lane_t0).count();
            if (lanes.empty()) continue;
            if (ix == 0) ++result.i1_tiles;

            const auto tb0 = clock::now();
            const std::size_t active_lanes = coherent_C1_tile_build_T_values_for_x(
                support, lanes, sj, packed_B, ix, order, T_build_options);
            result.T_build_seconds += std::chrono::duration<double>(clock::now() - tb0).count();
            if (active_lanes == 0) continue;

            active_for_x_total += active_lanes;
            result.i1_x_pairs_processed += static_cast<std::uint64_t>(active_lanes);

            if (!qs_built) {
                const auto qs0 = clock::now();
                coherent_C1_build_allq_skeletons_for_x(support.common, ix, qs);
                result.q_skeleton_seconds += std::chrono::duration<double>(clock::now() - qs0).count();
                result.q_skeletons_built += static_cast<std::uint64_t>(qs.size());
                for (const auto& q : qs) {
                    result.q_B3_row_descs += static_cast<std::uint64_t>(q.B3_rows.size());
                    result.q_B4_row_descs += static_cast<std::uint64_t>(q.B4_rows.size());
                }
                qs_built = true;
                qs_empty = qs.empty();
                if (debug_this_x) {
                    coherent_C1_print_q_skeleton_memory(log, "streamed-i1 q skeleton", qs);
                    coherent_C1_print_memory_snapshot(log, "streamed-i1 after q skeleton");
                }
                if (qs_empty) break;
            }

            const auto tq0 = clock::now();
            if (accumulate_C1_inside_q) {
                const b4utils::Complex64 z = coherent_C1_contract_qlocal_skeletons_tile_weighted(
                    support.common, sj, packed_B, lanes, qs, contraction_options,
                    &result.q_block_kernel_calls, &result.q_block_lane_calls);
                result.amplitude.re += z.re;
                result.amplitude.im += z.im;
            } else {
                coherent_C1_contract_qlocal_skeletons_tile(
                    support.common, sj, packed_B, lanes, qs, contraction_options,
                    &result.q_block_kernel_calls, &result.q_block_lane_calls);
                const auto ca0 = clock::now();
                for (const CoherentC1TileLaneState& lane : lanes) {
                    if (!lane.active_for_x) continue;
                    const b4utils::Complex64 c = lane.coeff;
                    const b4utils::Complex64 a_amp = lane.amplitude;
                    result.amplitude.re += c.re * a_amp.re - c.im * a_amp.im;
                    result.amplitude.im += c.re * a_amp.im + c.im * a_amp.re;
                }
                result.c1_accum_seconds += std::chrono::duration<double>(clock::now() - ca0).count();
            }
            result.q_loop_seconds += std::chrono::duration<double>(clock::now() - tq0).count();
            // lanes, sws, and vws die here. This is the entire point.
        }

        if (active_for_x_total == 0) {
            coherent_C1_progress_advance(opt, progress_units);
            continue;
        }
        if (debug_this_x) coherent_C1_print_memory_snapshot(log, "streamed-i1 after x");
        coherent_C1_progress_advance(opt, progress_units);
    }
    result.contract_seconds += std::chrono::duration<double>(clock::now() - ct0).count();

    if (log && opt.print_memory && opt.print_rss) {
        coherent_C1_print_memory_snapshot(log, "after streamed-i1 shared contraction");
    }
}

void coherent_C1_accumulate_shared_support_contraction(
    const B4Spins& spins,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const B4CoherentCoefficientsView& C1,
    const CoherentC1InputSupportPlan_HalfSjC& support,
    const PackedCoherentC1SharedOptions& opt,
    WingExecutionOrder order,
    PackedContractionOptions contraction_options,
    PackedCoherentC1SharedResult& result,
    std::ostream* log)
{
    using clock = std::chrono::high_resolution_clock;
    if (log && opt.print_memory && opt.print_rss) {
        coherent_C1_print_memory_snapshot(log, "shared contraction entry");
        coherent_C1_print_support_memory_breakdown(log, "shared contraction support", support);
    }

    const auto fill_t0 = clock::now();
    PackedAmplitudeSJTensors_HalfSjC sj;
    fill_packed_coherent_C1_sj_tensors_rec_half_sjC(spins, support, sj);
    result.sj_fill_seconds += std::chrono::duration<double>(clock::now() - fill_t0).count();
    if (log && opt.print_memory && opt.print_rss) {
        coherent_C1_print_sj_actual_buffers(log, "after shared sj fill", sj);
        coherent_C1_print_memory_snapshot(log, "after shared sj fill");
    }

    ++result.generic_allq_slabs;

    const std::size_t I1 = support.coherent_channels.i1.size();
    const std::size_t tile_size = coherent_C1_select_i1_tile_size(support, C1, opt, contraction_options);
    const bool tile_auto = opt.auto_i1_tile_size || opt.i1_tile_size == 0;
    const bool accumulate_C1_inside_q = opt.accumulate_C1_inside_q && !opt.print_i1_timing;
    result.selected_i1_tile_size = tile_size;
    result.requested_i1_tile_size = opt.i1_tile_size;
    result.i1_tile_auto_selected = tile_auto;
    result.c1_accumulated_inside_q = accumulate_C1_inside_q;
    result.used_flat_scalar_q_loop = accumulate_C1_inside_q;
    const std::size_t X = support.coherent_channels.xvals.size();

    result.i1_x_pairs_total += static_cast<std::uint64_t>(I1) * static_cast<std::uint64_t>(X);
    result.i1_x_pairs_valid += static_cast<std::uint64_t>(support.i1_x_valid_pairs);
    result.i1_x_pairs_skipped += static_cast<std::uint64_t>(I1) * static_cast<std::uint64_t>(X)
                               - static_cast<std::uint64_t>(support.i1_x_valid_pairs);

    if (log && opt.print_plan) {
        *log << "[coherent C1 shared] q mode=generic-allq"
             << " i1_tile=" << tile_size << (tile_auto ? "(auto)" : "(manual)")
             << " T-build=tiled"
             << " q-batch=off(stable-scalar)"
             << " flat-q=" << (accumulate_C1_inside_q ? "on" : "off")
             << " dense-C=nocheck C*T2-specialized"
             << " C1-accum=" << (accumulate_C1_inside_q ? "inside-q" : "final") << "\n";
    }
    if (log && opt.print_x_mask) {
        const double density = result.i1_x_pairs_total == 0 ? 0.0
            : static_cast<double>(support.i1_x_valid_pairs) /
              static_cast<double>(I1 * X);
        *log << "[coherent C1 shared] i1-x mask valid=" << support.i1_x_valid_pairs
             << "/" << (I1 * X)
             << " (" << std::fixed << std::setprecision(2) << (100.0 * density) << "%)\n";
    }

    PackedContractionOptions T_build_options = contraction_options;

    // Build i1 tiles once, then keep x as the outer loop.  sjB/sjC/sjD are
    // shared per x, and T1/T3 construction can reuse B5/sjB and B2/sjD loops
    // across the lanes in a tile.  A tiny triumph over the previous loop order.
    std::vector<std::vector<CoherentC1TileLaneState>> tile_groups;
    tile_groups.reserve((I1 + tile_size - 1) / tile_size);
    for (std::size_t a0 = 0; a0 < I1; a0 += tile_size) {
        const std::size_t i1_end = std::min<std::size_t>(I1, a0 + tile_size);
        std::vector<CoherentC1TileLaneState> lanes;
        const auto lane_t0 = clock::now();
        coherent_C1_build_tile_lanes(support, C1, a0, i1_end, opt.skip_zero_c1, lanes);
        result.lane_setup_seconds += std::chrono::duration<double>(clock::now() - lane_t0).count();
        if (lanes.empty()) continue;
        ++result.i1_tiles;
        tile_groups.emplace_back(std::move(lanes));
    }

    const auto ct0 = clock::now();
    for (std::size_t ix = 0; ix < X; ++ix) {
        const std::uint64_t progress_units = coherent_C1_progress_enabled(opt)
            ? coherent_C1_valid_i1_x_pairs_for_x(support, ix)
            : 0;
        const bool debug_this_x = (log && opt.print_memory && opt.print_rss && ix < opt.rss_debug_first_x);
        if (debug_this_x) {
            coherent_C1_print_memory_snapshot(log, "before T1/T3 build for x");
        }
        std::size_t active_for_x_total = 0;
        for (auto& lanes : tile_groups) {
            const auto tb0 = clock::now();
            const std::size_t active_lanes = coherent_C1_tile_build_T_values_for_x(
                support, lanes, sj, packed_B, ix, order, T_build_options);
            result.T_build_seconds += std::chrono::duration<double>(clock::now() - tb0).count();
            active_for_x_total += active_lanes;
            result.i1_x_pairs_processed += static_cast<std::uint64_t>(active_lanes);
        }
        if (debug_this_x) {
            coherent_C1_print_tile_group_memory(log, "after T1/T3 build for x", tile_groups);
            coherent_C1_print_memory_snapshot(log, "after T1/T3 build for x");
        }
        if (active_for_x_total == 0) {
            coherent_C1_progress_advance(opt, progress_units);
            continue;
        }

        // Step 1/2: build the x-local q skeleton and B3/B4 row descriptors once.
        // All i1 tiles reuse this list.  The old path rediscovered the same
        // q-grid once per tile, because apparently the CPU needed busywork.
        std::vector<CoherentC1QBlockSkeleton> qs;
        const auto qs0 = clock::now();
        coherent_C1_build_allq_skeletons_for_x(support.common, ix, qs);
        result.q_skeleton_seconds += std::chrono::duration<double>(clock::now() - qs0).count();
        result.q_skeletons_built += static_cast<std::uint64_t>(qs.size());
        for (const auto& q : qs) {
            result.q_B3_row_descs += static_cast<std::uint64_t>(q.B3_rows.size());
            result.q_B4_row_descs += static_cast<std::uint64_t>(q.B4_rows.size());
        }
        if (debug_this_x) {
            coherent_C1_print_q_skeleton_memory(log, "after q skeleton build", qs);
            coherent_C1_print_memory_snapshot(log, "after q skeleton build");
        }
        if (qs.empty()) {
            coherent_C1_progress_advance(opt, progress_units);
            continue;
        }

        const auto tq0 = clock::now();
        if (accumulate_C1_inside_q) {
            const b4utils::Complex64 z = coherent_C1_contract_qlocal_skeletons_flat_scalar_weighted(
                support.common, sj, packed_B, tile_groups, qs, contraction_options,
                opt.flat_q_task_chunk,
                &result.q_block_kernel_calls, &result.q_block_lane_calls);
            result.amplitude.re += z.re;
            result.amplitude.im += z.im;
        } else {
            for (auto& lanes : tile_groups) {
                coherent_C1_contract_qlocal_skeletons_tile(
                    support.common, sj, packed_B, lanes, qs, contraction_options,
                    &result.q_block_kernel_calls, &result.q_block_lane_calls);
            }
        }
        result.q_loop_seconds += std::chrono::duration<double>(clock::now() - tq0).count();
        if (debug_this_x) {
            coherent_C1_print_memory_snapshot(log, "after q-loop for x");
        }
        coherent_C1_progress_advance(opt, progress_units);
    }
    result.contract_seconds += std::chrono::duration<double>(clock::now() - ct0).count();

    const auto ca0 = clock::now();
    if (!accumulate_C1_inside_q) {
        for (const auto& lanes : tile_groups) {
            for (const CoherentC1TileLaneState& lane : lanes) {
                const b4utils::Complex64 c = lane.coeff;
                const b4utils::Complex64 a_amp = lane.amplitude;
                result.amplitude.re += c.re * a_amp.re - c.im * a_amp.im;
                result.amplitude.im += c.re * a_amp.im + c.im * a_amp.re;

                if (log && opt.print_i1_timing) {
                    *log << "[coherent C1:shared] i1=" << lane.Di1
                         << " C1=(" << c.re << ", " << c.im << ")\n"
                         << "  A(i1)=(" << std::scientific << std::setprecision(17)
                         << a_amp.re << ", " << a_amp.im << ") cumulative=("
                         << result.amplitude.re << ", " << result.amplitude.im << ")"
                         << std::fixed << std::setprecision(6) << "\n";
                }
            }
        }
    }
    result.c1_accum_seconds += std::chrono::duration<double>(clock::now() - ca0).count();
}

B4CoherentCoefficientsView coherent_C1_make_pruned_view(
    const B4CoherentCoefficientsView& C1,
    const PackedCoherentC1SharedOptions& opt,
    PackedCoherentC1SharedResult& result,
    std::vector<int>& kept_iotas,
    std::vector<b4utils::Complex64>& kept_coeffs)
{
    kept_iotas.clear();
    kept_coeffs.clear();

    const std::size_t n = C1.size();
    result.C1_original_count = n;
    result.C1_prune_relative_tol_used = std::max(0.0, opt.C1_prune_relative_tol);
    result.C1_prune_absolute_tol_used = std::max(0.0, opt.C1_prune_absolute_tol);
    result.C1_pruning_enabled = (result.C1_prune_relative_tol_used > 0.0 ||
                                 result.C1_prune_absolute_tol_used > 0.0);

    if (n == 0) {
        result.C1_kept_count = 0;
        return make_b4_coherent_coefficients_view(kept_iotas, kept_coeffs);
    }

    std::vector<double> absC(n, 0.0);
    std::size_t imax = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = coherent_C1_abs_coeff(C1.coefficients[i]);
        absC[i] = a;
        result.C1_l1_total += a;
        if (a > result.C1_max_abs) {
            result.C1_max_abs = a;
            imax = i;
        }
    }

    const double rel_cut = result.C1_prune_relative_tol_used * result.C1_max_abs;
    const double abs_cut = result.C1_prune_absolute_tol_used;
    const double cut = std::max(rel_cut, abs_cut);

    kept_iotas.reserve(n);
    kept_coeffs.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        const double a = absC[i];
        bool keep = true;
        if (opt.skip_zero_c1 && a == 0.0) keep = false;
        if (cut > 0.0 && a < cut) keep = false;

        if (keep) {
            kept_iotas.push_back(C1.doubled_iotas[i]);
            kept_coeffs.push_back(C1.coefficients[i]);
        } else {
            result.C1_l1_dropped += a;
        }
    }

    if (kept_iotas.empty() && opt.C1_prune_keep_largest_if_empty && result.C1_max_abs > 0.0) {
        kept_iotas.push_back(C1.doubled_iotas[imax]);
        kept_coeffs.push_back(C1.coefficients[imax]);
        result.C1_l1_dropped -= absC[imax];
        result.C1_prune_kept_largest_fallback = true;
    }

    result.C1_kept_count = kept_iotas.size();
    result.C1_pruned_count = (n >= kept_iotas.size()) ? (n - kept_iotas.size()) : 0;
    result.C1_l1_dropped_fraction = result.C1_l1_total > 0.0
        ? result.C1_l1_dropped / result.C1_l1_total
        : 0.0;

    return make_b4_coherent_coefficients_view(kept_iotas, kept_coeffs);
}

CoherentC1SharedXPreflight coherent_C1_shared_preflight_one_x_slices(
    const B4Spins& spins,
    int Dl,
    const CoherentAmplitudeChannels& ch,
    const std::uint64_t fixed_bytes,
    const std::uint64_t budget,
    PackedCoherentC1SharedResult& result,
    double& support_seconds_accum)
{
    using clock = std::chrono::high_resolution_clock;

    CoherentC1SharedXPreflight pf;
    pf.fixed_bytes = fixed_bytes;
    pf.budget_bytes = budget;

    const std::size_t X = ch.xvals.size();
    pf.incremental_by_x.assign(X, 0);
    pf.i1_values = ch.i1;
    pf.valid_x_pairs_by_i1.assign(ch.i1.size(), 0);

    for (std::size_t ix = 0; ix < X; ++ix) {
        const CoherentAmplitudeChannels one = coherent_C1_make_x_subchannels(ch, ix, ix + 1);
        const auto t0 = clock::now();
        CoherentC1InputSupportPlan_HalfSjC support =
            build_coherent_C1_input_support_plan_half_sjC(spins, Dl, one);
        support_seconds_accum += std::chrono::duration<double>(clock::now() - t0).count();

        const std::uint64_t inc = coherent_C1_shared_incremental_bytes(support);
        pf.incremental_by_x[ix] = inc;
        const std::uint64_t valid_pairs =
            static_cast<std::uint64_t>(support.i1_x_valid_pairs);
        for (std::size_t ia = 0; ia < ch.i1.size(); ++ia) {
            if (coherent_C1_i1_x_valid(support, ia, 0)) {
                ++pf.valid_x_pairs_by_i1[ia];
            }
        }
        if (valid_pairs > std::numeric_limits<std::uint64_t>::max() -
                              pf.valid_i1_x_pairs_total) {
            throw std::overflow_error(
                "coherent C1 progress size overflows uint64_t");
        }
        pf.valid_i1_x_pairs_total += valid_pairs;
        pf.full_incremental_upper = xqslab_sat_add_u64(pf.full_incremental_upper, inc);
        pf.max_single_incremental = std::max(pf.max_single_incremental, inc);
        pf.max_single_peak = std::max(pf.max_single_peak, xqslab_sat_add_u64(fixed_bytes, inc));

        result.incremental_bytes = std::max(result.incremental_bytes, inc);
        result.estimated_peak_bytes = std::max(result.estimated_peak_bytes,
                                               xqslab_sat_add_u64(fixed_bytes, inc));
    }

    result.incremental_bytes = std::max(result.incremental_bytes, pf.full_incremental_upper);
    result.estimated_peak_bytes = std::max(result.estimated_peak_bytes, pf.full_peak_upper());
    return pf;
}

void coherent_C1_shared_accumulate_xq_fallback_for_all_i1(
    const B4Spins& spins,
    int Dl,
    const std::array<int,10>& spins_array,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const B4CoherentCoefficientsView& C1,
    const PackedCoherentC1SharedOptions& opt,
    const CoherentC1SharedXPreflight& progress_plan,
    WingExecutionOrder order,
    PackedContractionOptions contraction_options,
    PackedCoherentC1SharedResult& result,
    std::ostream* log)
{
    PackedCoherentI1BatchOptions xq_opt;
    xq_opt.xq = opt.xq;
    xq_opt.print_plan = opt.print_xq_slab_summary;
    xq_opt.print_i1_timing = opt.print_i1_timing;
    xq_opt.skip_zero_c1 = opt.skip_zero_c1;
    xq_opt.require_same_booster_support = true;
    if (coherent_C1_progress_enabled(opt)) {
        xq_opt.progress_context = opt.progress_context;
        xq_opt.progress_advance = opt.progress_advance;
        xq_opt.progress_i1_values = &progress_plan.i1_values;
        xq_opt.progress_units_by_i1 =
            &progress_plan.valid_x_pairs_by_i1;
    }

    PackedCoherentI1BatchResult xq_res =
        contract_packed_amplitude_coherent_C1_streamed_xqslab(
            spins, Dl, spins_array, packed_B, C1, xq_opt, order, contraction_options, log);

    result.amplitude.re += xq_res.amplitude.re;
    result.amplitude.im += xq_res.amplitude.im;
    result.xq_slices += xq_res.active_i1;
    result.xq_max_estimated_peak_bytes =
        std::max(result.xq_max_estimated_peak_bytes, xq_res.max_estimated_peak_bytes);
    result.estimated_peak_bytes =
        std::max(result.estimated_peak_bytes, xq_res.max_estimated_peak_bytes);
    result.support_seconds += xq_res.support_seconds;
    result.sj_fill_seconds += xq_res.sj_fill_seconds;
    result.contract_seconds += std::max(
        0.0, xq_res.sj_contract_seconds - xq_res.sj_fill_seconds);
}

PackedCoherentC1SharedResult contract_packed_amplitude_coherent_C1_shared_auto(
    const B4Spins& spins,
    int Dl,
    const std::array<int,10>& spins_array,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const B4CoherentCoefficientsView& C1,
    const PackedCoherentC1SharedOptions& opt,
    WingExecutionOrder order,
    PackedContractionOptions contraction_options,
    std::ostream* log)
{
    using clock = std::chrono::high_resolution_clock;
    const auto total_t0 = clock::now();
    PackedCoherentC1SharedResult result;

    b4_assert_valid_coherent_coefficients(C1);
    if (log && opt.print_memory && opt.print_rss) {
        coherent_C1_print_memory_snapshot(log, "shared_auto entry");
    }

    std::vector<int> c1_iotas;
    std::vector<b4utils::Complex64> c1_coeffs;
    const B4CoherentCoefficientsView C1_active =
        coherent_C1_make_pruned_view(C1, opt, result, c1_iotas, c1_coeffs);

    if (log && opt.print_plan && opt.print_C1_prune_summary &&
        (result.C1_pruning_enabled || result.C1_original_count != result.C1_kept_count)) {
        *log << "[coherent C1 prune] kept " << result.C1_kept_count << "/" << result.C1_original_count
             << " dropped=" << result.C1_pruned_count
             << " rel_tol=" << std::scientific << std::setprecision(3) << result.C1_prune_relative_tol_used
             << " abs_tol=" << result.C1_prune_absolute_tol_used
             << " dropped_l1_frac=" << result.C1_l1_dropped_fraction
             << (result.C1_prune_kept_largest_fallback ? " kept-largest-fallback" : "")
             << std::fixed << std::setprecision(6) << "\n";
    }

    if (log && opt.print_memory && opt.print_rss) {
        coherent_C1_print_memory_snapshot(log, "after C1 prune/view");
    }

    const auto support_t0 = clock::now();
    const CoherentAmplitudeChannels ch = make_CoherentAmplitudeChannels(spins, Dl, c1_iotas);
    result.active_i1 = ch.i1.size();
    result.support_seconds += std::chrono::duration<double>(clock::now() - support_t0).count();

    if (ch.empty()) {
        result.total_seconds = std::chrono::duration<double>(clock::now() - total_t0).count();
        return result;
    }

    const std::size_t X = ch.xvals.size();
    const std::uint64_t fixed_bytes = coherent_C1_shared_fixed_bytes(packed_B, opt.xq);
    const std::uint64_t budget = xqslab_memory_budget(opt.xq);

    if (log && opt.print_memory && opt.print_rss) {
        coherent_C1_print_model_vs_rss(log, "after channel build/fixed model", fixed_bytes, 0, fixed_bytes, budget);
    }

    if (!opt.enable_full_x && !opt.enable_x_slab && !opt.enable_xq_slab) {
        throw b4utils::UnsupportedOperationFailure(
            "coherent C1 shared contraction: full-x, x-slab and x+q fallback are all disabled");
    }

    CoherentC1SharedXPreflight pf;
    bool have_preflight = false;
    const bool progress_enabled = coherent_C1_progress_enabled(opt);

    // The previous dispatcher built full-X support first and only then checked
    // the budget.  That is exactly backward for the pathological cases.  The
    // safe preflight builds one x-slice at a time, which is the smallest shared
    // support object; it gives a conservative full-X upper bound and a greedy
    // slab-packing model without ever materializing the all-X monster.
    if (opt.preflight_full_x_without_full_support ||
        (opt.auto_x_slab_size && opt.max_x_per_slab == std::numeric_limits<std::size_t>::max()) ||
        progress_enabled)
    {
        double preflight_support_seconds = 0.0;
        pf = coherent_C1_shared_preflight_one_x_slices(
            spins, Dl, ch, fixed_bytes, budget, result, preflight_support_seconds);
        result.support_seconds += preflight_support_seconds;
        have_preflight = true;
        if (log && opt.print_memory && opt.print_rss) {
            coherent_C1_print_model_vs_rss(log, "after one-x preflight", fixed_bytes, pf.full_incremental_upper, pf.full_peak_upper(), budget);
        }
    }

    if (progress_enabled) {
        opt.progress_begin(
            opt.progress_context, pf.valid_i1_x_pairs_total);
    }

    const std::uint64_t full_preflight_peak = have_preflight
        ? pf.full_peak_upper()
        : std::numeric_limits<std::uint64_t>::max();

    if (log && opt.print_plan) {
        *log << "\n[coherent C1 shared] active i1=" << ch.i1.size()
             << " X=" << X
             << " fixed="; xqslab_print_bytes(*log, fixed_bytes);
        *log << " budget="; xqslab_print_bytes(*log, budget);
        if (have_preflight) {
            *log << " preflight full-upper="; xqslab_print_bytes(*log, full_preflight_peak);
            *log << " max-single="; xqslab_print_bytes(*log, pf.max_single_peak);
        }
        *log << " mode=";
        if (opt.enable_full_x && (!have_preflight || full_preflight_peak <= budget)) {
            *log << "try-full-x";
        } else if (opt.enable_x_slab) {
            *log << "auto-x-slab";
        } else {
            *log << "x+q-fallback";
        }
        *log << "\n";
    }

    bool full_x_done = false;
    if (opt.enable_full_x && (!have_preflight || full_preflight_peak <= budget)) {
        if (log && opt.print_memory && opt.print_rss) {
            coherent_C1_print_memory_snapshot(log, "before full-x support build");
        }
        const auto st0 = clock::now();
        CoherentC1InputSupportPlan_HalfSjC full_support =
            build_coherent_C1_input_support_plan_half_sjC(spins, Dl, ch);
        result.support_seconds += std::chrono::duration<double>(clock::now() - st0).count();

        if (log && opt.print_memory && opt.print_rss) {
            coherent_C1_print_support_memory_breakdown(log, "full-x support", full_support);
            coherent_C1_print_memory_snapshot(log, "after full-x support build");
        }

        const std::uint64_t full_inc = coherent_C1_shared_incremental_bytes(full_support);
        const std::uint64_t full_peak = xqslab_sat_add_u64(fixed_bytes, full_inc);
        result.incremental_bytes = std::max(result.incremental_bytes, full_inc);
        result.estimated_peak_bytes = std::max(result.estimated_peak_bytes, full_peak);

        if (log && opt.print_plan) {
            *log << "[coherent C1 shared] full-x actual incremental=";
            xqslab_print_bytes(*log, full_inc);
            *log << " peak="; xqslab_print_bytes(*log, full_peak);
            *log << (full_peak <= budget ? " -> selected\n" : " -> rejected, slicing\n");
        }

        if (full_peak <= budget) {
            const std::uint64_t lane_ws = coherent_C1_resident_lane_workspace_estimate_bytes(
                full_support, full_support.coherent_channels.i1.size());
            const std::uint64_t resident_peak = xqslab_sat_add_u64(full_peak, lane_ws);
            const bool use_streamed = opt.enable_streamed_i1_tiles &&
                !coherent_C1_resident_lane_workspace_fits(
                    full_peak, lane_ws, budget, opt.resident_lane_workspace_budget_fraction);

            ++result.full_x_slabs;
            result.estimated_peak_bytes = std::max(result.estimated_peak_bytes, resident_peak);
            if (log && opt.print_plan && opt.print_streamed_i1_summary) {
                *log << "[coherent C1 shared] resident lane workspace estimate=";
                xqslab_print_bytes(*log, lane_ws);
                *log << " resident peak="; xqslab_print_bytes(*log, resident_peak);
                *log << (use_streamed ? " -> streamed-i1\n" : " -> resident-fast\n");
            }
            if (use_streamed) {
                coherent_C1_accumulate_shared_support_contraction_streamed_i1_tiles(
                    spins, packed_B, C1_active, full_support,
                    opt, order, contraction_options, result, log);
            } else {
                coherent_C1_accumulate_shared_support_contraction(
                    spins, packed_B, C1_active, full_support,
                    opt, order, contraction_options, result, log);
            }
            full_x_done = true;
        }
        // full_support is destroyed here before x-slab fallback allocates.
    }

    if (!full_x_done) {
        if (opt.enable_x_slab) {
            if (!have_preflight) {
                double preflight_support_seconds = 0.0;
                pf = coherent_C1_shared_preflight_one_x_slices(
                    spins, Dl, ch, fixed_bytes, budget, result, preflight_support_seconds);
                result.support_seconds += preflight_support_seconds;
                have_preflight = true;
            }

            if (pf.max_single_peak > budget || fixed_bytes > budget) {
                if (!opt.enable_xq_slab) {
                    throw b4utils::ResourceLimitFailure("coherent C1 shared contraction: one x-slice exceeds budget and x+q fallback is disabled");
                }
                if (log && opt.print_plan) {
                    *log << "[coherent C1 shared] max single shared x peak=";
                    xqslab_print_bytes(*log, pf.max_single_peak);
                    *log << " exceeds budget="; xqslab_print_bytes(*log, budget);
                    *log << " -> per-i1 x+q fallback for whole C1 sum\n";
                }
                if (log && opt.print_memory && opt.print_rss) {
                    coherent_C1_print_memory_snapshot(log, "before all-i1 x+q fallback");
                }
                coherent_C1_shared_accumulate_xq_fallback_for_all_i1(
                    spins, Dl, spins_array, packed_B, C1_active,
                    opt, pf, order, contraction_options, result, log);

                result.total_seconds = std::chrono::duration<double>(clock::now() - total_t0).count();
                if (log && opt.print_plan) result.print(*log);
                return result;
            }

            const bool explicit_x_cap =
                opt.max_x_per_slab != std::numeric_limits<std::size_t>::max();
            const std::size_t user_cap = explicit_x_cap
                ? std::max<std::size_t>(1, opt.max_x_per_slab)
                : X;

            std::size_t x = 0;
            while (x < X) {
                std::size_t x_end = x + 1;
                if (explicit_x_cap || !opt.auto_x_slab_size) {
                    x_end = std::min<std::size_t>(X, x + user_cap);
                } else {
                    x_end = coherent_C1_shared_choose_auto_x_end_from_preflight(
                        pf, x, X, opt.auto_x_slab_budget_fraction);
                }

                CoherentC1InputSupportPlan_HalfSjC slab_support;
                std::uint64_t slab_peak = 0;

                while (true) {
                    const CoherentAmplitudeChannels sub = coherent_C1_make_x_subchannels(ch, x, x_end);
                    if (log && opt.print_memory && opt.print_rss) {
                        coherent_C1_print_memory_snapshot(log, "before x slab support build");
                    }
                    const auto st0 = clock::now();
                    slab_support = build_coherent_C1_input_support_plan_half_sjC(spins, Dl, sub);
                    result.support_seconds += std::chrono::duration<double>(clock::now() - st0).count();

                    if (log && opt.print_memory && opt.print_rss) {
                        coherent_C1_print_support_memory_breakdown(log, "x slab support", slab_support);
                        coherent_C1_print_memory_snapshot(log, "after x slab support build");
                    }

                    const std::uint64_t inc = coherent_C1_shared_incremental_bytes(slab_support);
                    slab_peak = xqslab_sat_add_u64(fixed_bytes, inc);
                    result.incremental_bytes = std::max(result.incremental_bytes, inc);
                    result.estimated_peak_bytes = std::max(result.estimated_peak_bytes, slab_peak);

                    if (slab_peak <= budget || x_end == x + 1) break;
                    x_end = x + std::max<std::size_t>(1, (x_end - x) / 2);
                    // Release the rejected support before constructing its
                    // replacement. Move-assignment alone evaluates/builds the
                    // RHS first and can otherwise double the transient peak.
                    slab_support = CoherentC1InputSupportPlan_HalfSjC{};
                }

                const bool one_x_too_large = (slab_peak > budget && x_end == x + 1);
                if (one_x_too_large) {
                    // This should have been caught by the one-x preflight before
                    // any partial x-slab contribution was accumulated.  Falling
                    // back to the all-i1 x+q path here would double-count already
                    // processed x-slabs, so report the inconsistent plan rather
                    // than returning a double-counted result.
                    throw b4utils::ResourceLimitFailure("coherent C1 shared contraction: one x-slice exceeded budget after partial x-slab planning");
                }

                ++result.x_slabs;
                if (log && opt.print_plan) {
                    *log << "[coherent C1 shared] x slab [" << x << "," << x_end << ") peak=";
                    xqslab_print_bytes(*log, slab_peak);
                    if (!explicit_x_cap && opt.auto_x_slab_size) {
                        *log << " auto";
                    }
                    *log << "\n";
                }

                if (log && opt.print_memory && opt.print_rss) {
                    coherent_C1_print_model_vs_rss(log, "before x slab contraction", fixed_bytes, coherent_C1_shared_incremental_bytes(slab_support), slab_peak, budget);
                }
                {
                    const std::uint64_t lane_ws = coherent_C1_resident_lane_workspace_estimate_bytes(
                        slab_support, slab_support.coherent_channels.i1.size());
                    const std::uint64_t resident_peak = xqslab_sat_add_u64(slab_peak, lane_ws);
                    const bool use_streamed = opt.enable_streamed_i1_tiles &&
                        !coherent_C1_resident_lane_workspace_fits(
                            slab_peak, lane_ws, budget, opt.resident_lane_workspace_budget_fraction);
                    result.estimated_peak_bytes = std::max(result.estimated_peak_bytes, resident_peak);
                    if (log && opt.print_plan && opt.print_streamed_i1_summary) {
                        *log << "[coherent C1 shared] x slab resident lane workspace estimate=";
                        xqslab_print_bytes(*log, lane_ws);
                        *log << " resident peak="; xqslab_print_bytes(*log, resident_peak);
                        *log << (use_streamed ? " -> streamed-i1\n" : " -> resident-fast\n");
                    }
                    if (use_streamed) {
                        coherent_C1_accumulate_shared_support_contraction_streamed_i1_tiles(
                            spins, packed_B, C1_active, slab_support,
                            opt, order, contraction_options, result, log);
                    } else {
                        coherent_C1_accumulate_shared_support_contraction(
                            spins, packed_B, C1_active, slab_support,
                            opt, order, contraction_options, result, log);
                    }
                }
                if (log && opt.print_memory && opt.print_rss) {
                    coherent_C1_print_memory_snapshot(log, "after x slab contraction returns");
                }
                x = x_end;
            }
        } else {
            if (!opt.enable_xq_slab) {
                throw b4utils::ResourceLimitFailure("coherent C1 shared contraction: full-x rejected and no slicing fallback is enabled");
            }
            if (log && opt.print_plan) {
                *log << "[coherent C1 shared] x-slab disabled -> per-i1 x+q fallback\n";
            }
            if (log && opt.print_memory && opt.print_rss) {
                coherent_C1_print_memory_snapshot(log, "before all-i1 x+q fallback");
            }
            coherent_C1_shared_accumulate_xq_fallback_for_all_i1(
                spins, Dl, spins_array, packed_B, C1_active,
                opt, pf, order, contraction_options, result, log);
        }
    }

    result.total_seconds = std::chrono::duration<double>(clock::now() - total_t0).count();
    if (log && opt.print_plan) result.print(*log);
    return result;
}

} // namespace b4support
