// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/AmplitudeInputSupport.hpp"

namespace b4support {

void finalize_offsets(RangeBlockSupport1D& s)
{
    const std::size_t n = s.blocks.size();

#ifndef _OPENMP
    std::uint64_t offset = 0;

    for (Range1D& block : s.blocks) {
        block.offset = offset;
        offset += block.count();
    }

    s.packed_values = offset;
#else
    // For small metadata arrays, OpenMP launch and synchronization overhead
    // costs more than the serial prefix pass.
    if (n < 32768) {
        std::uint64_t offset = 0;

        for (Range1D& block : s.blocks) {
            block.offset = offset;
            offset += block.count();
        }

        s.packed_values = offset;
        return;
    }

    const int max_threads = omp_get_max_threads();
    const int nchunks = static_cast<int>(
        std::min<std::size_t>(static_cast<std::size_t>(max_threads), n)
    );

    std::vector<std::uint64_t> chunk_offsets(
        static_cast<std::size_t>(nchunks) + 1u,
        0ull
    );

#pragma omp parallel for schedule(static)
    for (int c = 0; c < nchunks; ++c) {
        const std::size_t begin =
            (static_cast<std::size_t>(c) * n) /
            static_cast<std::size_t>(nchunks);

        const std::size_t end =
            (static_cast<std::size_t>(c + 1) * n) /
            static_cast<std::size_t>(nchunks);

        std::uint64_t local = 0;

        for (std::size_t i = begin; i < end; ++i) {
            local += s.blocks[i].count();
        }

        chunk_offsets[static_cast<std::size_t>(c) + 1u] = local;
    }

    for (int c = 0; c < nchunks; ++c) {
        chunk_offsets[static_cast<std::size_t>(c) + 1u] +=
            chunk_offsets[static_cast<std::size_t>(c)];
    }

#pragma omp parallel for schedule(static)
    for (int c = 0; c < nchunks; ++c) {
        const std::size_t begin =
            (static_cast<std::size_t>(c) * n) /
            static_cast<std::size_t>(nchunks);

        const std::size_t end =
            (static_cast<std::size_t>(c + 1) * n) /
            static_cast<std::size_t>(nchunks);

        std::uint64_t offset = chunk_offsets[static_cast<std::size_t>(c)];

        for (std::size_t i = begin; i < end; ++i) {
            s.blocks[i].offset = offset;
            offset += s.blocks[i].count();
        }
    }

    s.packed_values = chunk_offsets[static_cast<std::size_t>(nchunks)];
#endif
}

void finalize_offsets(RangeBlockSupport2D& s)
{
    const std::size_t n = s.blocks.size();

#ifndef _OPENMP
    std::uint64_t offset = 0;

    for (Range2D& block : s.blocks) {
        block.offset = offset;
        offset += block.count();
    }

    s.packed_values = offset;
#else
    if (n < 32768) {
        std::uint64_t offset = 0;

        for (Range2D& block : s.blocks) {
            block.offset = offset;
            offset += block.count();
        }

        s.packed_values = offset;
        return;
    }

    const int max_threads = omp_get_max_threads();
    const int nchunks = static_cast<int>(
        std::min<std::size_t>(static_cast<std::size_t>(max_threads), n)
    );

    std::vector<std::uint64_t> chunk_offsets(
        static_cast<std::size_t>(nchunks) + 1u,
        0ull
    );

#pragma omp parallel for schedule(static)
    for (int c = 0; c < nchunks; ++c) {
        const std::size_t begin =
            (static_cast<std::size_t>(c) * n) /
            static_cast<std::size_t>(nchunks);

        const std::size_t end =
            (static_cast<std::size_t>(c + 1) * n) /
            static_cast<std::size_t>(nchunks);

        std::uint64_t local = 0;

        for (std::size_t i = begin; i < end; ++i) {
            local += s.blocks[i].count();
        }

        chunk_offsets[static_cast<std::size_t>(c) + 1u] = local;
    }

    for (int c = 0; c < nchunks; ++c) {
        chunk_offsets[static_cast<std::size_t>(c) + 1u] +=
            chunk_offsets[static_cast<std::size_t>(c)];
    }

#pragma omp parallel for schedule(static)
    for (int c = 0; c < nchunks; ++c) {
        const std::size_t begin =
            (static_cast<std::size_t>(c) * n) /
            static_cast<std::size_t>(nchunks);

        const std::size_t end =
            (static_cast<std::size_t>(c + 1) * n) /
            static_cast<std::size_t>(nchunks);

        std::uint64_t offset = chunk_offsets[static_cast<std::size_t>(c)];

        for (std::size_t i = begin; i < end; ++i) {
            s.blocks[i].offset = offset;
            offset += s.blocks[i].count();
        }
    }

    s.packed_values = chunk_offsets[static_cast<std::size_t>(nchunks)];
#endif
}

AmplitudeInputSupportPlan build_amplitude_input_support_plan(
                                                                    const B4Spins& s,
                                                                    int Dl,
                                                                    const AmplitudeChannels& AmC)
{
    AmplitudeInputSupportPlan plan;
    plan.channels = AmC;

    // ---------------------------------------------------------------------
    // Dimensions and axes
    // ---------------------------------------------------------------------

    plan.l23 = make_l_axis(s.Dj23, Dl);
    plan.l24 = make_l_axis(s.Dj24, Dl);
    plan.l25 = make_l_axis(s.Dj25, Dl);
    plan.l34 = make_l_axis(s.Dj34, Dl);
    plan.l35 = make_l_axis(s.Dj35, Dl);
    plan.l45 = make_l_axis(s.Dj45, Dl);

    const std::vector<int>& xvals = AmC.xvals;

    const std::vector<int>& k2 = AmC.k2;
    const std::vector<int>& k3 = AmC.k3;
    const std::vector<int>& k4 = AmC.k4;
    const std::vector<int>& k5 = AmC.k5;

    const std::vector<int>& l23 = plan.l23;
    const std::vector<int>& l24 = plan.l24;
    const std::vector<int>& l25 = plan.l25;
    const std::vector<int>& l34 = plan.l34;
    const std::vector<int>& l35 = plan.l35;
    const std::vector<int>& l45 = plan.l45;

    const std::size_t X   = xvals.size();

    const std::size_t K2d = k2.size();
    const std::size_t K3d = k3.size();
    const std::size_t K4d = k4.size();
    const std::size_t K5d = k5.size();

    const std::size_t L23 = l23.size();
    const std::size_t L24 = l24.size();
    const std::size_t L25 = l25.size();
    const std::size_t L34 = l34.size();
    const std::size_t L35 = l35.size();
    const std::size_t L45 = l45.size();

    // Fixed boundary intertwiner values for amplitude mode.
    const int Di1 = AmC.i[0];
    const int Di2 = AmC.i[1];
    const int Di3 = AmC.i[2];
    const int Di4 = AmC.i[3];
    const int Di5 = AmC.i[4];

    (void)Di2;
    (void)Di3;
    (void)Di4;
    (void)Di5;

    // ---------------------------------------------------------------------
    // Allocate support metadata.
    //
    // The outer block layouts are chosen to match the current Plan-A kernels.
    // ---------------------------------------------------------------------

    plan.sjA = make_support_1d(
                               "sjA support: block (x,l25) -> k5",
                               {X, L25},
                               K5d
                               );

    plan.sjB = make_support_1d(
                               "sjB support: block (x,k4,l35,l45) -> k5",
                               {X, K4d, L35, L45},
                               K5d
                               );

    plan.sjC = make_support_2d(
                               "sjC support: block (x,l24,l34,l35) -> k3 x k4",
                               {X, L24, L34, L35},
                               K3d,
                               K4d
                               );

    plan.sjD = make_support_1d(
                               "sjD support: block (x,k3,l23,l24) -> k2",
                               {X, K3d, L23, L24},
                               K2d
                               );

    plan.sjE = make_support_1d(
                               "sjE support: block (x,l25) -> k2",
                               {X, L25},
                               K2d
                               );

    plan.B2 = make_support_1d(
                              "B2 support: block (l23,l24,l25) -> k2",
                              {L23, L24, L25},
                              K2d
                              );

    plan.B3 = make_support_1d(
                              "B3 support: block (k3,l34,l35) -> l23",
                              {K3d, L34, L35},
                              L23
                              );

    plan.B4 = make_support_1d(
                              "B4 support: block (k4,l24,l34) -> l45",
                              {K4d, L24, L34},
                              L45
                              );

    plan.B5 = make_support_1d(
                              "B5 support: block (l25,l35,l45) -> k5",
                              {L25, L35, L45},
                              K5d
                              );

    // If the amplitude channels are empty, leave all blocks empty.
    if (X == 0 || K2d == 0 || K3d == 0 || K4d == 0 || K5d == 0) {
        finalize_offsets(plan.sjA);
        finalize_offsets(plan.sjB);
        finalize_offsets(plan.sjC);
        finalize_offsets(plan.sjD);
        finalize_offsets(plan.sjE);
        finalize_offsets(plan.B2);
        finalize_offsets(plan.B3);
        finalize_offsets(plan.B4);
        finalize_offsets(plan.B5);
        return plan;
    }

    // ---------------------------------------------------------------------
    // sjA(x,l25) -> k5 range
    //
    // nonzero if:
    //   triangle(i1,l25,x)
    //   triangle(i1,j14,j15)
    // and k5 satisfies:
    //   triangle(k5,l25,j15)
    //   triangle(k5,j14,x)
    // ---------------------------------------------------------------------

    {
        const bool constant_ok =
        triangle_condition(Di1, s.Dj14, s.Dj15);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(X * L25 >= 32768)
#endif
        for (std::ptrdiff_t ix_s = 0;
             ix_s < static_cast<std::ptrdiff_t>(X);
             ++ix_s)
        {
            for (std::ptrdiff_t il25_s = 0;
                 il25_s < static_cast<std::ptrdiff_t>(L25);
                 ++il25_s)
            {
                const std::size_t ix =
                    static_cast<std::size_t>(ix_s);

                const std::size_t il25 =
                    static_cast<std::size_t>(il25_s);

                const std::size_t block =
                    idx2(ix, il25, L25);

                const int Dx = xvals[ix];
                const int Dl25 = l25[il25];

                if (!constant_ok ||
                    !triangle_condition(Di1, Dl25, Dx))
                {
                    plan.sjA.blocks[block] = empty_range_1d();
                    continue;
                }

                plan.sjA.blocks[block] =
                    axis_range_for_two_triangles(
                        k5,
                        Dl25, s.Dj15,
                        s.Dj14, Dx
                    );
            }
        }
    }

    // ---------------------------------------------------------------------
    // sjB(x,k4,l35,l45) -> k5 range
    //
    // outer conditions:
    //   triangle(k4,j14,l45)
    //   triangle(k4,l35,x)
    // inner k5:
    //   triangle(k5,l35,l45)
    //   triangle(k5,j14,x)
    // ---------------------------------------------------------------------

    {
#ifdef _OPENMP
#pragma omp parallel for collapse(4) schedule(static)
#endif
        for (std::ptrdiff_t ix_s = 0;
             ix_s < static_cast<std::ptrdiff_t>(X);
             ++ix_s)
        {
            for (std::ptrdiff_t ik4_s = 0;
                 ik4_s < static_cast<std::ptrdiff_t>(K4d);
                 ++ik4_s)
            {
                for (std::ptrdiff_t il35_s = 0;
                     il35_s < static_cast<std::ptrdiff_t>(L35);
                     ++il35_s)
                {
                    for (std::ptrdiff_t il45_s = 0;
                         il45_s < static_cast<std::ptrdiff_t>(L45);
                         ++il45_s)
                    {
                        const std::size_t ix =
                        static_cast<std::size_t>(ix_s);

                        const std::size_t ik4 =
                        static_cast<std::size_t>(ik4_s);

                        const std::size_t il35 =
                        static_cast<std::size_t>(il35_s);

                        const std::size_t il45 =
                        static_cast<std::size_t>(il45_s);

                        const std::size_t block =
                        idx4(ix, ik4, il35, il45,
                             K4d, L35, L45);

                        const int Dx   = xvals[ix];
                        const int Dk4  = k4[ik4];
                        const int Dl35 = l35[il35];
                        const int Dl45 = l45[il45];

                        if (!triangle_condition(Dk4, s.Dj14, Dl45) ||
                            !triangle_condition(Dk4, Dl35, Dx))
                        {
                            plan.sjB.blocks[block] = empty_range_1d();
                            continue;
                        }

                        plan.sjB.blocks[block] =
                        axis_range_for_two_triangles(
                                                     k5,
                                                     Dl35, Dl45,
                                                     s.Dj14, Dx
                                                     );
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // sjC(x,l24,l34,l35) -> k3 x k4 rectangle
    //
    // k3:
    //   triangle(k3,l35,l34)
    //   triangle(k3,l24,x)
    //
    // k4:
    //   triangle(k4,l24,l34)
    //   triangle(k4,l35,x)
    // ---------------------------------------------------------------------

    {
#ifdef _OPENMP
#pragma omp parallel for collapse(4) schedule(static)
#endif
        for (std::ptrdiff_t ix_s = 0;
             ix_s < static_cast<std::ptrdiff_t>(X);
             ++ix_s)
        {
            for (std::ptrdiff_t il24_s = 0;
                 il24_s < static_cast<std::ptrdiff_t>(L24);
                 ++il24_s)
            {
                for (std::ptrdiff_t il34_s = 0;
                     il34_s < static_cast<std::ptrdiff_t>(L34);
                     ++il34_s)
                {
                    for (std::ptrdiff_t il35_s = 0;
                         il35_s < static_cast<std::ptrdiff_t>(L35);
                         ++il35_s)
                    {
                        const std::size_t ix =
                        static_cast<std::size_t>(ix_s);

                        const std::size_t il24 =
                        static_cast<std::size_t>(il24_s);

                        const std::size_t il34 =
                        static_cast<std::size_t>(il34_s);

                        const std::size_t il35 =
                        static_cast<std::size_t>(il35_s);

                        const std::size_t block =
                        idx4(ix, il24, il34, il35,
                             L24, L34, L35);

                        const int Dx   = xvals[ix];
                        const int Dl24 = l24[il24];
                        const int Dl34 = l34[il34];
                        const int Dl35 = l35[il35];

                        const Range1D rk3 =
                        axis_range_for_two_triangles(
                                                     k3,
                                                     Dl35, Dl34,
                                                     Dl24, Dx
                                                     );

                        const Range1D rk4 =
                        axis_range_for_two_triangles(
                                                     k4,
                                                     Dl24, Dl34,
                                                     Dl35, Dx
                                                     );

                        if (rk3.empty() || rk4.empty()) {
                            plan.sjC.blocks[block] = empty_range_2d();
                            continue;
                        }

                        plan.sjC.blocks[block] =
                        Range2D{
                            rk3.begin,
                            rk3.end,
                            rk4.begin,
                            rk4.end,
                            0ull
                        };
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // sjD(x,k3,l23,l24) -> k2 range
    //
    // outer conditions:
    //   triangle(k3,j13,l23)
    //   triangle(k3,l24,x)
    //
    // inner k2:
    //   triangle(k2,l24,l23)
    //   triangle(k2,j13,x)
    // ---------------------------------------------------------------------

    {
#ifdef _OPENMP
#pragma omp parallel for collapse(4) schedule(static)
#endif
        for (std::ptrdiff_t ix_s = 0;
             ix_s < static_cast<std::ptrdiff_t>(X);
             ++ix_s)
        {
            for (std::ptrdiff_t ik3_s = 0;
                 ik3_s < static_cast<std::ptrdiff_t>(K3d);
                 ++ik3_s)
            {
                for (std::ptrdiff_t il23_s = 0;
                     il23_s < static_cast<std::ptrdiff_t>(L23);
                     ++il23_s)
                {
                    for (std::ptrdiff_t il24_s = 0;
                         il24_s < static_cast<std::ptrdiff_t>(L24);
                         ++il24_s)
                    {
                        const std::size_t ix =
                        static_cast<std::size_t>(ix_s);

                        const std::size_t ik3 =
                        static_cast<std::size_t>(ik3_s);

                        const std::size_t il23 =
                        static_cast<std::size_t>(il23_s);

                        const std::size_t il24 =
                        static_cast<std::size_t>(il24_s);

                        const std::size_t block =
                        idx4(ix, ik3, il23, il24,
                             K3d, L23, L24);

                        const int Dx   = xvals[ix];
                        const int Dk3  = k3[ik3];
                        const int Dl23 = l23[il23];
                        const int Dl24 = l24[il24];

                        if (!triangle_condition(Dk3, s.Dj13, Dl23) ||
                            !triangle_condition(Dk3, Dl24, Dx))
                        {
                            plan.sjD.blocks[block] = empty_range_1d();
                            continue;
                        }

                        plan.sjD.blocks[block] =
                        axis_range_for_two_triangles(
                                                     k2,
                                                     Dl24, Dl23,
                                                     s.Dj13, Dx
                                                     );
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // sjE(x,l25) -> k2 range
    //
    // nonzero if:
    //   triangle(i1,l25,x)
    //   triangle(i1,j13,j12)
    //
    // inner k2:
    //   triangle(k2,l25,j12)
    //   triangle(k2,j13,x)
    // ---------------------------------------------------------------------

    {
        const bool constant_ok =
        triangle_condition(Di1, s.Dj13, s.Dj12);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(X * L25 >= 32768)
#endif
        for (std::ptrdiff_t ix_s = 0;
             ix_s < static_cast<std::ptrdiff_t>(X);
             ++ix_s)
        {
            for (std::ptrdiff_t il25_s = 0;
                 il25_s < static_cast<std::ptrdiff_t>(L25);
                 ++il25_s)
            {
                const std::size_t ix =
                    static_cast<std::size_t>(ix_s);

                const std::size_t il25 =
                    static_cast<std::size_t>(il25_s);

                const std::size_t block =
                    idx2(ix, il25, L25);

                const int Dx = xvals[ix];
                const int Dl25 = l25[il25];

                if (!constant_ok ||
                    !triangle_condition(Di1, Dl25, Dx))
                {
                    plan.sjE.blocks[block] = empty_range_1d();
                    continue;
                }

                plan.sjE.blocks[block] =
                    axis_range_for_two_triangles(
                        k2,
                        Dl25, s.Dj12,
                        s.Dj13, Dx
                    );
            }
        }
    }

    // ---------------------------------------------------------------------
    // B2(l23,l24,l25) -> k2 range
    //
    // inner k2:
    //   triangle(k2,l23,l24)
    //   triangle(k2,l25,j12)
    //
    // We intentionally ignore the booster symmetry/parity zero rule here.
    // ---------------------------------------------------------------------

    {
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
        for (std::ptrdiff_t il23_s = 0;
             il23_s < static_cast<std::ptrdiff_t>(L23);
             ++il23_s)
        {
            for (std::ptrdiff_t il24_s = 0;
                 il24_s < static_cast<std::ptrdiff_t>(L24);
                 ++il24_s)
            {
                for (std::ptrdiff_t il25_s = 0;
                     il25_s < static_cast<std::ptrdiff_t>(L25);
                     ++il25_s)
                {
                    const std::size_t il23 =
                    static_cast<std::size_t>(il23_s);

                    const std::size_t il24 =
                    static_cast<std::size_t>(il24_s);

                    const std::size_t il25 =
                    static_cast<std::size_t>(il25_s);

                    const std::size_t block =
                    idx3(il23, il24, il25,
                         L24, L25);

                    plan.B2.blocks[block] =
                    axis_range_for_two_triangles(
                                                 k2,
                                                 l23[il23], l24[il24],
                                                 l25[il25], s.Dj12
                                                 );
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // B3(k3,l34,l35) -> l23 range
    //
    // outer:
    //   triangle(k3,l34,l35)
    //
    // inner l23:
    //   triangle(k3,j13,l23)
    //
    // We intentionally ignore the booster symmetry/parity zero rule here.
    // ---------------------------------------------------------------------

    {
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
        for (std::ptrdiff_t ik3_s = 0;
             ik3_s < static_cast<std::ptrdiff_t>(K3d);
             ++ik3_s)
        {
            for (std::ptrdiff_t il34_s = 0;
                 il34_s < static_cast<std::ptrdiff_t>(L34);
                 ++il34_s)
            {
                for (std::ptrdiff_t il35_s = 0;
                     il35_s < static_cast<std::ptrdiff_t>(L35);
                     ++il35_s)
                {
                    const std::size_t ik3 =
                    static_cast<std::size_t>(ik3_s);

                    const std::size_t il34 =
                    static_cast<std::size_t>(il34_s);

                    const std::size_t il35 =
                    static_cast<std::size_t>(il35_s);

                    const std::size_t block =
                    idx3(ik3, il34, il35,
                         L34, L35);

                    const int Dk3  = k3[ik3];
                    const int Dl34 = l34[il34];
                    const int Dl35 = l35[il35];

                    if (!triangle_condition(Dk3, Dl34, Dl35)) {
                        plan.B3.blocks[block] = empty_range_1d();
                        continue;
                    }

                    plan.B3.blocks[block] =
                    axis_range_for_triangle(
                                            l23,
                                            Dk3,
                                            s.Dj13
                                            );
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // B4(k4,l24,l34) -> l45 range
    //
    // outer:
    //   triangle(k4,l24,l34)
    //
    // inner l45:
    //   triangle(k4,j14,l45)
    //
    // We intentionally ignore the booster symmetry/parity zero rule here.
    // ---------------------------------------------------------------------

    {
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
        for (std::ptrdiff_t ik4_s = 0;
             ik4_s < static_cast<std::ptrdiff_t>(K4d);
             ++ik4_s)
        {
            for (std::ptrdiff_t il24_s = 0;
                 il24_s < static_cast<std::ptrdiff_t>(L24);
                 ++il24_s)
            {
                for (std::ptrdiff_t il34_s = 0;
                     il34_s < static_cast<std::ptrdiff_t>(L34);
                     ++il34_s)
                {
                    const std::size_t ik4 =
                    static_cast<std::size_t>(ik4_s);

                    const std::size_t il24 =
                    static_cast<std::size_t>(il24_s);

                    const std::size_t il34 =
                    static_cast<std::size_t>(il34_s);

                    const std::size_t block =
                    idx3(ik4, il24, il34,
                         L24, L34);

                    const int Dk4  = k4[ik4];
                    const int Dl24 = l24[il24];
                    const int Dl34 = l34[il34];

                    if (!triangle_condition(Dk4, Dl24, Dl34)) {
                        plan.B4.blocks[block] = empty_range_1d();
                        continue;
                    }

                    plan.B4.blocks[block] =
                    axis_range_for_triangle(
                                            l45,
                                            Dk4,
                                            s.Dj14
                                            );
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // B5(l25,l35,l45) -> k5 range
    //
    // inner k5:
    //   triangle(k5,j15,l25)
    //   triangle(k5,l35,l45)
    //
    // We intentionally ignore the booster symmetry/parity zero rule here.
    // ---------------------------------------------------------------------

    {
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
        for (std::ptrdiff_t il25_s = 0;
             il25_s < static_cast<std::ptrdiff_t>(L25);
             ++il25_s)
        {
            for (std::ptrdiff_t il35_s = 0;
                 il35_s < static_cast<std::ptrdiff_t>(L35);
                 ++il35_s)
            {
                for (std::ptrdiff_t il45_s = 0;
                     il45_s < static_cast<std::ptrdiff_t>(L45);
                     ++il45_s)
                {
                    const std::size_t il25 =
                    static_cast<std::size_t>(il25_s);

                    const std::size_t il35 =
                    static_cast<std::size_t>(il35_s);

                    const std::size_t il45 =
                    static_cast<std::size_t>(il45_s);

                    const std::size_t block =
                    idx3(il25, il35, il45,
                         L35, L45);

                    plan.B5.blocks[block] =
                    axis_range_for_two_triangles(
                                                 k5,
                                                 s.Dj15, l25[il25],
                                                 l35[il35], l45[il45]
                                                 );
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Compute packed offsets.
    // ---------------------------------------------------------------------

    finalize_offsets(plan.sjA);
    finalize_offsets(plan.sjB);
    finalize_offsets(plan.sjC);
    finalize_offsets(plan.sjD);
    finalize_offsets(plan.sjE);

    finalize_offsets(plan.B2);
    finalize_offsets(plan.B3);
    finalize_offsets(plan.B4);
    finalize_offsets(plan.B5);

    return plan;
}

} // namespace b4support
