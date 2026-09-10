// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "full_tensor/FullTensorInputSupport_HalfSjC.hpp"

namespace b4support {

int full_tensor_half_sjC_max_value_in_vector(const std::vector<int>& v)
{
    if (v.empty()) {
        return 0;
    }
    return *std::max_element(v.begin(), v.end());
}

std::vector<int> full_tensor_half_sjC_make_value_to_index_map(
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

std::uint64_t FullTensorInputSupportPlan::total_payload_bytes() const noexcept
{
    return sjA.payload_bytes()
         + sjB.payload_bytes()
         + sjC.payload_bytes()
         + sjD.payload_bytes()
         + sjE.payload_bytes()
         + B2.payload_bytes()
         + B3.payload_bytes()
         + B4.payload_bytes()
         + B5.payload_bytes();
}

std::uint64_t FullTensorInputSupportPlan::total_metadata_bytes() const noexcept
{
    return sjA.metadata_bytes()
         + sjB.metadata_bytes()
         + sjC.metadata_bytes()
         + static_cast<std::uint64_t>(sjC_map.size() * sizeof(FullTensorSjCHalfBlockMap))
         + static_cast<std::uint64_t>((sjC_k3_to_k4.size() + sjC_k4_to_k3.size()) * sizeof(int))
         + sjD.metadata_bytes()
         + sjE.metadata_bytes()
         + B2.metadata_bytes()
         + B3.metadata_bytes()
         + B4.metadata_bytes()
         + B5.metadata_bytes();
}

void FullTensorInputSupportPlan::print(std::ostream& os) const
{
    const double mib = 1024.0 * 1024.0;

    os << "\nFull-tensor input support plan\n";
    os << "==============================\n\n";

    sjA.print(os); os << "\n";
    sjB.print(os); os << "\n";
    sjC.print(os);
    os << "  note          = half-stored; use sjC_map for logical blocks\n";
    os << "\n";
    sjD.print(os); os << "\n";
    sjE.print(os); os << "\n";

    B2.print(os); os << "\n";
    B3.print(os); os << "\n";
    B4.print(os); os << "\n";
    B5.print(os); os << "\n";

    os << "Total packed payload = "
       << static_cast<double>(total_payload_bytes()) / mib
       << " MiB\n";

    os << "Total metadata       = "
       << static_cast<double>(total_metadata_bytes()) / mib
       << " MiB\n";
}

FullTensorInputSupportPlan build_full_tensor_input_support_plan(
    const B4Spins& s,
    int Dl,
    const FullTensorChannels& ch)
{
    FullTensorInputSupportPlan plan;
    plan.channels = ch;
    plan.reflection_15432_preserves_spins =
        s.Dj12 == s.Dj15 &&
        s.Dj13 == s.Dj14 &&
        s.Dj23 == s.Dj45 &&
        s.Dj24 == s.Dj35;

    plan.l23 = make_l_axis(s.Dj23, Dl);
    plan.l24 = make_l_axis(s.Dj24, Dl);
    plan.l25 = make_l_axis(s.Dj25, Dl);
    plan.l34 = make_l_axis(s.Dj34, Dl);
    plan.l35 = make_l_axis(s.Dj35, Dl);
    plan.l45 = make_l_axis(s.Dj45, Dl);

    const std::vector<int>& i1 = ch.i1;
    const std::vector<int>& i2 = ch.i2;
    const std::vector<int>& i3 = ch.i3;
    const std::vector<int>& i4 = ch.i4;
    const std::vector<int>& i5 = ch.i5;

    const std::vector<int>& xvals = ch.xvals;

    const std::vector<int>& k2 = ch.k2;
    const std::vector<int>& k3 = ch.k3;
    const std::vector<int>& k4 = ch.k4;
    const std::vector<int>& k5 = ch.k5;

    const std::vector<int>& l23 = plan.l23;
    const std::vector<int>& l24 = plan.l24;
    const std::vector<int>& l25 = plan.l25;
    const std::vector<int>& l34 = plan.l34;
    const std::vector<int>& l35 = plan.l35;
    const std::vector<int>& l45 = plan.l45;

    const std::size_t I1 = i1.size();
    const std::size_t I2 = i2.size();
    const std::size_t I3 = i3.size();
    const std::size_t I4 = i4.size();
    const std::size_t I5 = i5.size();

    const std::size_t X = xvals.size();

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

    plan.sjC_map.assign(X * L24 * L34 * L35, FullTensorSjCHalfBlockMap{});
    plan.sjC_k3_to_k4.assign(K3d, -1);
    plan.sjC_k4_to_k3.assign(K4d, -1);

    int max_sjC_map_value = 0;
    max_sjC_map_value = std::max(max_sjC_map_value, full_tensor_half_sjC_max_value_in_vector(k3));
    max_sjC_map_value = std::max(max_sjC_map_value, full_tensor_half_sjC_max_value_in_vector(k4));
    max_sjC_map_value = std::max(max_sjC_map_value, full_tensor_half_sjC_max_value_in_vector(l24));
    max_sjC_map_value = std::max(max_sjC_map_value, full_tensor_half_sjC_max_value_in_vector(l35));

    const std::vector<int> sjC_val_to_ik3  = full_tensor_half_sjC_make_value_to_index_map(k3,  max_sjC_map_value);
    const std::vector<int> sjC_val_to_ik4  = full_tensor_half_sjC_make_value_to_index_map(k4,  max_sjC_map_value);
    const std::vector<int> sjC_val_to_il24 = full_tensor_half_sjC_make_value_to_index_map(l24, max_sjC_map_value);
    const std::vector<int> sjC_val_to_il35 = full_tensor_half_sjC_make_value_to_index_map(l35, max_sjC_map_value);

    for (std::size_t ik3 = 0; ik3 < K3d; ++ik3) {
        const int v = k3[ik3];
        if (v >= 0 && v <= max_sjC_map_value) {
            plan.sjC_k3_to_k4[ik3] = sjC_val_to_ik4[static_cast<std::size_t>(v)];
        }
    }
    for (std::size_t ik4 = 0; ik4 < K4d; ++ik4) {
        const int v = k4[ik4];
        if (v >= 0 && v <= max_sjC_map_value) {
            plan.sjC_k4_to_k3[ik4] = sjC_val_to_ik3[static_cast<std::size_t>(v)];
        }
    }

    // 6j supports.
    plan.sjA = make_support_1d(
        "sjA support: block (i1,x,l25) -> k5",
        {I1, X, L25},
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
        "sjE support: block (i1,x,l25) -> k2",
        {I1, X, L25},
        K2d
    );

    // Booster supports.
    plan.B2 = make_support_1d(
        "B2 support: block (i2,l23,l24,l25) -> k2",
        {I2, L23, L24, L25},
        K2d
    );

    plan.B3 = make_support_1d(
        "B3 support: block (i3,k3,l34,l35) -> l23",
        {I3, K3d, L34, L35},
        L23
    );

    plan.B4 = make_support_1d(
        "B4 support: block (i4,k4,l24,l34) -> l45",
        {I4, K4d, L24, L34},
        L45
    );

    plan.B5 = make_support_1d(
        "B5 support: block (i5,l25,l35,l45) -> k5",
        {I5, L25, L35, L45},
        K5d
    );

    auto finalize_all_and_return = [&]() -> FullTensorInputSupportPlan {
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
    };

    if (I1 == 0 || I2 == 0 || I3 == 0 || I4 == 0 || I5 == 0 ||
        X == 0 || K2d == 0 || K3d == 0 || K4d == 0 || K5d == 0)
    {
        return finalize_all_and_return();
    }

    // ---------------------------------------------------------------------
    // sjA(i1,x,l25) -> k5 range
    // ---------------------------------------------------------------------
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (std::ptrdiff_t ii1_s = 0; ii1_s < static_cast<std::ptrdiff_t>(I1); ++ii1_s) {
        for (std::ptrdiff_t ix_s = 0; ix_s < static_cast<std::ptrdiff_t>(X); ++ix_s) {
            for (std::ptrdiff_t il25_s = 0; il25_s < static_cast<std::ptrdiff_t>(L25); ++il25_s) {
                const std::size_t ii1  = static_cast<std::size_t>(ii1_s);
                const std::size_t ix   = static_cast<std::size_t>(ix_s);
                const std::size_t il25 = static_cast<std::size_t>(il25_s);

                const std::size_t block = idx3(ii1, ix, il25, X, L25);

                const int Di1  = i1[ii1];
                const int Dx   = xvals[ix];
                const int Dl25 = l25[il25];

                if (!triangle_condition(Di1, s.Dj14, s.Dj15) ||
                    !triangle_condition(Di1, Dl25, Dx))
                {
                    plan.sjA.blocks[block] = empty_range_1d();
                    continue;
                }

                plan.sjA.blocks[block] = axis_range_for_two_triangles(
                    k5,
                    Dl25, s.Dj15,
                    s.Dj14, Dx
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // sjB(x,k4,l35,l45) -> k5 range
    // ---------------------------------------------------------------------
#ifdef _OPENMP
#pragma omp parallel for collapse(4) schedule(static)
#endif
    for (std::ptrdiff_t ix_s = 0; ix_s < static_cast<std::ptrdiff_t>(X); ++ix_s) {
        for (std::ptrdiff_t ik4_s = 0; ik4_s < static_cast<std::ptrdiff_t>(K4d); ++ik4_s) {
            for (std::ptrdiff_t il35_s = 0; il35_s < static_cast<std::ptrdiff_t>(L35); ++il35_s) {
                for (std::ptrdiff_t il45_s = 0; il45_s < static_cast<std::ptrdiff_t>(L45); ++il45_s) {
                    const std::size_t ix   = static_cast<std::size_t>(ix_s);
                    const std::size_t ik4  = static_cast<std::size_t>(ik4_s);
                    const std::size_t il35 = static_cast<std::size_t>(il35_s);
                    const std::size_t il45 = static_cast<std::size_t>(il45_s);

                    const std::size_t block = idx4(ix, ik4, il35, il45, K4d, L35, L45);

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

                    plan.sjB.blocks[block] = axis_range_for_two_triangles(
                        k5,
                        Dl35, Dl45,
                        s.Dj14, Dx
                    );
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // sjC(x,l24,l34,l35) -> k3 x k4 rectangle, half-stored.
    // ---------------------------------------------------------------------
    // Logical identity:
    //   C(x,l24,l34,l35; k3,k4) = C(x,l35,l34,l24; k4,k3)
    // where the swapped values must actually exist in the corresponding axes.
    //
    // We store only the canonical l-pair block when possible.  Diagonal
    // l24==l35 blocks are still stored as full rectangles because Range2D does
    // not represent triangular virtual intertwiner storage.  Blocks whose swapped axes do not
    // exist, or whose virtual intertwiner ranges cannot be mapped value-wise, are stored
    // normally.  This keeps the scheme exact for generic spin data.
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

                    const int Dx   = xvals[ix];
                    const int Dl24 = l24[il24];
                    const int Dl34 = l34[il34];
                    const int Dl35 = l35[il35];

                    const Range1D rk3 = axis_range_for_two_triangles(
                        k3,
                        Dl35, Dl34,
                        Dl24, Dx
                    );

                    const Range1D rk4 = axis_range_for_two_triangles(
                        k4,
                        Dl24, Dl34,
                        Dl35, Dx
                    );

                    FullTensorSjCHalfBlockMap map;
                    map.logical = empty_range_2d();
                    map.storage_block = static_cast<std::uint32_t>(block);
                    map.transpose_from_storage = false;

                    if (rk3.empty() || rk4.empty()) {
                        plan.sjC.blocks[block] = empty_range_2d();
                        plan.sjC_map[block] = map;
                        continue;
                    }

                    const Range2D logical{
                        rk3.begin,
                        rk3.end,
                        rk4.begin,
                        rk4.end,
                        0ull
                    };

                    map.logical = logical;

                    const int sym_il24 = (Dl35 >= 0 && Dl35 <= max_sjC_map_value)
                        ? sjC_val_to_il24[static_cast<std::size_t>(Dl35)]
                        : -1;
                    const int sym_il35 = (Dl24 >= 0 && Dl24 <= max_sjC_map_value)
                        ? sjC_val_to_il35[static_cast<std::size_t>(Dl24)]
                        : -1;

                    bool virtual_ranges_map = true;
                    for (std::uint32_t ik3 = rk3.begin; ik3 < rk3.end; ++ik3) {
                        if (plan.sjC_k3_to_k4[ik3] < 0) {
                            virtual_ranges_map = false;
                            break;
                        }
                    }
                    if (virtual_ranges_map) {
                        for (std::uint32_t ik4 = rk4.begin; ik4 < rk4.end; ++ik4) {
                            if (plan.sjC_k4_to_k3[ik4] < 0) {
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

    // ---------------------------------------------------------------------
    // sjD(x,k3,l23,l24) -> k2 range
    // ---------------------------------------------------------------------
#ifdef _OPENMP
#pragma omp parallel for collapse(4) schedule(static)
#endif
    for (std::ptrdiff_t ix_s = 0; ix_s < static_cast<std::ptrdiff_t>(X); ++ix_s) {
        for (std::ptrdiff_t ik3_s = 0; ik3_s < static_cast<std::ptrdiff_t>(K3d); ++ik3_s) {
            for (std::ptrdiff_t il23_s = 0; il23_s < static_cast<std::ptrdiff_t>(L23); ++il23_s) {
                for (std::ptrdiff_t il24_s = 0; il24_s < static_cast<std::ptrdiff_t>(L24); ++il24_s) {
                    const std::size_t ix   = static_cast<std::size_t>(ix_s);
                    const std::size_t ik3  = static_cast<std::size_t>(ik3_s);
                    const std::size_t il23 = static_cast<std::size_t>(il23_s);
                    const std::size_t il24 = static_cast<std::size_t>(il24_s);

                    const std::size_t block = idx4(ix, ik3, il23, il24, K3d, L23, L24);

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

                    plan.sjD.blocks[block] = axis_range_for_two_triangles(
                        k2,
                        Dl24, Dl23,
                        s.Dj13, Dx
                    );
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // sjE(i1,x,l25) -> k2 range
    // ---------------------------------------------------------------------
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (std::ptrdiff_t ii1_s = 0; ii1_s < static_cast<std::ptrdiff_t>(I1); ++ii1_s) {
        for (std::ptrdiff_t ix_s = 0; ix_s < static_cast<std::ptrdiff_t>(X); ++ix_s) {
            for (std::ptrdiff_t il25_s = 0; il25_s < static_cast<std::ptrdiff_t>(L25); ++il25_s) {
                const std::size_t ii1  = static_cast<std::size_t>(ii1_s);
                const std::size_t ix   = static_cast<std::size_t>(ix_s);
                const std::size_t il25 = static_cast<std::size_t>(il25_s);

                const std::size_t block = idx3(ii1, ix, il25, X, L25);

                const int Di1  = i1[ii1];
                const int Dx   = xvals[ix];
                const int Dl25 = l25[il25];

                if (!triangle_condition(Di1, s.Dj13, s.Dj12) ||
                    !triangle_condition(Di1, Dl25, Dx))
                {
                    plan.sjE.blocks[block] = empty_range_1d();
                    continue;
                }

                plan.sjE.blocks[block] = axis_range_for_two_triangles(
                    k2,
                    Dl25, s.Dj12,
                    s.Dj13, Dx
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // B2(i2,l23,l24,l25) -> k2 range
    // ---------------------------------------------------------------------
#ifdef _OPENMP
#pragma omp parallel for collapse(4) schedule(static)
#endif
    for (std::ptrdiff_t ii2_s = 0; ii2_s < static_cast<std::ptrdiff_t>(I2); ++ii2_s) {
        for (std::ptrdiff_t il23_s = 0; il23_s < static_cast<std::ptrdiff_t>(L23); ++il23_s) {
            for (std::ptrdiff_t il24_s = 0; il24_s < static_cast<std::ptrdiff_t>(L24); ++il24_s) {
                for (std::ptrdiff_t il25_s = 0; il25_s < static_cast<std::ptrdiff_t>(L25); ++il25_s) {
                    const std::size_t ii2  = static_cast<std::size_t>(ii2_s);
                    const std::size_t il23 = static_cast<std::size_t>(il23_s);
                    const std::size_t il24 = static_cast<std::size_t>(il24_s);
                    const std::size_t il25 = static_cast<std::size_t>(il25_s);

                    const std::size_t block = idx4(ii2, il23, il24, il25, L23, L24, L25);

                    const int Di2 = i2[ii2];
                    if (!triangle_condition(Di2, s.Dj23, s.Dj24) ||
                        !triangle_condition(Di2, s.Dj25, s.Dj12))
                    {
                        plan.B2.blocks[block] = empty_range_1d();
                        continue;
                    }

                    plan.B2.blocks[block] = axis_range_for_two_triangles(
                        k2,
                        l23[il23], l24[il24],
                        l25[il25], s.Dj12
                    );
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // B3(i3,k3,l34,l35) -> l23 range
    // ---------------------------------------------------------------------
#ifdef _OPENMP
#pragma omp parallel for collapse(4) schedule(static)
#endif
    for (std::ptrdiff_t ii3_s = 0; ii3_s < static_cast<std::ptrdiff_t>(I3); ++ii3_s) {
        for (std::ptrdiff_t ik3_s = 0; ik3_s < static_cast<std::ptrdiff_t>(K3d); ++ik3_s) {
            for (std::ptrdiff_t il34_s = 0; il34_s < static_cast<std::ptrdiff_t>(L34); ++il34_s) {
                for (std::ptrdiff_t il35_s = 0; il35_s < static_cast<std::ptrdiff_t>(L35); ++il35_s) {
                    const std::size_t ii3  = static_cast<std::size_t>(ii3_s);
                    const std::size_t ik3  = static_cast<std::size_t>(ik3_s);
                    const std::size_t il34 = static_cast<std::size_t>(il34_s);
                    const std::size_t il35 = static_cast<std::size_t>(il35_s);

                    const std::size_t block = idx4(ii3, ik3, il34, il35, K3d, L34, L35);

                    const int Di3  = i3[ii3];
                    const int Dk3  = k3[ik3];
                    const int Dl34 = l34[il34];
                    const int Dl35 = l35[il35];

                    if (!triangle_condition(Di3, s.Dj34, s.Dj35) ||
                        !triangle_condition(Di3, s.Dj13, s.Dj23) ||
                        !triangle_condition(Dk3, Dl34, Dl35))
                    {
                        plan.B3.blocks[block] = empty_range_1d();
                        continue;
                    }

                    plan.B3.blocks[block] = axis_range_for_triangle(
                        l23,
                        Dk3,
                        s.Dj13
                    );
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // B4(i4,k4,l24,l34) -> l45 range
    // ---------------------------------------------------------------------
#ifdef _OPENMP
#pragma omp parallel for collapse(4) schedule(static)
#endif
    for (std::ptrdiff_t ii4_s = 0; ii4_s < static_cast<std::ptrdiff_t>(I4); ++ii4_s) {
        for (std::ptrdiff_t ik4_s = 0; ik4_s < static_cast<std::ptrdiff_t>(K4d); ++ik4_s) {
            for (std::ptrdiff_t il24_s = 0; il24_s < static_cast<std::ptrdiff_t>(L24); ++il24_s) {
                for (std::ptrdiff_t il34_s = 0; il34_s < static_cast<std::ptrdiff_t>(L34); ++il34_s) {
                    const std::size_t ii4  = static_cast<std::size_t>(ii4_s);
                    const std::size_t ik4  = static_cast<std::size_t>(ik4_s);
                    const std::size_t il24 = static_cast<std::size_t>(il24_s);
                    const std::size_t il34 = static_cast<std::size_t>(il34_s);

                    const std::size_t block = idx4(ii4, ik4, il24, il34, K4d, L24, L34);

                    const int Di4  = i4[ii4];
                    const int Dk4  = k4[ik4];
                    const int Dl24 = l24[il24];
                    const int Dl34 = l34[il34];

                    if (!triangle_condition(Di4, s.Dj45, s.Dj14) ||
                        !triangle_condition(Di4, s.Dj24, s.Dj34) ||
                        !triangle_condition(Dk4, Dl24, Dl34))
                    {
                        plan.B4.blocks[block] = empty_range_1d();
                        continue;
                    }

                    plan.B4.blocks[block] = axis_range_for_triangle(
                        l45,
                        Dk4,
                        s.Dj14
                    );
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // B5(i5,l25,l35,l45) -> k5 range
    // ---------------------------------------------------------------------
#ifdef _OPENMP
#pragma omp parallel for collapse(4) schedule(static)
#endif
    for (std::ptrdiff_t ii5_s = 0; ii5_s < static_cast<std::ptrdiff_t>(I5); ++ii5_s) {
        for (std::ptrdiff_t il25_s = 0; il25_s < static_cast<std::ptrdiff_t>(L25); ++il25_s) {
            for (std::ptrdiff_t il35_s = 0; il35_s < static_cast<std::ptrdiff_t>(L35); ++il35_s) {
                for (std::ptrdiff_t il45_s = 0; il45_s < static_cast<std::ptrdiff_t>(L45); ++il45_s) {
                    const std::size_t ii5  = static_cast<std::size_t>(ii5_s);
                    const std::size_t il25 = static_cast<std::size_t>(il25_s);
                    const std::size_t il35 = static_cast<std::size_t>(il35_s);
                    const std::size_t il45 = static_cast<std::size_t>(il45_s);

                    const std::size_t block = idx4(ii5, il25, il35, il45, L25, L35, L45);

                    const int Di5 = i5[ii5];
                    if (!triangle_condition(Di5, s.Dj15, s.Dj25) ||
                        !triangle_condition(Di5, s.Dj35, s.Dj45))
                    {
                        plan.B5.blocks[block] = empty_range_1d();
                        continue;
                    }

                    plan.B5.blocks[block] = axis_range_for_two_triangles(
                        k5,
                        s.Dj15, l25[il25],
                        l35[il35], l45[il45]
                    );
                }
            }
        }
    }

    return finalize_all_and_return();
}

} // namespace b4support
