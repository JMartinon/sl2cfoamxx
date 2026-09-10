// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "full_tensor/FullTensorPackedBoosters_HalfSjC.hpp"

#include "boosters/B4TensorFilling.hpp"

namespace b4support {


void pack_full_B2_from_canonical(
    const FullTensorInputSupportPlan& support,
    const Tensor<double, 5>& dense_B2,
    std::vector<double>& packed_B2)
{
    const std::size_t I2  = support.channels.i2.size();
    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();

    const double* __restrict src = dense_B2.data();
    double* __restrict dst = packed_B2.data();

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
                    const Range1D& r = support.B2.blocks[block];

                    for (std::size_t ik2 = r.begin; ik2 < r.end; ++ik2) {
                        const std::size_t src_idx = flat_full_B2_canonical(
                            il23, il24, il25, ii2, ik2,
                            L23, L24, L25, I2
                        );
                        const std::size_t dst_idx = r.offset + (ik2 - r.begin);
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}

void pack_full_B3_from_canonical(
    const FullTensorInputSupportPlan& support,
    const Tensor<double, 5>& dense_B3,
    std::vector<double>& packed_B3)
{
    const std::size_t I3  = support.channels.i3.size();
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t L23 = support.l23.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();

    const double* __restrict src = dense_B3.data();
    double* __restrict dst = packed_B3.data();

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
                    const Range1D& r = support.B3.blocks[block];

                    for (std::size_t il23 = r.begin; il23 < r.end; ++il23) {
                        const std::size_t src_idx = flat_full_B3_canonical(
                            il34, il35, il23, ii3, ik3,
                            L34, L35, L23, I3
                        );
                        const std::size_t dst_idx = r.offset + (il23 - r.begin);
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}

void pack_full_B4_from_canonical(
    const FullTensorInputSupportPlan& support,
    const Tensor<double, 5>& dense_B4,
    std::vector<double>& packed_B4)
{
    const std::size_t I4  = support.channels.i4.size();
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L45 = support.l45.size();

    const double* __restrict src = dense_B4.data();
    double* __restrict dst = packed_B4.data();

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
                    const Range1D& r = support.B4.blocks[block];

                    for (std::size_t il45 = r.begin; il45 < r.end; ++il45) {
                        const std::size_t src_idx = flat_full_B4_canonical(
                            il45, il24, il34, ii4, ik4,
                            L45, L24, L34, I4
                        );
                        const std::size_t dst_idx = r.offset + (il45 - r.begin);
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}

void pack_full_B5_from_canonical(
    const FullTensorInputSupportPlan& support,
    const Tensor<double, 5>& dense_B5,
    std::vector<double>& packed_B5)
{
    const std::size_t I5  = support.channels.i5.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    const double* __restrict src = dense_B5.data();
    double* __restrict dst = packed_B5.data();

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
                    const Range1D& r = support.B5.blocks[block];

                    for (std::size_t ik5 = r.begin; ik5 < r.end; ++ik5) {
                        const std::size_t src_idx = flat_full_B5_canonical(
                            il25, il35, il45, ii5, ik5,
                            L25, L35, L45, I5
                        );
                        const std::size_t dst_idx = r.offset + (ik5 - r.begin);
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}

void fill_packed_full_tensor_boosters_via_canonical(
    const B4Spins& spins,
    const std::array<std::size_t, 6>& Dim_ls,
    double gamma,
    const std::vector<double>& FactTable,
    const FullTensorInputSupportPlan& support,
    PackedFullTensorBoosters& packed,
    sl2cfoamxx::SimplicityConstraint prescription,
    std::uint32_t rho_tile)
{
    const FullTensorChannels& ch = support.channels;

    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

#ifndef NDEBUG
    assert(Dim_ls[0] == L23);
    assert(Dim_ls[1] == L24);
    assert(Dim_ls[2] == L25);
    assert(Dim_ls[3] == L34);
    assert(Dim_ls[4] == L35);
    assert(Dim_ls[5] == L45);
#endif

    packed.allocate_from_support(support);

    Tensor<double, 5> dense_B2({L23, L24, L25, ch.i2.size(), ch.k2.size()}, false);
    Tensor<double, 5> dense_B3({L34, L35, L23, ch.i3.size(), ch.k3.size()}, false);
    Tensor<double, 5> dense_B4({L45, L24, L34, ch.i4.size(), ch.k4.size()}, false);
    Tensor<double, 5> dense_B5({L25, L35, L45, ch.i5.size(), ch.k5.size()}, false);

    fill_rank5_tensors(
        spins,
        Dim_ls,
        gamma,
        FactTable,
        ch,
        dense_B2,
        dense_B3,
        dense_B4,
        dense_B5,
        prescription,
        rho_tile
    );

    pack_full_B2_from_canonical(support, dense_B2, packed.B2);
    pack_full_B3_from_canonical(support, dense_B3, packed.B3);
    pack_full_B4_from_canonical(support, dense_B4, packed.B4);
    pack_full_B5_from_canonical(support, dense_B5, packed.B5);
}

void fill_packed_full_tensor_booster_leg_via_canonical(
    const B4Spins& spins,
    double gamma,
    const std::vector<double>& FactTable,
    const FullTensorInputSupportPlan& support,
    FullTensorBoosterLeg leg,
    PackedFullTensorBoosters& packed,
    sl2cfoamxx::SimplicityConstraint prescription,
    std::uint32_t rho_tile)
{
    const FullTensorChannels& ch = support.channels;
    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    packed.clear();

    switch (leg) {
        case FullTensorBoosterLeg::B2: {
            Tensor<double, 5> dense({L23, L24, L25, ch.i2.size(), ch.k2.size()}, false);
            const std::array<std::size_t, 3> local_dims{{L23, L24, L25}};
            harvest_b4<4, 5>(
                spins.Dj23, spins.Dj24, spins.Dj25, spins.Dj12,
                local_dims, ch.i2, ch.k2, gamma, FactTable,
                dense, std::array<std::size_t, 3>{{0, 0, 0}}, false,
                prescription, rho_tile);
            packed.B2.resize(static_cast<std::size_t>(support.B2.packed_values));
            pack_full_B2_from_canonical(support, dense, packed.B2);
            break;
        }
        case FullTensorBoosterLeg::B3: {
            Tensor<double, 5> dense({L34, L35, L23, ch.i3.size(), ch.k3.size()}, false);
            const std::array<std::size_t, 3> local_dims{{L34, L35, L23}};
            harvest_b4<3, 5>(
                spins.Dj34, spins.Dj35, spins.Dj13, spins.Dj23,
                local_dims, ch.i3, ch.k3, gamma, FactTable,
                dense, std::array<std::size_t, 3>{{0, 0, 0}}, false,
                prescription, rho_tile);
            packed.B3.resize(static_cast<std::size_t>(support.B3.packed_values));
            pack_full_B3_from_canonical(support, dense, packed.B3);
            break;
        }
        case FullTensorBoosterLeg::B4: {
            Tensor<double, 5> dense({L45, L24, L34, ch.i4.size(), ch.k4.size()}, false);
            const std::array<std::size_t, 3> local_dims{{L45, L24, L34}};
            harvest_b4<2, 5>(
                spins.Dj45, spins.Dj14, spins.Dj24, spins.Dj34,
                local_dims, ch.i4, ch.k4, gamma, FactTable,
                dense, std::array<std::size_t, 3>{{0, 0, 0}}, false,
                prescription, rho_tile);
            packed.B4.resize(static_cast<std::size_t>(support.B4.packed_values));
            pack_full_B4_from_canonical(support, dense, packed.B4);
            break;
        }
        case FullTensorBoosterLeg::B5: {
            Tensor<double, 5> dense({L25, L35, L45, ch.i5.size(), ch.k5.size()}, false);
            const std::array<std::size_t, 3> local_dims{{L25, L35, L45}};
            harvest_b4<1, 5>(
                spins.Dj15, spins.Dj25, spins.Dj35, spins.Dj45,
                local_dims, ch.i5, ch.k5, gamma, FactTable,
                dense, std::array<std::size_t, 3>{{0, 0, 0}}, false,
                prescription, rho_tile);
            packed.B5.resize(static_cast<std::size_t>(support.B5.packed_values));
            pack_full_B5_from_canonical(support, dense, packed.B5);
            break;
        }
    }

}

} // namespace b4support
