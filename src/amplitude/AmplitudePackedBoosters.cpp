// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/AmplitudePackedBoosters.hpp"

namespace b4support {

void pack_B2_from_canonical(
                                   const AmplitudeInputSupportPlan& support,
                                   const Tensor<double, 4>& dense_B2,
                                   std::vector<double>& packed_B2)
{
    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();

    const double* __restrict src = dense_B2.data();
    double* __restrict dst = packed_B2.data();

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
                const std::size_t il23 = static_cast<std::size_t>(il23_s);
                const std::size_t il24 = static_cast<std::size_t>(il24_s);
                const std::size_t il25 = static_cast<std::size_t>(il25_s);

                const std::size_t block =
                idx3(il23, il24, il25, L24, L25);

                const Range1D& r = support.B2.blocks[block];

                for (std::size_t ik2 = r.begin; ik2 < r.end; ++ik2) {
                    const std::size_t src_idx =
                    flat_B2_canonical(
                                      il23, il24, il25, ik2,
                                      L23, L24, L25
                                      );

                    const std::size_t dst_idx =
                    r.offset + (ik2 - r.begin);

                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

void pack_B3_from_canonical(
                                   const AmplitudeInputSupportPlan& support,
                                   const Tensor<double, 4>& dense_B3,
                                   std::vector<double>& packed_B3)
{
    const std::size_t K3d = support.channels.k3.size();

    const std::size_t L23 = support.l23.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();

    const double* __restrict src = dense_B3.data();
    double* __restrict dst = packed_B3.data();

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
                const std::size_t ik3  = static_cast<std::size_t>(ik3_s);
                const std::size_t il34 = static_cast<std::size_t>(il34_s);
                const std::size_t il35 = static_cast<std::size_t>(il35_s);

                const std::size_t block =
                idx3(ik3, il34, il35, L34, L35);

                const Range1D& r = support.B3.blocks[block];

                for (std::size_t il23 = r.begin; il23 < r.end; ++il23) {
                    const std::size_t src_idx =
                    flat_B3_canonical(
                                      il34, il35, il23, ik3,
                                      L34, L35, L23
                                      );

                    const std::size_t dst_idx =
                    r.offset + (il23 - r.begin);

                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

void pack_B4_from_canonical(
                                   const AmplitudeInputSupportPlan& support,
                                   const Tensor<double, 4>& dense_B4,
                                   std::vector<double>& packed_B4)
{
    const std::size_t K4d = support.channels.k4.size();

    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L45 = support.l45.size();

    const double* __restrict src = dense_B4.data();
    double* __restrict dst = packed_B4.data();

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
                const std::size_t ik4  = static_cast<std::size_t>(ik4_s);
                const std::size_t il24 = static_cast<std::size_t>(il24_s);
                const std::size_t il34 = static_cast<std::size_t>(il34_s);

                const std::size_t block =
                idx3(ik4, il24, il34, L24, L34);

                const Range1D& r = support.B4.blocks[block];

                for (std::size_t il45 = r.begin; il45 < r.end; ++il45) {
                    const std::size_t src_idx =
                    flat_B4_canonical(
                                      il45, il24, il34, ik4,
                                      L45, L24, L34
                                      );

                    const std::size_t dst_idx =
                    r.offset + (il45 - r.begin);

                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

void pack_B5_from_canonical(
                                   const AmplitudeInputSupportPlan& support,
                                   const Tensor<double, 4>& dense_B5,
                                   std::vector<double>& packed_B5)
{
    const std::size_t L25 = support.l25.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    const double* __restrict src = dense_B5.data();
    double* __restrict dst = packed_B5.data();

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
                const std::size_t il25 = static_cast<std::size_t>(il25_s);
                const std::size_t il35 = static_cast<std::size_t>(il35_s);
                const std::size_t il45 = static_cast<std::size_t>(il45_s);

                const std::size_t block =
                idx3(il25, il35, il45, L35, L45);

                const Range1D& r = support.B5.blocks[block];

                for (std::size_t ik5 = r.begin; ik5 < r.end; ++ik5) {
                    const std::size_t src_idx =
                    flat_B5_canonical(
                                      il25, il35, il45, ik5,
                                      L25, L35, L45
                                      );

                    const std::size_t dst_idx =
                    r.offset + (ik5 - r.begin);

                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

void fill_packed_amplitude_boosters_via_canonical(
                                                         const B4Spins& spins,
                                                         const std::array<std::size_t, 6>& Dim_ls,
                                                         double gamma,
                                                         const std::vector<double>& FactTable,
                                                         const AmplitudeInputSupportPlan& support,
                                                         PackedAmplitudeBoosters& packed,
                                                         sl2cfoamxx::SimplicityConstraint prescription,
                                                         std::uint32_t rho_tile)
{
    const AmplitudeChannels& AmC = support.channels;

    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    const std::size_t Dim_k2 = AmC.k2.size();
    const std::size_t Dim_k3 = AmC.k3.size();
    const std::size_t Dim_k4 = AmC.k4.size();
    const std::size_t Dim_k5 = AmC.k5.size();

#ifndef NDEBUG
    assert(Dim_ls[0] == L23);
    assert(Dim_ls[1] == L24);
    assert(Dim_ls[2] == L25);
    assert(Dim_ls[3] == L34);
    assert(Dim_ls[4] == L35);
    assert(Dim_ls[5] == L45);
#endif

    // Allocate packed output first.
    packed.allocate_from_support(support);

    // Canonical dense recurrence buffers.
    //
    // These are small compared with sjC, and retaining the canonical layout
    // preserves the optimized recurrence implementation without an HPTT
    // transpose.
    Tensor<double, 4> dense_B2({L23, L24, L25, Dim_k2}, false);
    Tensor<double, 4> dense_B3({L34, L35, L23, Dim_k3}, false);
    Tensor<double, 4> dense_B4({L45, L24, L34, Dim_k4}, false);
    Tensor<double, 4> dense_B5({L25, L35, L45, Dim_k5}, false);

    fill_rank4_tensors(
                       spins,
                       Dim_ls,
                       gamma,
                       FactTable,
                       AmC,
                       dense_B2,
                       dense_B3,
                       dense_B4,
                       dense_B5,
                       prescription,
                       rho_tile
                       );

    // Pack directly from canonical layout into compact support layout.
    //
    // This replaces:
    //   canonical dense -> HPTT transpose -> dense optimal
    //
    // with:
    //   canonical dense -> compact packed
    pack_B2_from_canonical(support, dense_B2, packed.B2);
    pack_B3_from_canonical(support, dense_B3, packed.B3);
    pack_B4_from_canonical(support, dense_B4, packed.B4);
    pack_B5_from_canonical(support, dense_B5, packed.B5);

    // Canonical tensors are local and will free on return.
}

std::uint64_t b4_choose_coherent_rank5_slab_budget_bytes_impl(
    std::uint64_t future_output_bytes) noexcept
{
    if (const std::uint64_t env = b4_env_u64_or_zero("B4_COHERENT_SLAB_BUDGET_BYTES"); env != 0) {
        return env;
    }

    const std::uint64_t fallback = static_cast<std::uint64_t>(
        B4_COHERENT_RANK5_SLAB_TARGET_BYTES);
    const host_memory::Snapshot snapshot = host_memory::detect_snapshot();
    if (!host_memory::has_any_probe(snapshot)) return fallback;

    // host_memory::automatic_budget_bytes returns a safe process *peak*, not
    // an allocation size.  Convert it to incremental headroom from the
    // current RSS, then reserve allocations which may have been malloc'ed but
    // not faulted in yet.  In particular, do not use total physical memory on
    // a partially occupied node or inside a cgroup/Slurm allocation.
    const double fraction = static_cast<double>(
        B4_COHERENT_RANK5_SLAB_RAM_MARGIN_PERCENT) / 100.0;
    const std::uint64_t process_peak =
        host_memory::automatic_budget_bytes(snapshot, fraction);
    std::uint64_t budget = process_peak > snapshot.process_rss_bytes
        ? process_peak - snapshot.process_rss_bytes
        : 0;

    budget = budget > future_output_bytes ? budget - future_output_bytes : 0;

    // A one-virtual intertwiner slab is always legal: b4_choose_coherent_rank5_b_slab_size
    // clamps any positive target to at least one virtual intertwiner value.  Keeping this
    // below the former 256 MiB floor is essential on small-memory machines.
    budget = std::max<std::uint64_t>(budget, sizeof(double));
    budget = std::min<std::uint64_t>(budget, B4_COHERENT_RANK5_SLAB_MAX_BYTES);
    return budget;
}

void pack_B2_complex_from_contraction_native(
    const AmplitudeInputSupportPlan& support,
    const Tensor<b4utils::Complex64, 4>& dense_B2,
    b4utils::ComplexBuffer<>& packed_B2)
{
    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();

    const b4utils::Complex64* __restrict src = dense_B2.data();
    double* __restrict dst_re = packed_B2.re_data();
    double* __restrict dst_im = packed_B2.im_data();

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
                const Range1D& r = support.B2.blocks[block];

                for (std::size_t ik2 = r.begin; ik2 < r.end; ++ik2) {
                    const std::size_t src_idx =
                        flat_B2_contraction_native(il23, il25, il24, ik2, L23, L25, L24);
                    const std::size_t dst_idx = r.offset + (ik2 - r.begin);
                    const b4utils::Complex64 z = src[src_idx];
                    dst_re[dst_idx] = z.re;
                    dst_im[dst_idx] = z.im;
                }
            }
        }
    }
}

void pack_B3_complex_from_contraction_native(
    const AmplitudeInputSupportPlan& support,
    const Tensor<b4utils::Complex64, 4>& dense_B3,
    b4utils::ComplexBuffer<>& packed_B3)
{
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t L23 = support.l23.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();

    const b4utils::Complex64* __restrict src = dense_B3.data();
    double* __restrict dst_re = packed_B3.re_data();
    double* __restrict dst_im = packed_B3.im_data();

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (std::ptrdiff_t ik3_s = 0; ik3_s < static_cast<std::ptrdiff_t>(K3d); ++ik3_s) {
        for (std::ptrdiff_t il34_s = 0; il34_s < static_cast<std::ptrdiff_t>(L34); ++il34_s) {
            for (std::ptrdiff_t il35_s = 0; il35_s < static_cast<std::ptrdiff_t>(L35); ++il35_s) {
                const std::size_t ik3  = static_cast<std::size_t>(ik3_s);
                const std::size_t il34 = static_cast<std::size_t>(il34_s);
                const std::size_t il35 = static_cast<std::size_t>(il35_s);
                const std::size_t block = idx3(ik3, il34, il35, L34, L35);
                const Range1D& r = support.B3.blocks[block];

                for (std::size_t il23 = r.begin; il23 < r.end; ++il23) {
                    const std::size_t src_idx =
                        flat_B3_contraction_native(il23, il34, il35, ik3, L23, L34, L35);
                    const std::size_t dst_idx = r.offset + (il23 - r.begin);
                    const b4utils::Complex64 z = src[src_idx];
                    dst_re[dst_idx] = z.re;
                    dst_im[dst_idx] = z.im;
                }
            }
        }
    }
}

void pack_B4_complex_from_contraction_native(
    const AmplitudeInputSupportPlan& support,
    const Tensor<b4utils::Complex64, 4>& dense_B4,
    b4utils::ComplexBuffer<>& packed_B4)
{
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L45 = support.l45.size();

    const b4utils::Complex64* __restrict src = dense_B4.data();
    double* __restrict dst_re = packed_B4.re_data();
    double* __restrict dst_im = packed_B4.im_data();

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (std::ptrdiff_t ik4_s = 0; ik4_s < static_cast<std::ptrdiff_t>(K4d); ++ik4_s) {
        for (std::ptrdiff_t il24_s = 0; il24_s < static_cast<std::ptrdiff_t>(L24); ++il24_s) {
            for (std::ptrdiff_t il34_s = 0; il34_s < static_cast<std::ptrdiff_t>(L34); ++il34_s) {
                const std::size_t ik4  = static_cast<std::size_t>(ik4_s);
                const std::size_t il24 = static_cast<std::size_t>(il24_s);
                const std::size_t il34 = static_cast<std::size_t>(il34_s);
                const std::size_t block = idx3(ik4, il24, il34, L24, L34);
                const Range1D& r = support.B4.blocks[block];

                for (std::size_t il45 = r.begin; il45 < r.end; ++il45) {
                    const std::size_t src_idx =
                        flat_B4_contraction_native(il45, il24, il34, ik4, L45, L24, L34);
                    const std::size_t dst_idx = r.offset + (il45 - r.begin);
                    const b4utils::Complex64 z = src[src_idx];
                    dst_re[dst_idx] = z.re;
                    dst_im[dst_idx] = z.im;
                }
            }
        }
    }
}

void pack_B5_complex_from_contraction_native(
    const AmplitudeInputSupportPlan& support,
    const Tensor<b4utils::Complex64, 4>& dense_B5,
    b4utils::ComplexBuffer<>& packed_B5)
{
    const std::size_t L25 = support.l25.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    const b4utils::Complex64* __restrict src = dense_B5.data();
    double* __restrict dst_re = packed_B5.re_data();
    double* __restrict dst_im = packed_B5.im_data();

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
                const Range1D& r = support.B5.blocks[block];

                for (std::size_t ik5 = r.begin; ik5 < r.end; ++ik5) {
                    const std::size_t src_idx =
                        flat_B5_contraction_native(il45, il25, il35, ik5, L45, L25, L35);
                    const std::size_t dst_idx = r.offset + (ik5 - r.begin);
                    const b4utils::Complex64 z = src[src_idx];
                    dst_re[dst_idx] = z.re;
                    dst_im[dst_idx] = z.im;
                }
            }
        }
    }
}

void b4_contract_rank5_slab_to_packed_B2(
    const AmplitudeInputSupportPlan& support,
    std::size_t b_global_begin,
    const std::vector<b4utils::Complex64>& coeff,
    const Tensor<double, 5>& slab,
    b4utils::ComplexBuffer<>* out,
    CoherentQSlabNativeBoosters* native)
{
    const auto& d = slab.dims();
    const auto st = slab.strides();
    const std::size_t nb = d[4];
    const double* __restrict x = slab.data();

#ifndef NDEBUG
    assert(d[0] == support.l23.size());
    assert(d[1] == support.l24.size());
    assert(d[2] == support.l25.size());
    assert(d[3] == coeff.size());
#endif

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(nb * d[2] >= 32)
#endif
    for (std::ptrdiff_t ib_s = 0; ib_s < static_cast<std::ptrdiff_t>(nb); ++ib_s) {
        for (std::ptrdiff_t i2_s = 0; i2_s < static_cast<std::ptrdiff_t>(d[2]); ++i2_s) {
            const std::size_t ib_local = static_cast<std::size_t>(ib_s);
            const std::size_t ib = b_global_begin + ib_local;
            const std::size_t i2 = static_cast<std::size_t>(i2_s);
            for (std::size_t i1 = 0; i1 < d[1]; ++i1) {
                for (std::size_t i0 = 0; i0 < d[0]; ++i0) {
                    const Range1D& r = support.B2.blocks[
                        idx3(i0, i1, i2, d[1], d[2])];
                    if (ib < r.begin || ib >= r.end) continue;
                    const std::size_t src = i0 * st[0] + i1 * st[1] +
                        i2 * st[2] + ib_local * st[4];
                    const b4utils::Complex64 z =
                        b4_coherent_boundary_dot(x, src, st[3], coeff);
                    if (out != nullptr) {
                        b4_store_direct_packed(
                            *out, r.offset + ib - r.begin, z);
                    }
                    if (native != nullptr) {
                        const std::size_t K = support.channels.k2.size();
                        const std::size_t P = support.l25.size();
                        const std::size_t U = support.l23.size();
                        native->B2[ib + K * (i2 + P * (i0 + U * i1))] =
                            {z.re, z.im};
                    }
                }
            }
        }
    }
}

void b4_contract_rank5_slab_to_packed_B3(
    const AmplitudeInputSupportPlan& support,
    std::size_t b_global_begin,
    std::span<const int> a_vals,
    std::span<const int> b_vals,
    const std::vector<b4utils::Complex64>& coeff,
    const Tensor<double, 5>& slab,
    b4utils::ComplexBuffer<>* out,
    CoherentQSlabNativeBoosters* native,
    bool exchange34,
    int Dj3,
    int Dj4,
    int Dl3_base,
    int Dl4_fixed)
{
    const auto& d = slab.dims();
    const auto st = slab.strides();
    const std::size_t nb = d[4];
    const double* __restrict x = slab.data();

#ifndef NDEBUG
    assert(d[0] == support.l34.size());
    assert(d[1] == support.l35.size());
    assert(d[2] == support.l23.size());
    assert(d[3] == coeff.size());
    assert(a_vals.size() == coeff.size());
    assert(b_vals.size() == nb);
#endif

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(nb * d[1] >= 32)
#endif
    for (std::ptrdiff_t ib_s = 0; ib_s < static_cast<std::ptrdiff_t>(nb); ++ib_s) {
        for (std::ptrdiff_t i1_s = 0; i1_s < static_cast<std::ptrdiff_t>(d[1]); ++i1_s) {
            const std::size_t ib_local = static_cast<std::size_t>(ib_s);
            const std::size_t ib = b_global_begin + ib_local;
            const std::size_t i1 = static_cast<std::size_t>(i1_s);
            for (std::size_t i0 = 0; i0 < d[0]; ++i0) {
                const Range1D& r = support.B3.blocks[
                    idx3(ib, i0, i1, d[0], d[1])];
                for (std::size_t i2 = r.begin; i2 < r.end; ++i2) {
                    const std::size_t src = i0 * st[0] + i1 * st[1] +
                        i2 * st[2] + ib_local * st[4];
                    b4utils::Complex64 z{};
                    if (!exchange34) {
                        z = b4_coherent_boundary_dot(x, src, st[3], coeff);
                    } else {
                        const int Db = b_vals[ib_local];
                        const int Dl3 = Dl3_base + 2 * static_cast<int>(i2);
                        for (std::size_t ia = 0; ia < coeff.size(); ++ia) {
                            const double phase = static_cast<double>(
                                b4_phase_from_even_Dsum_local(
                                    Dj3 + Dj4 + a_vals[ia] + Dl3 + Dl4_fixed + Db));
                            const double v = x[src + ia * st[3]] * phase;
                            z.re += v * coeff[ia].re;
                            z.im += v * coeff[ia].im;
                        }
                    }
                    if (out != nullptr) {
                        b4_store_direct_packed(*out, r.offset + i2 - r.begin, z);
                    }
                    if (native != nullptr) {
                        const std::size_t I = support.channels.k3.size();
                        const std::size_t U = support.l23.size();
                        const std::size_t S = support.l35.size();
                        native->B3[ib + I * (i2 + U * (i1 + S * i0))] =
                            {z.re, z.im};
                    }
                }
            }
        }
    }
}

void b4_contract_rank5_slab_to_packed_B4(
    const AmplitudeInputSupportPlan& support,
    std::size_t b_global_begin,
    std::span<const int> a_vals,
    std::span<const int> b_vals,
    const std::vector<b4utils::Complex64>& coeff,
    const Tensor<double, 5>& slab,
    b4utils::ComplexBuffer<>* out,
    CoherentQSlabNativeBoosters* native,
    bool pair_exchange_120)
{
    const auto& d = slab.dims();
    const auto st = slab.strides();
    const std::size_t nb = d[4];
    const double* __restrict x = slab.data();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();

#ifndef NDEBUG
    const std::size_t L45 = support.l45.size();
    assert(d[3] == coeff.size());
    assert(a_vals.size() == coeff.size());
    assert(b_vals.size() == nb);
    if (pair_exchange_120) {
        assert(d[0] == L24 && d[1] == L34 && d[2] == L45);
    } else {
        assert(d[0] == L45 && d[1] == L24 && d[2] == L34);
    }
#endif

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(nb * L34 >= 32)
#endif
    for (std::ptrdiff_t ib_s = 0; ib_s < static_cast<std::ptrdiff_t>(nb); ++ib_s) {
        for (std::ptrdiff_t i2_s = 0; i2_s < static_cast<std::ptrdiff_t>(L34); ++i2_s) {
            const std::size_t ib_local = static_cast<std::size_t>(ib_s);
            const std::size_t ib = b_global_begin + ib_local;
            const std::size_t i2 = static_cast<std::size_t>(i2_s);
            for (std::size_t i1 = 0; i1 < L24; ++i1) {
                const Range1D& r = support.B4.blocks[
                    idx3(ib, i1, i2, L24, L34)];
                const int Db = b_vals[ib_local];
                for (std::size_t i0 = r.begin; i0 < r.end; ++i0) {
                    const std::size_t src = pair_exchange_120
                        ? i1 * st[0] + i2 * st[1] + i0 * st[2] + ib_local * st[4]
                        : i0 * st[0] + i1 * st[1] + i2 * st[2] + ib_local * st[4];
                    b4utils::Complex64 z{};
                    if (!pair_exchange_120) {
                        z = b4_coherent_boundary_dot(x, src, st[3], coeff);
                    } else {
                        for (std::size_t ia = 0; ia < coeff.size(); ++ia) {
                            const double phase = static_cast<double>(
                                b4_phase_pair_exchange_local(a_vals[ia], Db));
                            const double v = x[src + ia * st[3]] * phase;
                            z.re += v * coeff[ia].re;
                            z.im += v * coeff[ia].im;
                        }
                    }
                    if (out != nullptr) {
                        b4_store_direct_packed(*out, r.offset + i0 - r.begin, z);
                    }
                    if (native != nullptr) {
                        const std::size_t J = support.channels.k4.size();
                        const std::size_t T = support.l45.size();
                        const std::size_t R = support.l34.size();
                        native->B4[ib + J * (i0 + T * (i2 + R * i1))] =
                            {z.re, z.im};
                    }
                }
            }
        }
    }
}

void b4_contract_rank5_slab_to_packed_B5(
    const AmplitudeInputSupportPlan& support,
    std::size_t b_global_begin,
    std::span<const int> a_vals,
    std::span<const int> b_vals,
    const std::vector<b4utils::Complex64>& coeff,
    const Tensor<double, 5>& slab,
    b4utils::ComplexBuffer<>* out,
    CoherentQSlabNativeBoosters* native,
    B4PackedB5SlabTransform transform,
    int Dj3,
    int Dj4,
    int Dl3_base,
    int Dl4_fixed)
{
    const auto& d = slab.dims();
    const auto st = slab.strides();
    const std::size_t nb = d[4];
    const double* __restrict x = slab.data();
    const std::size_t L25 = support.l25.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();
    const bool pair = transform != B4PackedB5SlabTransform::direct;

#ifndef NDEBUG
    assert(d[3] == coeff.size());
    assert(a_vals.size() == coeff.size());
    assert(b_vals.size() == nb);
    if (pair) {
        assert(d[0] == L35 && d[1] == L45 && d[2] == L25);
    } else {
        assert(d[0] == L25 && d[1] == L35 && d[2] == L45);
    }
#endif

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(nb * L45 >= 32)
#endif
    for (std::ptrdiff_t ib_s = 0; ib_s < static_cast<std::ptrdiff_t>(nb); ++ib_s) {
        for (std::ptrdiff_t i2_s = 0; i2_s < static_cast<std::ptrdiff_t>(L45); ++i2_s) {
            const std::size_t ib_local = static_cast<std::size_t>(ib_s);
            const std::size_t ib = b_global_begin + ib_local;
            const std::size_t i2 = static_cast<std::size_t>(i2_s);
            const int Db = b_vals[ib_local];
            for (std::size_t i1 = 0; i1 < L35; ++i1) {
                for (std::size_t i0 = 0; i0 < L25; ++i0) {
                    const Range1D& r = support.B5.blocks[
                        idx3(i0, i1, i2, L35, L45)];
                    if (ib < r.begin || ib >= r.end) continue;
                    const std::size_t src = pair
                        ? i1 * st[0] + i2 * st[1] + i0 * st[2] + ib_local * st[4]
                        : i0 * st[0] + i1 * st[1] + i2 * st[2] + ib_local * st[4];
                    b4utils::Complex64 z{};
                    for (std::size_t ia = 0; ia < coeff.size(); ++ia) {
                        double phase = 1.0;
                        if (pair) {
                            phase = static_cast<double>(
                                b4_phase_pair_exchange_local(a_vals[ia], Db));
                        }
                        if (transform == B4PackedB5SlabTransform::exchange34_then_pair_120) {
                            const int Dl3 = Dl3_base + 2 * static_cast<int>(i0);
                            phase *= static_cast<double>(b4_phase_from_even_Dsum_local(
                                Dj3 + Dj4 + a_vals[ia] + Dl3 + Dl4_fixed + Db));
                        }
                        const double v = x[src + ia * st[3]] * phase;
                        z.re += v * coeff[ia].re;
                        z.im += v * coeff[ia].im;
                    }
                    if (out != nullptr) {
                        b4_store_direct_packed(*out, r.offset + ib - r.begin, z);
                    }
                    if (native != nullptr) {
                        const std::size_t V = support.channels.k5.size();
                        const std::size_t P = support.l25.size();
                        const std::size_t T = support.l45.size();
                        native->B5[ib + V * (i0 + P * (i2 + T * i1))] =
                            {z.re, z.im};
                    }
                }
            }
        }
    }
}

void fill_amplitude_coherent_boosters_direct_outputs(
    const B4Spins& spins,
    const std::array<std::size_t, 6>& Dim_ls,
    double gamma,
    const std::vector<double>& FactTable,
    const AmplitudeInputSupportPlan& support,
    const B4CoherentCoefficientsView& C2,
    const B4CoherentCoefficientsView& C3,
    const B4CoherentCoefficientsView& C4,
    const B4CoherentCoefficientsView& C5,
    CoherentBoosterOutputTarget target,
    PackedAmplitudeCoherentBoosters* packed,
    CoherentQSlabNativeBoosters* q_slab_native,
    std::uint64_t slab_target_bytes,
    CoherentBoosterProductionStats* stats,
    sl2cfoamxx::SimplicityConstraint prescription,
    std::uint32_t rho_tile)
{
    const auto production_start = std::chrono::steady_clock::now();
    B4HarvestThreadStats* thread_stats = nullptr;
    if (stats != nullptr) {
        stats->harvest_threads = {};
        thread_stats = &stats->harvest_threads;
    }
    b4_assert_valid_coherent_coefficients(C2);
    b4_assert_valid_coherent_coefficients(C3);
    b4_assert_valid_coherent_coefficients(C4);
    b4_assert_valid_coherent_coefficients(C5);

#ifndef NDEBUG
    assert(Dim_ls[0] == support.l23.size());
    assert(Dim_ls[1] == support.l24.size());
    assert(Dim_ls[2] == support.l25.size());
    assert(Dim_ls[3] == support.l34.size());
    assert(Dim_ls[4] == support.l35.size());
    assert(Dim_ls[5] == support.l45.size());
#endif

    const bool want_packed = coherent_booster_wants_support_packed(target);
    const bool want_native = coherent_booster_wants_q_slab_native(target);
    if ((want_packed && packed == nullptr) ||
        (want_native && q_slab_native == nullptr)) {
        throw std::invalid_argument(
            "coherent booster output target is missing its destination");
    }

    // Measure headroom before zero-initializing the requested outputs: those
    // pages are part of the explicitly reserved future payload below.
    if (slab_target_bytes == 0) {
        slab_target_bytes =
            b4_choose_coherent_direct_rank5_slab_budget_bytes(support, target);
    }

    if (want_packed) {
        packed->allocate_from_support(support);
        packed->B2.zero();
        packed->B3.zero();
        packed->B4.zero();
        packed->B5.zero();
    }
    if (want_native) {
        q_slab_native->allocate_from_support(support);
    }

    b4utils::ComplexBuffer<>* packed_B2 = want_packed ? &packed->B2 : nullptr;
    b4utils::ComplexBuffer<>* packed_B3 = want_packed ? &packed->B3 : nullptr;
    b4utils::ComplexBuffer<>* packed_B4 = want_packed ? &packed->B4 : nullptr;
    b4utils::ComplexBuffer<>* packed_B5 = want_packed ? &packed->B5 : nullptr;
    CoherentQSlabNativeBoosters* native_output =
        want_native ? q_slab_native : nullptr;

    const AmplitudeChannels& ch = support.channels;
    const std::array<std::size_t, 6> ell0{0, 0, 0, 0, 0, 0};

    const bool make_B3_from_B2 =
        can_derive_B3_from_B2_exchange34_coherent(spins, Dim_ls, ell0, ch);
    const bool make_B4_from_B2 =
        can_derive_B4_from_B2_pair_exchange_coherent(spins, Dim_ls, ell0, ch);
    const bool make_B5_from_B3 =
        can_derive_B5_from_B3_pair_exchange_coherent(spins, Dim_ls, ell0, ch);
    const bool make_B5_from_B2 = make_B3_from_B2 && make_B5_from_B3;

#if B4_ENABLE_DIAGNOSTICS
    std::cerr << "direct packed coherent booster symmetry plan\n"
              << "  B3 from B2 = " << (make_B3_from_B2 ? "yes" : "no") << "\n"
              << "  B4 from B2 = " << (make_B4_from_B2 ? "yes" : "no") << "\n"
              << "  B5 from B3 = " << (make_B5_from_B3 ? "yes" : "no") << "\n"
              << "  slab budget = " << slab_target_bytes << " bytes\n";
#endif

    // B2 is the primary representative and can fan out to all four packed
    // tensors when the exact spin/channel/layout guards hold.
    {
        const B4CoherentCoeffUnion4 U = b4_make_coherent_coeff_union4(
            true, C2,
            make_B3_from_B2, C3,
            make_B4_from_B2, C4,
            make_B5_from_B2, C5);
        if (!U.empty()) {
            const std::array<std::size_t, 3> D{
                Dim_ls[0], Dim_ls[1], Dim_ls[2]};
            const std::size_t b_size = ch.k2.size();
            const std::size_t b_slab = b4_choose_coherent_rank5_b_slab_size(
                D[0] * D[1] * D[2], U.size(), b_size, slab_target_bytes);
            const bool reduce12 =
                can_reduce_B2_exchange12_harvest_amp(spins, Dim_ls, ell0);
            const std::span<const int> a_span(U.a.data(), U.a.size());
            const bool use2 = b4_any_nonzero_coeff(U.c2);
            const bool use3 = make_B3_from_B2 && b4_any_nonzero_coeff(U.c3);
            const bool use4 = make_B4_from_B2 && b4_any_nonzero_coeff(U.c4);
            const bool use5 = make_B5_from_B2 && b4_any_nonzero_coeff(U.c5);

            if (b_slab != 0) {
                Tensor<double, 5> full(
                    {D[0], D[1], D[2], U.size(), b_slab}, false);
                for (std::size_t b_begin = 0; b_begin < b_size; b_begin += b_slab) {
                    const std::size_t nb = std::min(b_slab, b_size - b_begin);
                    const std::span<const int> b_sub(ch.k2.data() + b_begin, nb);
                    std::unique_ptr<Tensor<double, 5>> tail;
                    Tensor<double, 5>* slab = &full;
                    if (nb != b_slab) {
                        tail = std::make_unique<Tensor<double, 5>>(
                            std::array<std::size_t, 5>{D[0], D[1], D[2], U.size(), nb},
                            false);
                        slab = tail.get();
                    }

                    harvest_b4<4, 5>(
                        spins.Dj23, spins.Dj24, spins.Dj25, spins.Dj12,
                        D, a_span, b_sub, gamma, FactTable,
                        *slab, {0, 0, 0}, reduce12, prescription, rho_tile, thread_stats);
                    if (reduce12) {
                        complete_rank5_exchange12_lower_from_upper(
                            *slab, spins.Dj23, spins.Dj24,
                            spins.Dj23, spins.Dj24, a_span, b_sub);
                    }

                    if (use2) {
                        b4_contract_rank5_slab_to_packed_B2(
                            support, b_begin, U.c2, *slab,
                            packed_B2, native_output);
                    }
                    if (use3) {
                        b4_contract_rank5_slab_to_packed_B3(
                            support, b_begin, a_span, b_sub, U.c3, *slab,
                            packed_B3, native_output, true,
                            spins.Dj25, spins.Dj12, spins.Dj25, spins.Dj12);
                    }
                    if (use4) {
                        b4_contract_rank5_slab_to_packed_B4(
                            support, b_begin, a_span, b_sub, U.c4, *slab,
                            packed_B4, native_output, true);
                    }
                    if (use5) {
                        b4_contract_rank5_slab_to_packed_B5(
                            support, b_begin, a_span, b_sub, U.c5, *slab,
                            packed_B5, native_output,
                            B4PackedB5SlabTransform::exchange34_then_pair_120,
                            spins.Dj25, spins.Dj12, spins.Dj25, spins.Dj12);
                    }
                }
            }
        }
    }

    // If B3 is not symmetry-equivalent to B2, harvest it once.  This slab may
    // still feed B5 through the pair-exchange symmetry.
    if (!make_B3_from_B2) {
        const bool derive5 = make_B5_from_B3;
        const B4CoherentCoeffUnion4 U = b4_make_coherent_coeff_union4(
            false, C2, true, C3, false, C4, derive5, C5);
        if (!U.empty()) {
            const std::array<std::size_t, 3> D{
                Dim_ls[3], Dim_ls[4], Dim_ls[0]};
            const std::size_t b_size = ch.k3.size();
            const std::size_t b_slab = b4_choose_coherent_rank5_b_slab_size(
                D[0] * D[1] * D[2], U.size(), b_size, slab_target_bytes);
            const bool reduce12 = spins.Dj34 == spins.Dj35 &&
                Dim_ls[3] == Dim_ls[4];
            const std::span<const int> a_span(U.a.data(), U.a.size());
            const bool use3 = b4_any_nonzero_coeff(U.c3);
            const bool use5 = derive5 && b4_any_nonzero_coeff(U.c5);

            if (b_slab != 0) {
                Tensor<double, 5> full(
                    {D[0], D[1], D[2], U.size(), b_slab}, false);
                for (std::size_t b_begin = 0; b_begin < b_size; b_begin += b_slab) {
                    const std::size_t nb = std::min(b_slab, b_size - b_begin);
                    const std::span<const int> b_sub(ch.k3.data() + b_begin, nb);
                    std::unique_ptr<Tensor<double, 5>> tail;
                    Tensor<double, 5>* slab = &full;
                    if (nb != b_slab) {
                        tail = std::make_unique<Tensor<double, 5>>(
                            std::array<std::size_t, 5>{D[0], D[1], D[2], U.size(), nb},
                            false);
                        slab = tail.get();
                    }

                    harvest_b4<3, 5>(
                        spins.Dj34, spins.Dj35, spins.Dj13, spins.Dj23,
                        D, a_span, b_sub, gamma, FactTable,
                        *slab, {0, 0, 0}, reduce12, prescription, rho_tile, thread_stats);
                    if (reduce12) {
                        complete_rank5_exchange12_lower_from_upper(
                            *slab, spins.Dj34, spins.Dj35,
                            spins.Dj34, spins.Dj35, a_span, b_sub);
                    }
                    if (use3) {
                        b4_contract_rank5_slab_to_packed_B3(
                            support, b_begin, a_span, b_sub, U.c3, *slab,
                            packed_B3, native_output, false);
                    }
                    if (use5) {
                        b4_contract_rank5_slab_to_packed_B5(
                            support, b_begin, a_span, b_sub, U.c5, *slab,
                            packed_B5, native_output,
                            B4PackedB5SlabTransform::pair_exchange_120);
                    }
                }
            }
        }
    }

    // Independent B4 fallback.
    if (!make_B4_from_B2) {
        B4ActiveCoherentCoefficients A = b4_make_active_coherent_coefficients(C4);
        if (!A.empty()) {
            const std::array<std::size_t, 3> D{
                Dim_ls[5], Dim_ls[1], Dim_ls[3]};
            const std::size_t b_size = ch.k4.size();
            const std::size_t b_slab = b4_choose_coherent_rank5_b_slab_size(
                D[0] * D[1] * D[2], A.size(), b_size, slab_target_bytes);
            const std::span<const int> a_span(A.a.data(), A.a.size());
            if (b_slab != 0) {
                Tensor<double, 5> full(
                    {D[0], D[1], D[2], A.size(), b_slab}, false);
                for (std::size_t b_begin = 0; b_begin < b_size; b_begin += b_slab) {
                    const std::size_t nb = std::min(b_slab, b_size - b_begin);
                    const std::span<const int> b_sub(ch.k4.data() + b_begin, nb);
                    std::unique_ptr<Tensor<double, 5>> tail;
                    Tensor<double, 5>* slab = &full;
                    if (nb != b_slab) {
                        tail = std::make_unique<Tensor<double, 5>>(
                            std::array<std::size_t, 5>{D[0], D[1], D[2], A.size(), nb},
                            false);
                        slab = tail.get();
                    }
                    harvest_b4<2, 5>(
                        spins.Dj45, spins.Dj14, spins.Dj24, spins.Dj34,
                        D, a_span, b_sub, gamma, FactTable,
                        *slab, {0, 0, 0}, false, prescription, rho_tile, thread_stats);
                    b4_contract_rank5_slab_to_packed_B4(
                        support, b_begin, a_span, b_sub, A.c, *slab,
                        packed_B4, native_output, false);
                }
            }
        }
    }

    // Independent B5 fallback.  The condition mirrors the dense symmetry
    // producer: exactly one route owns every packed B5 value.
    if (!make_B5_from_B2 && !(make_B5_from_B3 && !make_B3_from_B2)) {
        B4ActiveCoherentCoefficients A = b4_make_active_coherent_coefficients(C5);
        if (!A.empty()) {
            const std::array<std::size_t, 3> D{
                Dim_ls[2], Dim_ls[4], Dim_ls[5]};
            const std::size_t b_size = ch.k5.size();
            const std::size_t b_slab = b4_choose_coherent_rank5_b_slab_size(
                D[0] * D[1] * D[2], A.size(), b_size, slab_target_bytes);
            const std::span<const int> a_span(A.a.data(), A.a.size());
            if (b_slab != 0) {
                Tensor<double, 5> full(
                    {D[0], D[1], D[2], A.size(), b_slab}, false);
                for (std::size_t b_begin = 0; b_begin < b_size; b_begin += b_slab) {
                    const std::size_t nb = std::min(b_slab, b_size - b_begin);
                    const std::span<const int> b_sub(ch.k5.data() + b_begin, nb);
                    std::unique_ptr<Tensor<double, 5>> tail;
                    Tensor<double, 5>* slab = &full;
                    if (nb != b_slab) {
                        tail = std::make_unique<Tensor<double, 5>>(
                            std::array<std::size_t, 5>{D[0], D[1], D[2], A.size(), nb},
                            false);
                        slab = tail.get();
                    }
                    harvest_b4<1, 5>(
                        spins.Dj15, spins.Dj25, spins.Dj35, spins.Dj45,
                        D, a_span, b_sub, gamma, FactTable,
                        *slab, {0, 0, 0}, false, prescription, rho_tile, thread_stats);
                    b4_contract_rank5_slab_to_packed_B5(
                        support, b_begin, a_span, b_sub, A.c, *slab,
                        packed_B5, native_output,
                        B4PackedB5SlabTransform::direct);
                }
            }
        }
    }

    if (stats != nullptr) {
        stats->direct_support_packed = want_packed;
        stats->slab_budget_bytes = slab_target_bytes;
        stats->packed_payload_bytes = want_packed ? packed->payload_bytes() : 0;
        stats->q_slab_native_payload_bytes =
            want_native ? q_slab_native->payload_bytes() : 0;
        stats->avoided_dense_payload_bytes =
            b4_coherent_dense_native_payload_bytes(support);
        stats->total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - production_start).count();
    }
}

void fill_packed_amplitude_coherent_boosters_via_canonical(
    const B4Spins& spins,
    const std::array<std::size_t, 6>& Dim_ls,
    double gamma,
    const std::vector<double>& FactTable,
    const AmplitudeInputSupportPlan& support,
    const B4CoherentCoefficientsView& C2,
    const B4CoherentCoefficientsView& C3,
    const B4CoherentCoefficientsView& C4,
    const B4CoherentCoefficientsView& C5,
    PackedAmplitudeCoherentBoosters& packed,
    std::uint64_t slab_target_bytes,
    CoherentBoosterProductionStats* stats,
    sl2cfoamxx::SimplicityConstraint prescription,
    std::uint32_t rho_tile)
{
    const auto production_start = std::chrono::steady_clock::now();
    if (stats != nullptr) stats->harvest_threads = {};
    const AmplitudeChannels& AmC = support.channels;

    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    const std::size_t Dim_k2 = AmC.k2.size();
    const std::size_t Dim_k3 = AmC.k3.size();
    const std::size_t Dim_k4 = AmC.k4.size();
    const std::size_t Dim_k5 = AmC.k5.size();

#ifndef NDEBUG
    assert(Dim_ls[0] == L23);
    assert(Dim_ls[1] == L24);
    assert(Dim_ls[2] == L25);
    assert(Dim_ls[3] == L34);
    assert(Dim_ls[4] == L35);
    assert(Dim_ls[5] == L45);
#endif

    packed.allocate_from_support(support);

    if (slab_target_bytes == 0) {
        slab_target_bytes = b4_choose_coherent_rank5_slab_budget_bytes(support);
    }

    Tensor<b4utils::Complex64, 4> dense_B2({L23, L25, L24, Dim_k2}, true);
    Tensor<b4utils::Complex64, 4> dense_B3({L23, L34, L35, Dim_k3}, true);
    Tensor<b4utils::Complex64, 4> dense_B4({L45, L24, L34, Dim_k4}, true);
    Tensor<b4utils::Complex64, 4> dense_B5({L45, L25, L35, Dim_k5}, true);

    fill_rank4_tensors_coherent_contraction_native(
        spins,
        Dim_ls,
        gamma,
        FactTable,
        AmC,
        C2, C3, C4, C5,
        dense_B2,
        dense_B3,
        dense_B4,
        dense_B5,
        slab_target_bytes,
        prescription,
        rho_tile);

    pack_B2_complex_from_contraction_native(support, dense_B2, packed.B2);
    pack_B3_complex_from_contraction_native(support, dense_B3, packed.B3);
    pack_B4_complex_from_contraction_native(support, dense_B4, packed.B4);
    pack_B5_complex_from_contraction_native(support, dense_B5, packed.B5);

    if (stats != nullptr) {
        stats->direct_support_packed = false;
        stats->slab_budget_bytes = slab_target_bytes;
        stats->packed_payload_bytes = packed.payload_bytes();
        stats->q_slab_native_payload_bytes = 0;
        stats->avoided_dense_payload_bytes = 0;
        stats->total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - production_start).count();
    }
}

} // namespace b4support
