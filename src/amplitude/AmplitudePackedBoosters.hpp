// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cerrno>
#include <cstring>

#include "amplitude/AmplitudeInputSupport.hpp"
#include "boosters/B4TensorFilling.hpp"
#include "core/HostMemoryBudget.hpp"
#include "gpu/CoherentGpuApi.hpp"
#include "core/utils.hpp"

namespace b4support {

// ============================================================================
// Packed booster tensors
// ============================================================================
//
// Layouts are defined by AmplitudeInputSupportPlan:
//
//   B2 support: block (l23,l24,l25) -> k2
//   B3 support: block (k3,l34,l35)  -> l23
//   B4 support: block (k4,l24,l34)  -> l45
//   B5 support: block (l25,l35,l45) -> k5
//
// Each packed tensor is one continuous values[] array.
// ============================================================================

struct PackedAmplitudeBoosters {
    std::vector<double> B2;
    std::vector<double> B3;
    std::vector<double> B4;
    std::vector<double> B5;
    
    void allocate_from_support(const AmplitudeInputSupportPlan& support) {
        B2.resize(static_cast<std::size_t>(support.B2.packed_values));
        B3.resize(static_cast<std::size_t>(support.B3.packed_values));
        B4.resize(static_cast<std::size_t>(support.B4.packed_values));
        B5.resize(static_cast<std::size_t>(support.B5.packed_values));
    }
    
    void clear() {
        B2.clear();
        B3.clear();
        B4.clear();
        B5.clear();
    }
    
    std::uint64_t payload_bytes() const noexcept {
        return static_cast<std::uint64_t>(
                                          (B2.size() + B3.size() + B4.size() + B5.size())
                                          * sizeof(double)
                                          );
    }
    
    void print(std::ostream& os) const {
        const double mib = 1024.0 * 1024.0;
        
        os << "\nPacked amplitude booster tensors\n";
        os << "================================\n";
        os << "B2 values = " << B2.size()
        << " | " << static_cast<double>(B2.size() * sizeof(double)) / mib
        << " MiB\n";
        os << "B3 values = " << B3.size()
        << " | " << static_cast<double>(B3.size() * sizeof(double)) / mib
        << " MiB\n";
        os << "B4 values = " << B4.size()
        << " | " << static_cast<double>(B4.size() * sizeof(double)) / mib
        << " MiB\n";
        os << "B5 values = " << B5.size()
        << " | " << static_cast<double>(B5.size() * sizeof(double)) / mib
        << " MiB\n";
        os << "total     = "
        << static_cast<double>(payload_bytes()) / mib
        << " MiB\n";
    }
};


struct PackedAmplitudeCoherentBoosters {
    b4utils::ComplexBuffer<> B2;
    b4utils::ComplexBuffer<> B3;
    b4utils::ComplexBuffer<> B4;
    b4utils::ComplexBuffer<> B5;

    void allocate_from_support(const AmplitudeInputSupportPlan& support) {
        B2.resize_discard(static_cast<std::size_t>(support.B2.packed_values));
        B3.resize_discard(static_cast<std::size_t>(support.B3.packed_values));
        B4.resize_discard(static_cast<std::size_t>(support.B4.packed_values));
        B5.resize_discard(static_cast<std::size_t>(support.B5.packed_values));
    }

    void clear() noexcept {
        B2.release();
        B3.release();
        B4.release();
        B5.release();
    }

    std::uint64_t payload_bytes() const noexcept {
        return static_cast<std::uint64_t>(B2.bytes() + B3.bytes() + B4.bytes() + B5.bytes());
    }

    void print(std::ostream& os) const {
        const double mib = 1024.0 * 1024.0;
        os << "\nPacked coherent amplitude booster tensors\n";
        os << "=========================================\n";
        os << "B2 values = " << B2.size() << " complex | " << static_cast<double>(B2.bytes()) / mib << " MiB\n";
        os << "B3 values = " << B3.size() << " complex | " << static_cast<double>(B3.bytes()) / mib << " MiB\n";
        os << "B4 values = " << B4.size() << " complex | " << static_cast<double>(B4.bytes()) / mib << " MiB\n";
        os << "B5 values = " << B5.size() << " complex | " << static_cast<double>(B5.bytes()) / mib << " MiB\n";
        os << "total     = " << static_cast<double>(payload_bytes()) / mib << " MiB\n";
    }
};

// Dense host layout consumed by the CUDA q-slab route.  The contraction axes
// are deliberately ordered with virtual intertwiner contiguous, matching the device upload:
//
//   B2[k,p,u,q], B3[i,u,s,r], B4[j,t,r,q], B5[v,p,t,s].
//
// This is not the old recurrence-canonical rank-4 layout.  Producing it while
// a real rank-5 slab is resident avoids a later support-packed -> dense scan.
inline std::size_t b4_checked_coherent_booster_count(
    std::initializer_list<std::size_t> extents)
{
    std::size_t count = 1;
    for (const std::size_t extent : extents) {
        if (extent != 0 &&
            count > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::overflow_error("coherent booster extent product overflow");
        }
        count *= extent;
    }
    return count;
}

struct CoherentQSlabNativeBoosters {
    b4gpu::DenseCoherentDimensions dimensions{};
    std::vector<b4gpu::Complex64> B2;
    std::vector<b4gpu::Complex64> B3;
    std::vector<b4gpu::Complex64> B4;
    std::vector<b4gpu::Complex64> B5;

    void allocate_from_support(const AmplitudeInputSupportPlan& support) {
        const std::size_t K = support.channels.k2.size();
        const std::size_t I = support.channels.k3.size();
        const std::size_t J = support.channels.k4.size();
        const std::size_t V = support.channels.k5.size();
        const std::size_t U = support.l23.size();
        const std::size_t Q = support.l24.size();
        const std::size_t P = support.l25.size();
        const std::size_t R = support.l34.size();
        const std::size_t S = support.l35.size();
        const std::size_t T = support.l45.size();
        dimensions = b4gpu::DenseCoherentDimensions{
            0, K, I, J, V, U, Q, P, R, S, T};
        B2.assign(b4_checked_coherent_booster_count({K, P, U, Q}), {0.0, 0.0});
        B3.assign(b4_checked_coherent_booster_count({I, U, S, R}), {0.0, 0.0});
        B4.assign(b4_checked_coherent_booster_count({J, T, R, Q}), {0.0, 0.0});
        B5.assign(b4_checked_coherent_booster_count({V, P, T, S}), {0.0, 0.0});
    }

    void clear() noexcept {
        B2.clear(); B3.clear(); B4.clear(); B5.clear();
    }

    std::uint64_t payload_bytes() const noexcept {
        std::uint64_t values = static_cast<std::uint64_t>(B2.size());
        values = host_memory::sat_add(values, B3.size());
        values = host_memory::sat_add(values, B4.size());
        values = host_memory::sat_add(values, B5.size());
        return host_memory::sat_mul(values, sizeof(b4gpu::Complex64));
    }

    [[nodiscard]] b4gpu::DenseCoherentBoosters view(
        std::size_t coherent_i1_count) const noexcept {
        b4gpu::DenseCoherentDimensions d = dimensions;
        d.a = coherent_i1_count;
        return b4gpu::DenseCoherentBoosters{
            d,
            b4gpu::make_first_axis_contiguous_view<b4gpu::Complex64, 4>(
                B2.data(), {d.k2, d.l25, d.l23, d.l24}),
            b4gpu::make_first_axis_contiguous_view<b4gpu::Complex64, 4>(
                B3.data(), {d.k3, d.l23, d.l35, d.l34}),
            b4gpu::make_first_axis_contiguous_view<b4gpu::Complex64, 4>(
                B4.data(), {d.k4, d.l45, d.l34, d.l24}),
            b4gpu::make_first_axis_contiguous_view<b4gpu::Complex64, 4>(
                B5.data(), {d.k5, d.l25, d.l45, d.l35})};
    }
};

enum class CoherentBoosterOutputTarget : unsigned {
    support_packed = 1u,
    q_slab_native = 2u,
    support_and_q_slab_native = 3u
};

inline bool coherent_booster_wants_support_packed(
    CoherentBoosterOutputTarget target) noexcept
{
    return (static_cast<unsigned>(target) & 1u) != 0u;
}

inline bool coherent_booster_wants_q_slab_native(
    CoherentBoosterOutputTarget target) noexcept
{
    return (static_cast<unsigned>(target) & 2u) != 0u;
}

struct CoherentBoosterProductionStats {
    bool direct_support_packed = false;
    std::uint64_t slab_budget_bytes = 0;
    std::uint64_t packed_payload_bytes = 0;
    std::uint64_t q_slab_native_payload_bytes = 0;
    std::uint64_t avoided_dense_payload_bytes = 0;
    double total_seconds = 0.0;
    B4HarvestThreadStats harvest_threads{};
};

} // namespace b4support


namespace b4support {

inline std::size_t flat_B2_canonical(
                                     std::size_t il23,
                                     std::size_t il24,
                                     std::size_t il25,
                                     std::size_t ik2,
                                     std::size_t L23,
                                     std::size_t L24,
                                     std::size_t L25) noexcept
{
    return il23 + L23 * (il24 + L24 * (il25 + L25 * ik2));
}

inline std::size_t flat_B3_canonical(
                                     std::size_t il34,
                                     std::size_t il35,
                                     std::size_t il23,
                                     std::size_t ik3,
                                     std::size_t L34,
                                     std::size_t L35,
                                     std::size_t L23) noexcept
{
    return il34 + L34 * (il35 + L35 * (il23 + L23 * ik3));
}

inline std::size_t flat_B4_canonical(
                                     std::size_t il45,
                                     std::size_t il24,
                                     std::size_t il34,
                                     std::size_t ik4,
                                     std::size_t L45,
                                     std::size_t L24,
                                     std::size_t L34) noexcept
{
    return il45 + L45 * (il24 + L24 * (il34 + L34 * ik4));
}

inline std::size_t flat_B5_canonical(
                                     std::size_t il25,
                                     std::size_t il35,
                                     std::size_t il45,
                                     std::size_t ik5,
                                     std::size_t L25,
                                     std::size_t L35,
                                     std::size_t L45) noexcept
{
    return il25 + L25 * (il35 + L35 * (il45 + L45 * ik5));
}

} // namespace b4support


namespace b4support {

void pack_B2_from_canonical(
                                   const AmplitudeInputSupportPlan& support,
                                   const Tensor<double, 4>& dense_B2,
                                   std::vector<double>& packed_B2);

void pack_B3_from_canonical(
                                   const AmplitudeInputSupportPlan& support,
                                   const Tensor<double, 4>& dense_B3,
                                   std::vector<double>& packed_B3);

void pack_B4_from_canonical(
                                   const AmplitudeInputSupportPlan& support,
                                   const Tensor<double, 4>& dense_B4,
                                   std::vector<double>& packed_B4);

void pack_B5_from_canonical(
                                   const AmplitudeInputSupportPlan& support,
                                   const Tensor<double, 4>& dense_B5,
                                   std::vector<double>& packed_B5);

} // namespace b4support

namespace b4support {

void fill_packed_amplitude_boosters_via_canonical(
                                                         const B4Spins& spins,
                                                         const std::array<std::size_t, 6>& Dim_ls,
                                                         double gamma,
                                                         const std::vector<double>& FactTable,
                                                         const AmplitudeInputSupportPlan& support,
                                                         PackedAmplitudeBoosters& packed,
                                                         sl2cfoamxx::SimplicityConstraint prescription =
                                                             sl2cfoamxx::SimplicityConstraint::gamma_j,
                                                         std::uint32_t rho_tile = 0);

} // namespace b4support

namespace b4support {

inline std::size_t flat_B2_contraction_native(
    std::size_t il23,
    std::size_t il25,
    std::size_t il24,
    std::size_t ik2,
    std::size_t L23,
    std::size_t L25,
    std::size_t L24) noexcept
{
    return il23 + L23 * (il25 + L25 * (il24 + L24 * ik2));
}

inline std::size_t flat_B3_contraction_native(
    std::size_t il23,
    std::size_t il34,
    std::size_t il35,
    std::size_t ik3,
    std::size_t L23,
    std::size_t L34,
    std::size_t L35) noexcept
{
    return il23 + L23 * (il34 + L34 * (il35 + L35 * ik3));
}

inline std::size_t flat_B4_contraction_native(
    std::size_t il45,
    std::size_t il24,
    std::size_t il34,
    std::size_t ik4,
    std::size_t L45,
    std::size_t L24,
    std::size_t L34) noexcept
{
    return il45 + L45 * (il24 + L24 * (il34 + L34 * ik4));
}

inline std::size_t flat_B5_contraction_native(
    std::size_t il45,
    std::size_t il25,
    std::size_t il35,
    std::size_t ik5,
    std::size_t L45,
    std::size_t L25,
    std::size_t L35) noexcept
{
    return il45 + L45 * (il25 + L25 * (il35 + L35 * ik5));
}


#ifndef B4_COHERENT_RANK5_SLAB_MAX_BYTES
#define B4_COHERENT_RANK5_SLAB_MAX_BYTES (2ull * 1024ull * 1024ull * 1024ull)
#endif

#ifndef B4_COHERENT_RANK5_SLAB_RAM_MARGIN_PERCENT
#define B4_COHERENT_RANK5_SLAB_RAM_MARGIN_PERCENT 60ull
#endif

inline std::uint64_t b4_env_u64_or_zero(const char* name) noexcept {
    const char* s = std::getenv(name);
    if (s == nullptr || *s == '\0') return 0;

    errno = 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (errno != 0 || end == s || v == 0ull) return 0;
    return static_cast<std::uint64_t>(v);
}

inline std::uint64_t b4_coherent_dense_native_payload_bytes(
    const AmplitudeInputSupportPlan& support) noexcept
{
    const std::uint64_t L23 = support.l23.size();
    const std::uint64_t L24 = support.l24.size();
    const std::uint64_t L25 = support.l25.size();
    const std::uint64_t L34 = support.l34.size();
    const std::uint64_t L35 = support.l35.size();
    const std::uint64_t L45 = support.l45.size();

    const std::uint64_t K2d = support.channels.k2.size();
    const std::uint64_t K3d = support.channels.k3.size();
    const std::uint64_t K4d = support.channels.k4.size();
    const std::uint64_t K5d = support.channels.k5.size();

    auto product4 = [](std::uint64_t a, std::uint64_t b,
                       std::uint64_t c, std::uint64_t d) noexcept {
        return host_memory::sat_mul(
            host_memory::sat_mul(a, b), host_memory::sat_mul(c, d));
    };
    std::uint64_t values = product4(L23, L25, L24, K2d);
    values = host_memory::sat_add(values, product4(L23, L34, L35, K3d));
    values = host_memory::sat_add(values, product4(L45, L24, L34, K4d));
    values = host_memory::sat_add(values, product4(L45, L25, L35, K5d));

    return host_memory::sat_mul(
        values, static_cast<std::uint64_t>(sizeof(b4utils::Complex64)));
}

inline std::uint64_t b4_coherent_packed_booster_payload_bytes(
    const AmplitudeInputSupportPlan& support) noexcept
{
    std::uint64_t values = host_memory::sat_add(
        support.B2.packed_values, support.B3.packed_values);
    values = host_memory::sat_add(values, support.B4.packed_values);
    values = host_memory::sat_add(values, support.B5.packed_values);
    return host_memory::sat_mul(
        values, static_cast<std::uint64_t>(sizeof(b4utils::Complex64)));
}

std::uint64_t b4_choose_coherent_rank5_slab_budget_bytes_impl(
    std::uint64_t future_output_bytes) noexcept;

inline std::uint64_t b4_choose_coherent_rank5_slab_budget_bytes(
    const AmplitudeInputSupportPlan& support) noexcept
{
    return b4_choose_coherent_rank5_slab_budget_bytes_impl(
        b4_coherent_packed_booster_payload_bytes(support) +
        b4_coherent_dense_native_payload_bytes(support));
}

inline std::uint64_t b4_choose_coherent_direct_rank5_slab_budget_bytes(
    const AmplitudeInputSupportPlan& support) noexcept
{
    return b4_choose_coherent_rank5_slab_budget_bytes_impl(
        b4_coherent_packed_booster_payload_bytes(support));
}

inline std::uint64_t b4_choose_coherent_direct_rank5_slab_budget_bytes(
    const AmplitudeInputSupportPlan& support,
    CoherentBoosterOutputTarget target) noexcept
{
    std::uint64_t outputs = 0;
    if (coherent_booster_wants_support_packed(target)) {
        outputs = host_memory::sat_add(
            outputs, b4_coherent_packed_booster_payload_bytes(support));
    }
    if (coherent_booster_wants_q_slab_native(target)) {
        outputs = host_memory::sat_add(
            outputs, b4_coherent_dense_native_payload_bytes(support));
    }
    return b4_choose_coherent_rank5_slab_budget_bytes_impl(outputs);
}

void pack_B2_complex_from_contraction_native(
    const AmplitudeInputSupportPlan& support,
    const Tensor<b4utils::Complex64, 4>& dense_B2,
    b4utils::ComplexBuffer<>& packed_B2);

void pack_B3_complex_from_contraction_native(
    const AmplitudeInputSupportPlan& support,
    const Tensor<b4utils::Complex64, 4>& dense_B3,
    b4utils::ComplexBuffer<>& packed_B3);

void pack_B4_complex_from_contraction_native(
    const AmplitudeInputSupportPlan& support,
    const Tensor<b4utils::Complex64, 4>& dense_B4,
    b4utils::ComplexBuffer<>& packed_B4);

void pack_B5_complex_from_contraction_native(
    const AmplitudeInputSupportPlan& support,
    const Tensor<b4utils::Complex64, 4>& dense_B5,
    b4utils::ComplexBuffer<>& packed_B5);

// ============================================================================
// Direct rank-5 slab -> support-packed coherent sinks
// ============================================================================
//
// The recurrence harvester naturally produces a real slab
//   canonical_l[3] x active_boundary x virtual_slab.
// These sinks perform the coherent boundary intertwiner dot product and write only values
// present in the contraction support.  No dense complex rank-4 tensor is
// materialized, and every packed value is owned by one (virtual intertwiner, l) loop tile, so
// the kernels need neither atomics nor thread-local output copies.

inline b4utils::Complex64 b4_coherent_boundary_dot(
    const double* __restrict slab,
    std::size_t base,
    std::size_t boundary_stride,
    const std::vector<b4utils::Complex64>& coeff) noexcept
{
    double re = 0.0;
    double im = 0.0;
    for (std::size_t ia = 0; ia < coeff.size(); ++ia) {
        const double v = slab[base + ia * boundary_stride];
        re += v * coeff[ia].re;
        im += v * coeff[ia].im;
    }
    return {re, im};
}

inline void b4_store_direct_packed(
    b4utils::ComplexBuffer<>& out,
    std::size_t offset,
    b4utils::Complex64 z) noexcept
{
    out.re_data()[offset] = z.re;
    out.im_data()[offset] = z.im;
}

void b4_contract_rank5_slab_to_packed_B2(
    const AmplitudeInputSupportPlan& support,
    std::size_t b_global_begin,
    const std::vector<b4utils::Complex64>& coeff,
    const Tensor<double, 5>& slab,
    b4utils::ComplexBuffer<>* out,
    CoherentQSlabNativeBoosters* native);

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
    int Dj3 = 0,
    int Dj4 = 0,
    int Dl3_base = 0,
    int Dl4_fixed = 0);

void b4_contract_rank5_slab_to_packed_B4(
    const AmplitudeInputSupportPlan& support,
    std::size_t b_global_begin,
    std::span<const int> a_vals,
    std::span<const int> b_vals,
    const std::vector<b4utils::Complex64>& coeff,
    const Tensor<double, 5>& slab,
    b4utils::ComplexBuffer<>* out,
    CoherentQSlabNativeBoosters* native,
    bool pair_exchange_120);

enum class B4PackedB5SlabTransform {
    direct,
    pair_exchange_120,
    exchange34_then_pair_120
};

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
    int Dj3 = 0,
    int Dj4 = 0,
    int Dl3_base = 0,
    int Dl4_fixed = 0);

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
    std::uint64_t slab_target_bytes = 0,
    CoherentBoosterProductionStats* stats = nullptr,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0);

inline void fill_packed_amplitude_coherent_boosters_direct(
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
    std::uint64_t slab_target_bytes = 0,
    CoherentBoosterProductionStats* stats = nullptr,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0)
{
    fill_amplitude_coherent_boosters_direct_outputs(
        spins, Dim_ls, gamma, FactTable, support,
        C2, C3, C4, C5,
        CoherentBoosterOutputTarget::support_packed,
        &packed, nullptr, slab_target_bytes, stats, prescription, rho_tile);
}

inline void fill_q_slab_native_amplitude_coherent_boosters_direct(
    const B4Spins& spins,
    const std::array<std::size_t, 6>& Dim_ls,
    double gamma,
    const std::vector<double>& FactTable,
    const AmplitudeInputSupportPlan& support,
    const B4CoherentCoefficientsView& C2,
    const B4CoherentCoefficientsView& C3,
    const B4CoherentCoefficientsView& C4,
    const B4CoherentCoefficientsView& C5,
    CoherentQSlabNativeBoosters& q_slab_native,
    std::uint64_t slab_target_bytes = 0,
    CoherentBoosterProductionStats* stats = nullptr,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0)
{
    fill_amplitude_coherent_boosters_direct_outputs(
        spins, Dim_ls, gamma, FactTable, support,
        C2, C3, C4, C5,
        CoherentBoosterOutputTarget::q_slab_native,
        nullptr, &q_slab_native, slab_target_bytes, stats, prescription, rho_tile);
}

inline void fill_packed_and_q_slab_native_amplitude_coherent_boosters_direct(
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
    CoherentQSlabNativeBoosters& q_slab_native,
    std::uint64_t slab_target_bytes = 0,
    CoherentBoosterProductionStats* stats = nullptr,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0)
{
    fill_amplitude_coherent_boosters_direct_outputs(
        spins, Dim_ls, gamma, FactTable, support,
        C2, C3, C4, C5,
        CoherentBoosterOutputTarget::support_and_q_slab_native,
        &packed, &q_slab_native, slab_target_bytes, stats, prescription, rho_tile);
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
    std::uint64_t slab_target_bytes = 0,
    CoherentBoosterProductionStats* stats = nullptr,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0);

inline void fill_packed_amplitude_coherent_boosters_via_canonical(
    const B4Spins& spins,
    const std::array<std::size_t, 6>& Dim_ls,
    double gamma,
    const std::vector<double>& FactTable,
    const AmplitudeInputSupportPlan& support,
    const std::vector<int>& C2_iotas,
    const std::vector<b4utils::Complex64>& C2_coeffs,
    const std::vector<int>& C3_iotas,
    const std::vector<b4utils::Complex64>& C3_coeffs,
    const std::vector<int>& C4_iotas,
    const std::vector<b4utils::Complex64>& C4_coeffs,
    const std::vector<int>& C5_iotas,
    const std::vector<b4utils::Complex64>& C5_coeffs,
    PackedAmplitudeCoherentBoosters& packed,
    std::uint64_t slab_target_bytes = 0,
    CoherentBoosterProductionStats* stats = nullptr,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0)
{
    fill_packed_amplitude_coherent_boosters_via_canonical(
        spins,
        Dim_ls,
        gamma,
        FactTable,
        support,
        make_b4_coherent_coefficients_view(C2_iotas, C2_coeffs),
        make_b4_coherent_coefficients_view(C3_iotas, C3_coeffs),
        make_b4_coherent_coefficients_view(C4_iotas, C4_coeffs),
        make_b4_coherent_coefficients_view(C5_iotas, C5_coeffs),
        packed,
        slab_target_bytes,
        stats,
        prescription,
        rho_tile);
}

inline void fill_packed_amplitude_coherent_boosters_direct(
    const B4Spins& spins,
    const std::array<std::size_t, 6>& Dim_ls,
    double gamma,
    const std::vector<double>& FactTable,
    const AmplitudeInputSupportPlan& support,
    const std::vector<int>& C2_iotas,
    const std::vector<b4utils::Complex64>& C2_coeffs,
    const std::vector<int>& C3_iotas,
    const std::vector<b4utils::Complex64>& C3_coeffs,
    const std::vector<int>& C4_iotas,
    const std::vector<b4utils::Complex64>& C4_coeffs,
    const std::vector<int>& C5_iotas,
    const std::vector<b4utils::Complex64>& C5_coeffs,
    PackedAmplitudeCoherentBoosters& packed,
    std::uint64_t slab_target_bytes = 0,
    CoherentBoosterProductionStats* stats = nullptr,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0)
{
    fill_packed_amplitude_coherent_boosters_direct(
        spins,
        Dim_ls,
        gamma,
        FactTable,
        support,
        make_b4_coherent_coefficients_view(C2_iotas, C2_coeffs),
        make_b4_coherent_coefficients_view(C3_iotas, C3_coeffs),
        make_b4_coherent_coefficients_view(C4_iotas, C4_coeffs),
        make_b4_coherent_coefficients_view(C5_iotas, C5_coeffs),
        packed,
        slab_target_bytes,
        stats,
        prescription,
        rho_tile);
}

} // namespace b4support
