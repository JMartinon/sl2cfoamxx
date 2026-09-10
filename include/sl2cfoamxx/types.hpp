// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <ostream>
#include <string_view>

namespace sl2cfoamxx {

// Public angular-momentum labels use doubled integer units. DSpins follow
// {2*j12,2*j13,2*j14,2*j15,2*j23,2*j24,2*j25,2*j34,2*j35,2*j45};
// DIntertwiners follow canonical axes {2*i1,2*i2,2*i3,2*i4,2*i5}.
using DSpins = std::array<int, 10>;
using DIntertwiners = std::array<int, 5>;
using TetrahedronDSpins = std::array<int, 4>;

enum class Backend {
    CPU = 0,
    GPU = 1,

    // Source-compatible spellings retained for 0.7 clients. New code should
    // use the conventional uppercase acronym.
    cpu [[deprecated("use Backend::CPU")]] = CPU,
    gpu [[deprecated("use Backend::GPU")]] = GPU
};

// Full-tensor representation and backing. The symmetry-aware modes retain the
// compact reassociated output, while the dense modes materialize canonical
// (i1,i2,i3,i4,i5) storage.
enum class TensorStorage : std::uint8_t {
    in_memory_sym,
    in_memory_dense,
    on_disk_sym,
    on_disk_dense,

    // Source-compatible spellings for the original dense-only public API.
    memory [[deprecated("use TensorStorage::in_memory_dense")]] =
        in_memory_dense,
    mapped_file [[deprecated("use TensorStorage::on_disk_dense")]] =
        on_disk_dense
};

// Choice of EPRL simplicity map used when constructing booster functions.
// Public spin labels are doubled, hence j = Dspin / 2.
enum class SimplicityConstraint : std::uint8_t {
    gamma_j = 0,
    gamma_j_p1 = 1
};

// Optional source for a fixed-intertwiner amplitude.  The normal overload
// computes the amplitude directly.  prefer_full_tensor first checks for a
// compatible on-disk full tensor and falls back to that same direct path when
// no valid cache file is available.
enum class AmplitudeCachePolicy : std::uint8_t {
    disabled = 0,
    prefer_full_tensor = 1
};

[[nodiscard]] constexpr std::string_view to_string(Backend backend) noexcept
{
    switch (backend) {
    case Backend::CPU:
        return "CPU";
    case Backend::GPU:
        return "GPU";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(
    TensorStorage storage) noexcept
{
    switch (storage) {
    case TensorStorage::in_memory_sym:
        return "in_memory_sym";
    case TensorStorage::in_memory_dense:
        return "in_memory_dense";
    case TensorStorage::on_disk_sym:
        return "on_disk_sym";
    case TensorStorage::on_disk_dense:
        return "on_disk_dense";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(
    SimplicityConstraint prescription) noexcept
{
    switch (prescription) {
    case SimplicityConstraint::gamma_j:
        return "gamma_j";
    case SimplicityConstraint::gamma_j_p1:
        return "gamma_j_p1";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(
    AmplitudeCachePolicy policy) noexcept
{
    switch (policy) {
    case AmplitudeCachePolicy::disabled:
        return "disabled";
    case AmplitudeCachePolicy::prefer_full_tensor:
        return "prefer_full_tensor";
    }
    return "unknown";
}

inline std::ostream& operator<<(std::ostream& output, Backend backend)
{
    return output << to_string(backend);
}

inline std::ostream& operator<<(
    std::ostream& output,
    TensorStorage storage)
{
    return output << to_string(storage);
}

inline std::ostream& operator<<(
    std::ostream& output,
    SimplicityConstraint prescription)
{
    return output << to_string(prescription);
}

inline std::ostream& operator<<(
    std::ostream& output,
    AmplitudeCachePolicy policy)
{
    return output << to_string(policy);
}

} // namespace sl2cfoamxx
