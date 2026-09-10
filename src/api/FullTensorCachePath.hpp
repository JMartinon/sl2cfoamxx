// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/configuration.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace sl2cfoamxx::detail {

[[nodiscard]] inline const char* full_tensor_cache_simplicity_component(
    SimplicityConstraint prescription) noexcept
{
    return prescription == SimplicityConstraint::gamma_j_p1
        ? "gamma_j_p1"
        : "gamma_j";
}

[[nodiscard]] inline std::string full_tensor_cache_gamma_component(
    double gamma)
{
    std::ostringstream name;
    name.imbue(std::locale::classic());
    name << "gamma_" << std::fixed << std::setprecision(3) << gamma;
    std::string component = name.str();
    for (char& character : component) {
        if (character == '+') character = 'p';
        if (character == '-') character = 'm';
    }
    return component;
}

[[nodiscard]] inline std::string full_tensor_cache_spins_component(
    const std::array<int, 10>& spins)
{
    std::ostringstream name;
    name << "Dspins";
    for (const int spin : spins) name << '_' << spin;
    return name.str();
}

// This is the one authority for the automatic public tensor-cache tree. The
// suffix records the requested physical mode; metadata in the file remains the
// authority on the representation actually produced.
[[nodiscard]] inline std::filesystem::path automatic_full_tensor_cache_path(
    const LibraryConfig& config,
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma,
    bool symmetry_name)
{
    return config.tensors.directory /
           "full_tensor" /
           full_tensor_cache_simplicity_component(
               config.simplicity_constraint) /
           full_tensor_cache_gamma_component(gamma) /
           full_tensor_cache_spins_component(spins) /
           ("Dl_" + std::to_string(Dl) +
            (symmetry_name ? "_sym.sl2cxx" : ".sl2cxx"));
}

// Per-call amplitude reuse probes an explicit path only when one is configured.
// Otherwise it checks both automatic names, preferring the currently selected
// storage family without making that preference a compatibility requirement.
[[nodiscard]] inline std::vector<std::filesystem::path>
full_tensor_cache_candidate_paths(
    const LibraryConfig& config,
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma)
{
    if (!config.tensors.output_path.empty()) {
        return {config.tensors.output_path};
    }

    const bool prefer_symmetry_name =
        config.tensors.storage == TensorStorage::in_memory_sym ||
        config.tensors.storage == TensorStorage::on_disk_sym;
    return {
        automatic_full_tensor_cache_path(
            config, spins, Dl, gamma, prefer_symmetry_name),
        automatic_full_tensor_cache_path(
            config, spins, Dl, gamma, !prefer_symmetry_name)};
}

} // namespace sl2cfoamxx::detail
