// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/sl2cfoamxx.hpp"

namespace sl2cfoamxx::detail {

// Immutable internal copy of the relevant LibraryConfig fields. This prevents
// a concurrent configure() call from changing an active computation.
struct FullTensorConfigSnapshot {
    LibraryConfig config{};
    Backend backend = Backend::CPU;
    TensorStorage storage = TensorStorage::in_memory_sym;
    std::filesystem::path output_path{};
    bool reuse_existing = true;
    std::uint64_t memory_budget_bytes = 0;
    std::uint64_t maximum_output_bytes = 0;
};

[[nodiscard]] FullTensor compute_full_tensor_cpu(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    const FullTensorConfigSnapshot& config);

[[nodiscard]] FullTensor compute_full_tensor_gpu(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    const FullTensorConfigSnapshot& config);

} // namespace sl2cfoamxx::detail
