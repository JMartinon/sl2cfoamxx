// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "full_tensor/FullTensorInputSupport_HalfSjC.hpp"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>

namespace b4support {

// Lightweight owning/view containers shared by the host producers and the
// CUDA-native consumer.  Keep these definitions separate from recurrence and
// dense-booster filling code so nvcc (including CUDA 11.x) does not have to
// parse the complete host implementation merely to upload packed buffers.
struct PackedFullTensorBoosters {
    std::vector<double> B2;
    std::vector<double> B3;
    std::vector<double> B4;
    std::vector<double> B5;

    const double* B2_alias = nullptr;
    const double* B3_alias = nullptr;
    const double* B4_alias = nullptr;
    const double* B5_alias = nullptr;

    const double* B2_data() const noexcept { return B2_alias ? B2_alias : B2.data(); }
    const double* B3_data() const noexcept { return B3_alias ? B3_alias : B3.data(); }
    const double* B4_data() const noexcept { return B4_alias ? B4_alias : B4.data(); }
    const double* B5_data() const noexcept { return B5_alias ? B5_alias : B5.data(); }

    void allocate_from_support(const FullTensorInputSupportPlan& support) {
        B2_alias = B3_alias = B4_alias = B5_alias = nullptr;
        B2.resize(static_cast<std::size_t>(support.B2.packed_values));
        B3.resize(static_cast<std::size_t>(support.B3.packed_values));
        B4.resize(static_cast<std::size_t>(support.B4.packed_values));
        B5.resize(static_cast<std::size_t>(support.B5.packed_values));
    }

    void clear() {
        B2_alias = B3_alias = B4_alias = B5_alias = nullptr;
        B2.clear(); B3.clear(); B4.clear(); B5.clear();
    }

    std::uint64_t payload_bytes() const noexcept {
        return static_cast<std::uint64_t>(
            (B2.size() + B3.size() + B4.size() + B5.size()) * sizeof(double));
    }

    void print(std::ostream& os) const {
        const double mib = 1024.0 * 1024.0;
        os << "\nPacked full-tensor booster tensors\n"
           << "==================================\n"
           << "B2 values = " << B2.size() << " | "
           << static_cast<double>(B2.size() * sizeof(double)) / mib << " MiB\n"
           << "B3 values = " << B3.size() << " | "
           << static_cast<double>(B3.size() * sizeof(double)) / mib << " MiB\n"
           << "B4 values = " << B4.size() << " | "
           << static_cast<double>(B4.size() * sizeof(double)) / mib << " MiB\n"
           << "B5 values = " << B5.size() << " | "
           << static_cast<double>(B5.size() * sizeof(double)) / mib << " MiB\n"
           << "total     = " << static_cast<double>(payload_bytes()) / mib
           << " MiB\n";
    }
};

struct PackedFullTensorSJTensors {
    std::vector<double> sjA;
    std::vector<double> sjB;
    std::vector<double> sjC;
    std::vector<double> sjD;
    std::vector<double> sjE;

    void allocate_from_support(const FullTensorInputSupportPlan& support) {
        sjA.resize(static_cast<std::size_t>(support.sjA.packed_values));
        sjB.resize(static_cast<std::size_t>(support.sjB.packed_values));
        sjC.resize(static_cast<std::size_t>(support.sjC.packed_values));
        sjD.resize(static_cast<std::size_t>(support.sjD.packed_values));
        sjE.resize(static_cast<std::size_t>(support.sjE.packed_values));
    }

    void clear() {
        sjA.clear(); sjB.clear(); sjC.clear(); sjD.clear(); sjE.clear();
    }

    std::uint64_t payload_bytes() const noexcept {
        return static_cast<std::uint64_t>(
            (sjA.size() + sjB.size() + sjC.size() + sjD.size() + sjE.size()) *
            sizeof(double));
    }

    void print(std::ostream& os) const {
        const double mib = 1024.0 * 1024.0;
        os << "\nPacked full-tensor sj tensors\n"
           << "=============================\n"
           << "sjA values = " << sjA.size() << " | "
           << static_cast<double>(sjA.size() * sizeof(double)) / mib << " MiB\n"
           << "sjB values = " << sjB.size() << " | "
           << static_cast<double>(sjB.size() * sizeof(double)) / mib << " MiB\n"
           << "sjC values = " << sjC.size() << " | "
           << static_cast<double>(sjC.size() * sizeof(double)) / mib
           << " MiB (half-stored logical sjC)\n"
           << "sjD values = " << sjD.size() << " | "
           << static_cast<double>(sjD.size() * sizeof(double)) / mib << " MiB\n"
           << "sjE values = " << sjE.size() << " | "
           << static_cast<double>(sjE.size() * sizeof(double)) / mib << " MiB\n"
           << "total      = " << static_cast<double>(payload_bytes()) / mib
           << " MiB\n";
    }
};

} // namespace b4support
