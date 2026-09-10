// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "gpu/CuTensorCommon.hpp"
#include "gpu/CuTensorCompatibility.hpp"
#include "gpu/FullTensorSemiFusedGpuApi.hpp"
#include "gpu/FullTensorSemiFusedGpuTypes.cuh"

#include <cutensor.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace b4gpu::semifused_cutensor_detail {

using Clock = std::chrono::steady_clock;
using cutensor_detail::algorithm_name;
using cutensor_detail::check;
using cutensor_detail::planning_mutex;
using semifused_detail::SemiFusedDims;

inline const char* jit_name(cutensorJitMode_t jit) noexcept
{
    return jit == CUTENSOR_JIT_MODE_NONE ? "nojit" : "jit";
}

inline const char* workspace_name(
    cutensorWorksizePreference_t preference) noexcept
{
    switch (preference) {
    case CUTENSOR_WORKSPACE_MIN: return "min";
    case CUTENSOR_WORKSPACE_DEFAULT: return "default";
    case CUTENSOR_WORKSPACE_MAX: return "max";
    default: return "unknown";
    }
}

struct PlanMetrics {
    std::uint64_t i1_count = 0;
    std::uint64_t workspace_estimate = 0;
    std::uint64_t workspace_limit = 0;
    std::uint64_t workspace_actual = 0;
    std::uint64_t workspace_default_estimate = 0;
    std::uint64_t workspace_min_estimate = 0;
    // cuTENSOR's descriptor model. These are per invocation and describe the
    // logical binary contraction, not hardware counters or padded instructions.
    float operation_flops = 0.0f;
    float operation_moved_bytes = 0.0f;
    double creation_seconds = 0.0;
    double serialization_wait_seconds = 0.0;
    std::uint32_t attempts = 0;
    std::uint32_t failed_attempts = 0;
    bool fallback_used = false;
    bool selected_jit = false;
    bool selected_min_workspace = false;
    std::int32_t selected_algorithm = static_cast<std::int32_t>(
        CUTENSOR_ALGO_DEFAULT);
    std::string selected_summary;
};

class Context {
public:
    explicit Context(bool enable_jit)
        : requested_jit_(enable_jit)
    {
        const auto begin = Clock::now();
        // Handle creation is cheap and serialized with planning to avoid
        // entering process-global JIT/cache setup from two GPU worker threads.
        std::lock_guard<std::mutex> lock(planning_mutex());
        check(cutensorCreate(&handle_), "cutensorCreate");
        creation_seconds_ = std::chrono::duration<double>(
            Clock::now() - begin).count();
    }

    ~Context()
    {
        if (handle_ != nullptr) {
            // Destruction is documented thread-safe, but serializing it with
            // JIT planning avoids exercising a second vendor-library concurrency
            // path while another handle is being configured.
            std::lock_guard<std::mutex> lock(planning_mutex());
            (void)cutensorDestroy(handle_);
        }
    }

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    [[nodiscard]] cutensorHandle_t handle() const noexcept { return handle_; }
    [[nodiscard]] double creation_seconds() const noexcept
    {
        return creation_seconds_;
    }
    [[nodiscard]] bool requested_jit() const noexcept
    {
        return requested_jit_;
    }

private:
    cutensorHandle_t handle_ = nullptr;
    bool requested_jit_ = false;
    double creation_seconds_ = 0.0;
};

// Explicitly form the beta-fast pointwise products
//
//   W1[v,t,f,j,p,s] = sjB[v,j,s,t] * B5[v,t,f,p,s]
//   W3[k,u,e,i,p,q] = sjD[k,i,u,q] * B2[k,u,e,p,q]
//
// and submit the mature binary contractions
//
//   T1[t,f,j,p,s,a] = sum_v W1[v,t,f,j,p,s] * sjA[v,a,p]
//   T3[u,e,i,p,q,a] = sum_k W3[k,u,e,i,p,q] * sjE[k,a,p].
class BinaryPlan {
public:
    BinaryPlan() = default;
    ~BinaryPlan() { reset(); }

    BinaryPlan(const BinaryPlan&) = delete;
    BinaryPlan& operator=(const BinaryPlan&) = delete;

    BinaryPlan(BinaryPlan&& other) noexcept { move_from(other); }
    BinaryPlan& operator=(BinaryPlan&& other) noexcept
    {
        if (this != &other) {
            reset();
            move_from(other);
        }
        return *this;
    }

    [[nodiscard]] static BinaryPlan make_t1(
        const Context& context,
        const SemiFusedDims& d,
        std::uint64_t i1_count,
        std::uint64_t workspace_cap)
    {
        // A[v,a,p] * W1[v,t,f,j,p,s]
        //        -> T1[t,f,j,p,s,a], summing v.
        const TensorSpec A{{
            static_cast<std::int64_t>(d.K5),
            static_cast<std::int64_t>(i1_count),
            static_cast<std::int64_t>(d.P)}, {
            1,
            static_cast<std::int64_t>(d.K5),
            static_cast<std::int64_t>(d.K5 * i1_count)},
            {'v','a','p'}};
        const TensorSpec B{{
            static_cast<std::int64_t>(d.K5),
            static_cast<std::int64_t>(d.T),
            static_cast<std::int64_t>(d.I5),
            static_cast<std::int64_t>(d.K4),
            static_cast<std::int64_t>(d.P),
            static_cast<std::int64_t>(d.S)}, {
            1,
            static_cast<std::int64_t>(d.K5),
            static_cast<std::int64_t>(d.K5 * d.T),
            static_cast<std::int64_t>(d.K5 * d.T * d.I5),
            static_cast<std::int64_t>(d.K5 * d.T * d.I5 * d.K4),
            static_cast<std::int64_t>(
                d.K5 * d.T * d.I5 * d.K4 * d.P)},
            {'v','t','f','j','p','s'}};
        const TensorSpec D{{
            static_cast<std::int64_t>(d.T),
            static_cast<std::int64_t>(d.I5),
            static_cast<std::int64_t>(d.K4),
            static_cast<std::int64_t>(d.P),
            static_cast<std::int64_t>(d.S),
            static_cast<std::int64_t>(i1_count)}, {
            1,
            static_cast<std::int64_t>(d.T),
            static_cast<std::int64_t>(d.T * d.I5),
            static_cast<std::int64_t>(d.T * d.I5 * d.K4),
            static_cast<std::int64_t>(d.T * d.I5 * d.K4 * d.P),
            static_cast<std::int64_t>(d.T1Logical)},
            {'t','f','j','p','s','a'}};
        return make(context, A, B, D, "T1", i1_count, workspace_cap);
    }

    [[nodiscard]] static BinaryPlan make_t3(
        const Context& context,
        const SemiFusedDims& d,
        std::uint64_t i1_count,
        std::uint64_t workspace_cap)
    {
        // A[k,a,p] * W3[k,u,e,i,p,q]
        //        -> T3[u,e,i,p,q,a], summing k.
        const TensorSpec A{{
            static_cast<std::int64_t>(d.K2),
            static_cast<std::int64_t>(i1_count),
            static_cast<std::int64_t>(d.P)}, {
            1,
            static_cast<std::int64_t>(d.K2),
            static_cast<std::int64_t>(d.K2 * i1_count)},
            {'k','a','p'}};
        const TensorSpec B{{
            static_cast<std::int64_t>(d.K2),
            static_cast<std::int64_t>(d.U),
            static_cast<std::int64_t>(d.I2),
            static_cast<std::int64_t>(d.K3),
            static_cast<std::int64_t>(d.P),
            static_cast<std::int64_t>(d.Q)}, {
            1,
            static_cast<std::int64_t>(d.K2),
            static_cast<std::int64_t>(d.K2 * d.U),
            static_cast<std::int64_t>(d.K2 * d.U * d.I2),
            static_cast<std::int64_t>(d.K2 * d.U * d.I2 * d.K3),
            static_cast<std::int64_t>(
                d.K2 * d.U * d.I2 * d.K3 * d.P)},
            {'k','u','e','i','p','q'}};
        const TensorSpec D{{
            static_cast<std::int64_t>(d.U),
            static_cast<std::int64_t>(d.I2),
            static_cast<std::int64_t>(d.K3),
            static_cast<std::int64_t>(d.P),
            static_cast<std::int64_t>(d.Q),
            static_cast<std::int64_t>(i1_count)}, {
            1,
            static_cast<std::int64_t>(d.U),
            static_cast<std::int64_t>(d.U * d.I2),
            static_cast<std::int64_t>(d.U * d.I2 * d.K3),
            static_cast<std::int64_t>(d.U * d.I2 * d.K3 * d.P),
            static_cast<std::int64_t>(d.T3Logical)},
            {'u','e','i','p','q','a'}};
        return make(context, A, B, D, "T3", i1_count, workspace_cap);
    }

    [[nodiscard]] double execute(
        const Context& context,
        const double* A,
        const double* B,
        double* output,
        void* workspace,
        cudaStream_t stream) const
    {
        constexpr double alpha = 1.0;
        constexpr double beta = 0.0;
        const auto begin = Clock::now();
        check(cutensorContract(
                  context.handle(), plan_, &alpha, A, B, &beta,
                  output, output, workspace, metrics_.workspace_actual, stream),
              "cutensorContract");
        return std::chrono::duration<double>(Clock::now() - begin).count();
    }

    [[nodiscard]] const PlanMetrics& metrics() const noexcept
    {
        return metrics_;
    }

private:
    struct TensorSpec {
        std::vector<std::int64_t> extents;
        std::vector<std::int64_t> strides;
        std::vector<std::int32_t> modes;
    };

    struct Attempt {
        cutensorAlgo_t algorithm;
        cutensorJitMode_t jit;
        cutensorWorksizePreference_t workspace;
    };

    static void append_attempt(
        std::vector<Attempt>& attempts,
        cutensorAlgo_t algorithm,
        cutensorJitMode_t jit,
        cutensorWorksizePreference_t workspace)
    {
        const auto duplicate = std::find_if(
            attempts.begin(), attempts.end(), [&](const Attempt& existing) {
                return existing.algorithm == algorithm &&
                    existing.jit == jit && existing.workspace == workspace;
            });
        if (duplicate == attempts.end()) {
            attempts.push_back(Attempt{algorithm, jit, workspace});
        }
    }

    [[nodiscard]] static std::string describe_spec(const TensorSpec& spec)
    {
        std::ostringstream out;
        out << '[';
        for (std::size_t i = 0; i < spec.extents.size(); ++i) {
            if (i != 0) out << ',';
            out << static_cast<char>(spec.modes[i]) << '=' << spec.extents[i]
                << "@" << spec.strides[i];
        }
        out << ']';
        return out.str();
    }

    [[nodiscard]] static cutensorTensorDescriptor_t create_descriptor(
        cutensorHandle_t handle, const TensorSpec& spec)
    {
        if (spec.extents.size() != spec.strides.size() ||
            spec.extents.size() != spec.modes.size()) {
            throw std::logic_error("invalid cuTENSOR tensor specification");
        }
        cutensorTensorDescriptor_t descriptor = nullptr;
        check(cutensorCreateTensorDescriptor(
                  handle, &descriptor,
                  static_cast<std::uint32_t>(spec.extents.size()),
                  spec.extents.data(), spec.strides.data(),
                  cutensor_compat::kReal64, 256u),
              "cutensorCreateTensorDescriptor");
        return descriptor;
    }

    [[nodiscard]] static BinaryPlan make(
        const Context& context,
        const TensorSpec& A,
        const TensorSpec& B,
        const TensorSpec& D,
        const char* family,
        std::uint64_t i1_count,
        std::uint64_t workspace_cap)
    {
        BinaryPlan result;
        const auto total_begin = Clock::now();
        const auto wait_begin = Clock::now();
        std::unique_lock<std::mutex> planning_lock(planning_mutex());
        result.metrics_.serialization_wait_seconds =
            std::chrono::duration<double>(Clock::now() - wait_begin).count();

        std::ostringstream failures;
        try {
            result.desc_a_ = create_descriptor(context.handle(), A);
            result.desc_b_ = create_descriptor(context.handle(), B);
            result.desc_d_ = create_descriptor(context.handle(), D);
            check(cutensorCreateContraction(
                      context.handle(), &result.operation_,
                      result.desc_a_, A.modes.data(), CUTENSOR_OP_IDENTITY,
                      result.desc_b_, B.modes.data(), CUTENSOR_OP_IDENTITY,
                      result.desc_d_, D.modes.data(), CUTENSOR_OP_IDENTITY,
                      result.desc_d_, D.modes.data(),
                      CUTENSOR_COMPUTE_DESC_64F),
                  "cutensorCreateContraction");
            check(cutensorOperationDescriptorGetAttribute(
                      context.handle(), result.operation_,
                      CUTENSOR_OPERATION_DESCRIPTOR_FLOPS,
                      &result.metrics_.operation_flops,
                      sizeof(result.metrics_.operation_flops)),
                  "cutensorOperationDescriptorGetAttribute(FLOPS)");
            check(cutensorOperationDescriptorGetAttribute(
                      context.handle(), result.operation_,
                      CUTENSOR_OPERATION_DESCRIPTOR_MOVED_BYTES,
                      &result.metrics_.operation_moved_bytes,
                      sizeof(result.metrics_.operation_moved_bytes)),
                  "cutensorOperationDescriptorGetAttribute(MOVED_BYTES)");

            const cutensorJitMode_t requested_jit = context.requested_jit()
                ? CUTENSOR_JIT_MODE_DEFAULT : CUTENSOR_JIT_MODE_NONE;

            std::vector<Attempt> attempts;
            // First honor the user's request.  If its DEFAULT workspace does
            // not fit, use cuTENSOR's MIN estimate instead of passing an
            // arbitrary clipped limit to cutensorCreatePlan.  The latter is
            // what triggered the medium-shape INTERNAL_ERROR on the cluster.
            append_attempt(attempts, CUTENSOR_ALGO_DEFAULT, requested_jit,
                           CUTENSOR_WORKSPACE_DEFAULT);
            append_attempt(attempts, CUTENSOR_ALGO_DEFAULT, requested_jit,
                           CUTENSOR_WORKSPACE_MIN);
            // JIT is an optional performance choice, not a correctness
            // requirement. Fall back progressively.
            append_attempt(attempts, CUTENSOR_ALGO_DEFAULT,
                           CUTENSOR_JIT_MODE_NONE,
                           CUTENSOR_WORKSPACE_DEFAULT);
            append_attempt(attempts, CUTENSOR_ALGO_DEFAULT,
                           CUTENSOR_JIT_MODE_NONE,
                           CUTENSOR_WORKSPACE_MIN);
            // Last-resort fixed GEMM-like algorithms. Unsupported candidates
            // are simply recorded and skipped.
            append_attempt(attempts, CUTENSOR_ALGO_GETT,
                           CUTENSOR_JIT_MODE_NONE,
                           CUTENSOR_WORKSPACE_MIN);
            append_attempt(attempts, CUTENSOR_ALGO_TGETT,
                           CUTENSOR_JIT_MODE_NONE,
                           CUTENSOR_WORKSPACE_MIN);
            append_attempt(attempts, CUTENSOR_ALGO_TTGT,
                           CUTENSOR_JIT_MODE_NONE,
                           CUTENSOR_WORKSPACE_MIN);

            for (std::size_t index = 0; index < attempts.size(); ++index) {
                const Attempt& attempt = attempts[index];
                ++result.metrics_.attempts;
                cutensorPlanPreference_t preference = nullptr;
                cutensorStatus_t status = cutensorCreatePlanPreference(
                    context.handle(), &preference,
                    attempt.algorithm, attempt.jit);
                if (status != CUTENSOR_STATUS_SUCCESS) {
                    ++result.metrics_.failed_attempts;
                    failures << "  " << algorithm_name(attempt.algorithm)
                             << '/' << jit_name(attempt.jit) << '/'
                             << workspace_name(attempt.workspace)
                             << ": preference=" << cutensorGetErrorString(status)
                             << '\n';
                    continue;
                }

                std::uint64_t estimate = 0;
                status = cutensorEstimateWorkspaceSize(
                    context.handle(), result.operation_, preference,
                    attempt.workspace, &estimate);
                if (attempt.workspace == CUTENSOR_WORKSPACE_DEFAULT) {
                    result.metrics_.workspace_default_estimate = std::max(
                        result.metrics_.workspace_default_estimate, estimate);
                } else if (attempt.workspace == CUTENSOR_WORKSPACE_MIN) {
                    result.metrics_.workspace_min_estimate = std::max(
                        result.metrics_.workspace_min_estimate, estimate);
                }
                if (status != CUTENSOR_STATUS_SUCCESS) {
                    ++result.metrics_.failed_attempts;
                    failures << "  " << algorithm_name(attempt.algorithm)
                             << '/' << jit_name(attempt.jit) << '/'
                             << workspace_name(attempt.workspace)
                             << ": estimate=" << cutensorGetErrorString(status)
                             << '\n';
                    (void)cutensorDestroyPlanPreference(preference);
                    continue;
                }
                if (estimate > workspace_cap) {
                    ++result.metrics_.failed_attempts;
                    failures << "  " << algorithm_name(attempt.algorithm)
                             << '/' << jit_name(attempt.jit) << '/'
                             << workspace_name(attempt.workspace)
                             << ": estimate=" << estimate
                             << " exceeds cap=" << workspace_cap << '\n';
                    (void)cutensorDestroyPlanPreference(preference);
                    continue;
                }

                cutensorPlan_t candidate = nullptr;
                status = cutensorCreatePlan(
                    context.handle(), &candidate, result.operation_,
                    preference, estimate);
                if (status != CUTENSOR_STATUS_SUCCESS) {
                    ++result.metrics_.failed_attempts;
                    failures << "  " << algorithm_name(attempt.algorithm)
                             << '/' << jit_name(attempt.jit) << '/'
                             << workspace_name(attempt.workspace)
                             << ": create-plan="
                             << cutensorGetErrorString(status)
                             << " estimate=" << estimate << '\n';
                    if (candidate != nullptr) (void)cutensorDestroyPlan(candidate);
                    (void)cutensorDestroyPlanPreference(preference);
                    continue;
                }

                std::uint64_t actual = 0;
                status = cutensorPlanGetAttribute(
                    context.handle(), candidate,
                    CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                    &actual, sizeof(actual));
                if (status != CUTENSOR_STATUS_SUCCESS || actual > estimate ||
                    actual > workspace_cap) {
                    ++result.metrics_.failed_attempts;
                    failures << "  " << algorithm_name(attempt.algorithm)
                             << '/' << jit_name(attempt.jit) << '/'
                             << workspace_name(attempt.workspace)
                             << ": required-workspace=";
                    if (status == CUTENSOR_STATUS_SUCCESS) {
                        failures << actual << " (estimate=" << estimate
                                 << ", cap=" << workspace_cap << ')';
                    } else {
                        failures << cutensorGetErrorString(status);
                    }
                    failures << '\n';
                    (void)cutensorDestroyPlan(candidate);
                    (void)cutensorDestroyPlanPreference(preference);
                    continue;
                }

                result.plan_ = candidate;
                result.metrics_.i1_count = i1_count;
                result.metrics_.workspace_estimate = estimate;
                result.metrics_.workspace_limit = estimate;
                result.metrics_.workspace_actual = actual;
                result.metrics_.selected_algorithm =
                    static_cast<std::int32_t>(attempt.algorithm);
                result.metrics_.selected_jit =
                    attempt.jit != CUTENSOR_JIT_MODE_NONE;
                result.metrics_.selected_min_workspace =
                    attempt.workspace == CUTENSOR_WORKSPACE_MIN;
                result.metrics_.fallback_used = index != 0;
                std::ostringstream selected;
                selected << family << "(i1=" << i1_count << ")="
                         << algorithm_name(attempt.algorithm) << '/'
                         << jit_name(attempt.jit) << '/'
                         << workspace_name(attempt.workspace)
                         << " ws=" << actual << '/' << estimate
                         << " attempts=" << result.metrics_.attempts;
                result.metrics_.selected_summary = selected.str();
                (void)cutensorDestroyPlanPreference(preference);
                result.metrics_.creation_seconds =
                    std::chrono::duration<double>(
                        Clock::now() - total_begin).count();
                return result;
            }

            throw std::runtime_error(
                std::string("cuTENSOR ") + family +
                " plan creation exhausted all fallbacks for i1=" +
                std::to_string(i1_count) + " cap=" +
                std::to_string(workspace_cap) + " version=" +
                std::to_string(cutensorGetVersion()) + " cudart=" +
                std::to_string(cutensorGetCudartVersion()) +
                "\nA=" + describe_spec(A) + "\nB=" + describe_spec(B) +
                "\nD=" + describe_spec(D) + "\nattempts:\n" +
                failures.str());
        } catch (...) {
            result.reset();
            throw;
        }
    }

    void reset() noexcept
    {
        if (plan_ != nullptr) (void)cutensorDestroyPlan(plan_);
        if (operation_ != nullptr) {
            (void)cutensorDestroyOperationDescriptor(operation_);
        }
        if (desc_a_ != nullptr) (void)cutensorDestroyTensorDescriptor(desc_a_);
        if (desc_b_ != nullptr) (void)cutensorDestroyTensorDescriptor(desc_b_);
        if (desc_d_ != nullptr) (void)cutensorDestroyTensorDescriptor(desc_d_);
        plan_ = nullptr;
        operation_ = nullptr;
        desc_a_ = nullptr;
        desc_b_ = nullptr;
        desc_d_ = nullptr;
    }

    void move_from(BinaryPlan& other) noexcept
    {
        desc_a_ = std::exchange(other.desc_a_, nullptr);
        desc_b_ = std::exchange(other.desc_b_, nullptr);
        desc_d_ = std::exchange(other.desc_d_, nullptr);
        operation_ = std::exchange(other.operation_, nullptr);
        plan_ = std::exchange(other.plan_, nullptr);
        metrics_ = std::move(other.metrics_);
    }

    cutensorTensorDescriptor_t desc_a_ = nullptr;
    cutensorTensorDescriptor_t desc_b_ = nullptr;
    cutensorTensorDescriptor_t desc_d_ = nullptr;
    cutensorOperationDescriptor_t operation_ = nullptr;
    cutensorPlan_t plan_ = nullptr;
    PlanMetrics metrics_{};
};

} // namespace b4gpu::semifused_cutensor_detail
