// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "CoherentGpuApi.hpp"
#include "CuTensorCommon.hpp"
#include "CuTensorCompatibility.hpp"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cutensor.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace b4gpu::coherent_cutensor_detail {

using cutensor_detail::algorithm_name;
using cutensor_detail::check;
using cutensor_detail::planning_mutex;

struct TensorSpec {
  std::vector<std::int64_t> extents;
  std::vector<std::int64_t> strides;
  std::vector<std::int32_t> modes;
  cutensor_compat::DataType data_type = cutensor_compat::kReal64;
};

struct PlanMetrics {
  std::uint64_t workspace_actual = 0;
};

class Context {
public:
  explicit Context(bool enable_jit) : enable_jit_(enable_jit) {
    std::lock_guard<std::mutex> lock(planning_mutex());
    check(cutensorCreate(&handle_), "cutensorCreate coherent context");
  }

  ~Context() {
    if (handle_ != nullptr) {
      std::lock_guard<std::mutex> lock(planning_mutex());
      (void)cutensorDestroy(handle_);
    }
  }

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  [[nodiscard]] cutensorHandle_t handle() const noexcept { return handle_; }
  [[nodiscard]] bool enable_jit() const noexcept { return enable_jit_; }

private:
  cutensorHandle_t handle_ = nullptr;
  bool enable_jit_ = false;
};

inline std::string describe(const TensorSpec &spec) {
  std::ostringstream out;
  out << "type=" << static_cast<int>(spec.data_type) << " ext=[";
  for (std::size_t i = 0; i < spec.extents.size(); ++i) {
    if (i)
      out << ',';
    out << spec.extents[i];
  }
  out << "] stride=[";
  for (std::size_t i = 0; i < spec.strides.size(); ++i) {
    if (i)
      out << ',';
    out << spec.strides[i];
  }
  out << "] modes=[";
  for (std::size_t i = 0; i < spec.modes.size(); ++i) {
    if (i)
      out << ',';
    out << static_cast<char>(spec.modes[i]);
  }
  out << ']';
  return out.str();
}

inline cutensorTensorDescriptor_t create_descriptor(cutensorHandle_t handle,
                                                    const TensorSpec &spec) {
  if (spec.extents.empty() || spec.extents.size() != spec.strides.size() ||
      spec.extents.size() != spec.modes.size()) {
    throw std::logic_error("invalid coherent cuTENSOR tensor specification");
  }
  cutensorTensorDescriptor_t descriptor = nullptr;
  check(cutensorCreateTensorDescriptor(
            handle, &descriptor,
            static_cast<std::uint32_t>(spec.extents.size()),
            spec.extents.data(), spec.strides.data(), spec.data_type, 256u),
        "cutensorCreateTensorDescriptor coherent");
  return descriptor;
}

struct Attempt {
  cutensorAlgo_t algorithm;
  cutensorJitMode_t jit;
  cutensorWorksizePreference_t workspace;
};

inline void append_attempt(std::vector<Attempt> &attempts, Attempt attempt) {
  const auto duplicate = std::find_if(
      attempts.begin(), attempts.end(), [&](const Attempt &current) {
        return current.algorithm == attempt.algorithm &&
               current.jit == attempt.jit &&
               current.workspace == attempt.workspace;
      });
  if (duplicate == attempts.end())
    attempts.push_back(attempt);
}

class BinaryPlan {
public:
  BinaryPlan() = default;
  ~BinaryPlan() { reset(); }
  BinaryPlan(const BinaryPlan &) = delete;
  BinaryPlan &operator=(const BinaryPlan &) = delete;
  BinaryPlan(BinaryPlan &&other) noexcept { move_from(other); }
  BinaryPlan &operator=(BinaryPlan &&other) noexcept {
    if (this != &other) {
      reset();
      move_from(other);
    }
    return *this;
  }

  [[nodiscard]] static BinaryPlan
  make(const Context &context, const TensorSpec &A, const TensorSpec &B,
       const TensorSpec &D, std::uint64_t workspace_cap, std::string family) {
    BinaryPlan result;
    result.family_ = std::move(family);
    std::unique_lock<std::mutex> lock(planning_mutex());
    std::ostringstream failures;
    try {
      result.desc_a_ = create_descriptor(context.handle(), A);
      result.desc_b_ = create_descriptor(context.handle(), B);
      result.desc_d_ = create_descriptor(context.handle(), D);
      check(cutensorCreateContraction(
                context.handle(), &result.operation_, result.desc_a_,
                A.modes.data(), CUTENSOR_OP_IDENTITY, result.desc_b_,
                B.modes.data(), CUTENSOR_OP_IDENTITY, result.desc_d_,
                D.modes.data(), CUTENSOR_OP_IDENTITY, result.desc_d_,
                D.modes.data(), CUTENSOR_COMPUTE_DESC_64F),
            "cutensorCreateContraction coherent binary");

      const cutensorJitMode_t requested_jit = context.enable_jit()
                                                  ? CUTENSOR_JIT_MODE_DEFAULT
                                                  : CUTENSOR_JIT_MODE_NONE;
      std::vector<Attempt> attempts;
      append_attempt(attempts, {CUTENSOR_ALGO_DEFAULT, requested_jit,
                                CUTENSOR_WORKSPACE_DEFAULT});
      append_attempt(attempts, {CUTENSOR_ALGO_DEFAULT, requested_jit,
                                CUTENSOR_WORKSPACE_MIN});
      append_attempt(attempts, {CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE,
                                CUTENSOR_WORKSPACE_DEFAULT});
      append_attempt(attempts, {CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE,
                                CUTENSOR_WORKSPACE_MIN});
      append_attempt(attempts, {CUTENSOR_ALGO_GETT, CUTENSOR_JIT_MODE_NONE,
                                CUTENSOR_WORKSPACE_MIN});
      append_attempt(attempts, {CUTENSOR_ALGO_TGETT, CUTENSOR_JIT_MODE_NONE,
                                CUTENSOR_WORKSPACE_MIN});
      append_attempt(attempts, {CUTENSOR_ALGO_TTGT, CUTENSOR_JIT_MODE_NONE,
                                CUTENSOR_WORKSPACE_MIN});

      for (const Attempt &attempt : attempts) {
        cutensorPlanPreference_t preference = nullptr;
        cutensorStatus_t status = cutensorCreatePlanPreference(
            context.handle(), &preference, attempt.algorithm, attempt.jit);
        if (status != CUTENSOR_STATUS_SUCCESS) {
          failures << "preference " << algorithm_name(attempt.algorithm) << ": "
                   << cutensorGetErrorString(status) << '\n';
          continue;
        }
        std::uint64_t estimate = 0;
        status = cutensorEstimateWorkspaceSize(context.handle(),
                                               result.operation_, preference,
                                               attempt.workspace, &estimate);
        if (status != CUTENSOR_STATUS_SUCCESS || estimate > workspace_cap) {
          failures << "workspace " << algorithm_name(attempt.algorithm)
                   << " estimate=" << estimate << " cap=" << workspace_cap
                   << " status=" << cutensorGetErrorString(status) << '\n';
          (void)cutensorDestroyPlanPreference(preference);
          continue;
        }
        cutensorPlan_t plan = nullptr;
        status = cutensorCreatePlan(context.handle(), &plan, result.operation_,
                                    preference, estimate);
        (void)cutensorDestroyPlanPreference(preference);
        if (status != CUTENSOR_STATUS_SUCCESS) {
          if (plan != nullptr)
            (void)cutensorDestroyPlan(plan);
          failures << "plan " << algorithm_name(attempt.algorithm) << ": "
                   << cutensorGetErrorString(status) << '\n';
          continue;
        }
        std::uint64_t actual = 0;
        status = cutensorPlanGetAttribute(context.handle(), plan,
                                          CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                                          &actual, sizeof(actual));
        if (status != CUTENSOR_STATUS_SUCCESS || actual > workspace_cap) {
          (void)cutensorDestroyPlan(plan);
          failures << "required workspace invalid\n";
          continue;
        }
        result.plan_ = plan;
        result.metrics_.workspace_actual = actual;
        return result;
      }
      throw std::runtime_error(
          "coherent cuTENSOR binary plan creation exhausted fallbacks for " +
          result.family_ + "\nA=" + describe(A) + "\nB=" + describe(B) +
          "\nD=" + describe(D) + "\n" + failures.str());
    } catch (...) {
      result.reset();
      throw;
    }
  }

  void execute(const Context &context, const void *A, const void *B, void *D,
               void *workspace, cudaStream_t stream) const {
    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    check(cutensorContract(context.handle(), plan_, &alpha, A, B, &beta, D, D,
                           workspace, metrics_.workspace_actual, stream),
          "cutensorContract coherent binary");
  }

  [[nodiscard]] const PlanMetrics &metrics() const noexcept { return metrics_; }

private:
  void reset() noexcept {
    if (plan_ != nullptr)
      (void)cutensorDestroyPlan(plan_);
    if (operation_ != nullptr)
      (void)cutensorDestroyOperationDescriptor(operation_);
    if (desc_a_ != nullptr)
      (void)cutensorDestroyTensorDescriptor(desc_a_);
    if (desc_b_ != nullptr)
      (void)cutensorDestroyTensorDescriptor(desc_b_);
    if (desc_d_ != nullptr)
      (void)cutensorDestroyTensorDescriptor(desc_d_);
    plan_ = nullptr;
    operation_ = nullptr;
    desc_a_ = desc_b_ = desc_d_ = nullptr;
  }
  void move_from(BinaryPlan &other) noexcept {
    desc_a_ = std::exchange(other.desc_a_, nullptr);
    desc_b_ = std::exchange(other.desc_b_, nullptr);
    desc_d_ = std::exchange(other.desc_d_, nullptr);
    operation_ = std::exchange(other.operation_, nullptr);
    plan_ = std::exchange(other.plan_, nullptr);
    metrics_ = std::move(other.metrics_);
    family_ = std::move(other.family_);
  }

  cutensorTensorDescriptor_t desc_a_ = nullptr;
  cutensorTensorDescriptor_t desc_b_ = nullptr;
  cutensorTensorDescriptor_t desc_d_ = nullptr;
  cutensorOperationDescriptor_t operation_ = nullptr;
  cutensorPlan_t plan_ = nullptr;
  PlanMetrics metrics_{};
  std::string family_;
};

} // namespace b4gpu::coherent_cutensor_detail
