// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/Failure.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace b4gpu::cuda_detail {

inline void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        if (status == cudaErrorMemoryAllocation) {
            throw b4utils::ResourceLimitFailure(
                std::string(operation) + ": " + cudaGetErrorString(status));
        }
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) { allocate(count); }

    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0))
    {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void allocate(std::size_t count)
    {
        reset();
        if (count == 0) return;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::overflow_error("CUDA allocation byte count overflow");
        }
        void* allocation = nullptr;
        check_cuda(cudaMalloc(&allocation, count * sizeof(T)), "cudaMalloc");
        data_ = static_cast<T*>(allocation);
        count_ = count;
    }

    void reset() noexcept
    {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
            data_ = nullptr;
            count_ = 0;
        }
    }

    [[nodiscard]] T* get() noexcept { return data_; }
    [[nodiscard]] const T* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class Stream {
public:
    Stream() { check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                          "cudaStreamCreateWithFlags"); }
    ~Stream() { if (stream_ != nullptr) (void)cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

} // namespace b4gpu::cuda_detail
