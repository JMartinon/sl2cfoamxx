// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

#if defined(B4_SEMIFUSED_ENABLE_NVTX)
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#endif

namespace b4gpu::semifused_nvtx {

inline void name_stream(cudaStream_t stream, const char* name) noexcept
{
#if defined(B4_SEMIFUSED_ENABLE_NVTX)
    if (stream != nullptr && name != nullptr) nvtxNameCudaStreamA(stream, name);
#else
    (void)stream;
    (void)name;
#endif
}

class ScopedRange {
public:
    ScopedRange(
        bool enabled,
        const char* name,
        std::uint32_t color = UINT32_C(0xff4f81bd)) noexcept
        : active_(enabled)
    {
#if defined(B4_SEMIFUSED_ENABLE_NVTX)
        if (!active_ || name == nullptr) {
            active_ = false;
            return;
        }
        nvtxEventAttributes_t attributes{};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.colorType = NVTX_COLOR_ARGB;
        attributes.color = color;
        attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attributes.message.ascii = name;
        nvtxDomainRangePushEx(domain(), &attributes);
#else
        (void)name;
        (void)color;
        active_ = false;
#endif
    }

    ~ScopedRange()
    {
#if defined(B4_SEMIFUSED_ENABLE_NVTX)
        if (active_) nvtxDomainRangePop(domain());
#endif
    }

    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;

private:
#if defined(B4_SEMIFUSED_ENABLE_NVTX)
    [[nodiscard]] static nvtxDomainHandle_t domain() noexcept
    {
        static nvtxDomainHandle_t handle =
            nvtxDomainCreateA("b4.full_tensor.semifused");
        return handle;
    }
#endif
    bool active_ = false;
};

} // namespace b4gpu::semifused_nvtx
