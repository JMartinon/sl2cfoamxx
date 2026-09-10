// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/capabilities.hpp"

#include <cstddef>

namespace sl2cfoamxx::detail {

[[nodiscard]] std::size_t cuda_device_count_impl();

[[nodiscard]] CudaDeviceInfo cuda_device_info_impl(std::size_t ordinal);

} // namespace sl2cfoamxx::detail
