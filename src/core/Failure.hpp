// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <stdexcept>

namespace b4utils {

// Private transport type used by memory planners and execution policies.
// Public API boundaries translate it to
// sl2cfoamxx::ErrorCode::resource_limit, while direct internal callers retain
// the familiar std::runtime_error base class.
class ResourceLimitFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class NumericalFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class InternalFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class UnsupportedOperationFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

} // namespace b4utils
