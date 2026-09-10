// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/Failure.hpp"
#include "sl2cfoamxx/error.hpp"

#include <exception>
#include <filesystem>
#include <new>
#include <stdexcept>
#include <string>

namespace sl2cfoamxx::detail {

// Public computation boundaries use this to preserve the standard exception
// classes documented by the API while assigning a stable code to formerly
// unclassified runtime_error failures from private numerical implementations.
[[noreturn]] inline void rethrow_as_public_error(
    std::exception_ptr exception,
    ErrorCode fallback)
{
    if (exception == nullptr) {
        throw Error(
            ErrorCode::internal_failure,
            "sl2cfoam++: attempted to translate an empty exception");
    }

    try {
        std::rethrow_exception(exception);
    } catch (const Error&) {
        throw;
    } catch (const b4utils::ResourceLimitFailure& error) {
        throw Error(ErrorCode::resource_limit, error.what());
    } catch (const b4utils::NumericalFailure& error) {
        throw Error(ErrorCode::numerical_failure, error.what());
    } catch (const b4utils::InternalFailure& error) {
        throw Error(ErrorCode::internal_failure, error.what());
    } catch (const b4utils::UnsupportedOperationFailure& error) {
        throw Error(ErrorCode::unsupported_operation, error.what());
    } catch (const std::invalid_argument&) {
        throw;
    } catch (const std::out_of_range&) {
        throw;
    } catch (const std::length_error&) {
        throw;
    } catch (const std::overflow_error&) {
        throw;
    } catch (const std::filesystem::filesystem_error&) {
        throw;
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::logic_error& error) {
        throw Error(ErrorCode::internal_failure, error.what());
    } catch (const std::runtime_error& error) {
        throw Error(fallback, error.what());
    } catch (...) {
        throw Error(
            ErrorCode::internal_failure,
            "sl2cfoam++: unknown non-standard exception");
    }
}

} // namespace sl2cfoamxx::detail
