// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Public shared-library visibility for the supported Linux and macOS builds.
//
// sl2cfoam++ is built with hidden visibility by default so private numerical
// helpers do not accidentally become part of its binary interface. Public
// declarations carry this attribute and remain available to applications.
// This is unrelated to C++ namespaces: users still call only
// sl2cfoamxx::... names.
#if defined(__GNUC__) || defined(__clang__)
#  define SL2CFOAMXX_API __attribute__((visibility("default")))
#else
// The production build currently requires GNU C++, but keeping the installed
// headers parseable gives unsupported tooling a harmless fallback.
#  define SL2CFOAMXX_API
#endif
