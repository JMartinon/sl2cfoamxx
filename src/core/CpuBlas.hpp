// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Use the platform CBLAS interface directly.  Accelerate does not install a
// top-level <cblas.h> for Apple Clang, while Homebrew GCC builds use the CBLAS
// header exposed from the selected macOS SDK (configured in CMake).
#if defined(__APPLE__)
#  include <cblas_new.h>
#elif __has_include(<cblas.h>)
#  include <cblas.h>
#elif __has_include(<openblas/cblas.h>)
#  include <openblas/cblas.h>
#elif __has_include(<blis/cblas.h>)
#  include <blis/cblas.h>
#elif __has_include(<mkl_cblas.h>)
#  include <mkl_cblas.h>
#else
#  error "A CBLAS header is required (Accelerate, OpenBLAS, BLIS, or MKL)"
#endif
