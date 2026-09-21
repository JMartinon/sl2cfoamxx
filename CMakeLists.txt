# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later


cmake_minimum_required(VERSION 3.22)

include(cmake/SelectMacOSGNU.cmake)

project(sl2cfoamxx
  VERSION 1.0.2
  DESCRIPTION "High-performance sl2cfoam++ vertex amplitudes and tensors"
  LANGUAGES CXX
)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR
    "sl2cfoam++ requires GCC/G++ with OpenMP. Configure a fresh build directory "
    "with -DCMAKE_CXX_COMPILER=/path/to/g++")
endif()

if(NOT UNIX)
  message(FATAL_ERROR
    "sl2cfoam++ currently requires a POSIX platform for memory-mapped tensor "
    "storage and atomic tensor-cache locking")
endif()

# ==============================================================================
# Production build options
# ==============================================================================
#
# The production surface contains:
#
#   * one versioned shared host library behind a small public header;
#   * one CUDA archive containing only coherent/full-tensor production sources;
#   * optional public-API workflow tests, one source per language.
#
# Keep the option names used by the cluster scripts.

option(ENABLE_CUDA        "Build the production NVIDIA GPU backends" OFF)
option(ENABLE_NATIVE_ARCH "Compile host code for the local CPU" ON)
option(ENABLE_FAST_MATH   "Enable CUDA fast math in production GPU kernels" ON)
option(BUILD_TESTING       "Build public API integration tests" OFF)
option(SL2CFOAMXX_BUILD_PYTHON
       "Build the Python API (downloads pinned nanobind into third-party/)" OFF)
option(SL2CFOAMXX_BUILD_JULIA
       "Build the Julia API through Julia's built-in C interface" OFF)
option(SL2CFOAMXX_BUILD_MATHEMATICA
       "Build the Mathematica API with the installed Wolfram LibraryLink SDK" OFF)

if(ENABLE_CUDA AND DEFINED HEAVY_COMPUTE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR
    "HEAVY_COMPUTE_CUDA_ARCHITECTURES was renamed to CUDA_ARCHITECTURES. "
    "Reconfigure with -UHEAVY_COMPUTE_CUDA_ARCHITECTURES "
    "-DCUDA_ARCHITECTURES=<your GPU architectures>.")
endif()

set(CUDA_ARCHITECTURES
    "80"
    CACHE STRING
    "CUDA architectures for production GPU backends; select every deployed GPU architecture")

set(CUTENSOR_ROOT
    ""
    CACHE PATH
    "Root of an external cuTENSOR installation containing include/cutensor.h and libcutensor")

# Use CMake's standard FindBLAS selector rather than a project-specific vendor
# option.  Keep Accelerate as the macOS default and let other platforms search
# all providers unless the caller requests one explicitly (for example AOCL or
# AOCL_mt).  Preserve FindBLAS's environment-variable behavior on the first
# configure while making the resolved choice visible and editable in the cache.
if(DEFINED BLA_VENDOR AND NOT BLA_VENDOR STREQUAL "")
  set(_sl2cfoamxx_blas_vendor_default "${BLA_VENDOR}")
elseif(DEFINED ENV{BLA_VENDOR} AND NOT "$ENV{BLA_VENDOR}" STREQUAL "")
  set(_sl2cfoamxx_blas_vendor_default "$ENV{BLA_VENDOR}")
elseif(APPLE)
  set(_sl2cfoamxx_blas_vendor_default "Apple")
else()
  set(_sl2cfoamxx_blas_vendor_default "All")
endif()
set(BLA_VENDOR
    "${_sl2cfoamxx_blas_vendor_default}"
    CACHE STRING
    "FindBLAS vendor (for example Apple, AOCL, AOCL_mt, OpenBLAS, FLAME, Intel10_64lp, or All)"
    FORCE)
unset(_sl2cfoamxx_blas_vendor_default)

# ==============================================================================
# Language levels and build type
# ==============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ==============================================================================
# Host dependencies
# ==============================================================================

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# The numerical kernels use only the standard CBLAS surface. Accelerate is the
# native provider on macOS; elsewhere accept any BLAS implementation that also
# supplies cblas.h and the CBLAS entry points (OpenBLAS, BLIS, MKL, ...).
add_library(sl2cfoamxx_blas INTERFACE)
if(APPLE)
  if(NOT BLA_VENDOR STREQUAL "Apple")
    message(FATAL_ERROR
      "macOS builds use the Accelerate CBLAS header supplied by the selected "
      "SDK and therefore require -DBLA_VENDOR=Apple (the default). Requested "
      "BLA_VENDOR='${BLA_VENDOR}'.")
  endif()
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(sl2cfoamxx_blas INTERFACE
    "${ACCELERATE_FRAMEWORK}")
  # GCC consumes the CBLAS header directly instead of the Clang-only umbrella.
  # Framework discovery honors CMake's standard SDK selection.
  find_path(SL2CFOAMXX_CBLAS_INCLUDE_DIR cblas_new.h
    HINTS "${ACCELERATE_FRAMEWORK}/Frameworks/vecLib.framework/Headers"
    REQUIRED)
  target_include_directories(sl2cfoamxx_blas INTERFACE
    "${SL2CFOAMXX_CBLAS_INCLUDE_DIR}")
  set(SL2CFOAMXX_BLAS_PROVIDER "Accelerate")
else()
  set(BLA_SIZEOF_INTEGER 4)
  if(BLA_VENDOR MATCHES "^AOCL(_mt)?$" AND CMAKE_VERSION VERSION_LESS 3.27)
    message(FATAL_ERROR
      "BLA_VENDOR='${BLA_VENDOR}' requires CMake 3.27 or newer because older "
      "FindBLAS modules do not support AOCL. Upgrade CMake or select another "
      "CBLAS-capable vendor.")
  endif()
  find_package(BLAS QUIET)
  if(NOT BLAS_FOUND)
    message(FATAL_ERROR
      "Could not find the requested 32-bit-integer BLAS vendor "
      "BLA_VENDOR='${BLA_VENDOR}'. Use a vendor name supported by CMake's "
      "FindBLAS module and make the installation visible with "
      "-DCMAKE_PREFIX_PATH=/path/to/prefix. For AOCL with CMake 3.27 or newer, "
      "select -DBLA_VENDOR=AOCL (single-threaded) or AOCL_mt (threaded); if its "
      "LP64 library directory is outside the prefix's standard layout, also "
      "set -DCMAKE_LIBRARY_PATH=/path/to/aocl/lib/directory.")
  endif()
  find_path(SL2CFOAMXX_CBLAS_INCLUDE_DIR
    NAMES cblas.h blis/cblas.h mkl_cblas.h
    PATH_SUFFIXES LP64 include/LP64 openblas openblas64)
  if(NOT SL2CFOAMXX_CBLAS_INCLUDE_DIR)
    message(FATAL_ERROR
      "BLAS vendor '${BLA_VENDOR}' was found, but no supported CBLAS header "
      "(cblas.h, blis/cblas.h, or mkl_cblas.h) was found. Install the provider's "
      "development headers and make their prefix visible through "
      "CMAKE_PREFIX_PATH or their directory visible through CMAKE_INCLUDE_PATH.")
  endif()

  target_include_directories(sl2cfoamxx_blas INTERFACE
    "${SL2CFOAMXX_CBLAS_INCLUDE_DIR}")
  if(TARGET BLAS::BLAS)
    target_link_libraries(sl2cfoamxx_blas INTERFACE BLAS::BLAS)
    set(_sl2cfoamxx_blas_check_libraries BLAS::BLAS)
  else()
    target_link_libraries(sl2cfoamxx_blas INTERFACE ${BLAS_LIBRARIES})
    set(_sl2cfoamxx_blas_check_libraries ${BLAS_LIBRARIES})
  endif()

  include(CheckCXXSourceCompiles)
  set(_saved_required_includes "${CMAKE_REQUIRED_INCLUDES}")
  set(_saved_required_libraries "${CMAKE_REQUIRED_LIBRARIES}")
  set(_saved_required_link_options "${CMAKE_REQUIRED_LINK_OPTIONS}")
  set(CMAKE_REQUIRED_INCLUDES "${SL2CFOAMXX_CBLAS_INCLUDE_DIR}")
  set(CMAKE_REQUIRED_LIBRARIES ${_sl2cfoamxx_blas_check_libraries})
  set(CMAKE_REQUIRED_LINK_OPTIONS ${BLAS_LINKER_FLAGS})
  unset(SL2CFOAMXX_BLAS_HAS_CBLAS CACHE)
  check_cxx_source_compiles(
    "#if __has_include(<cblas.h>)
     #  include <cblas.h>
     #elif __has_include(<blis/cblas.h>)
     #  include <blis/cblas.h>
     #elif __has_include(<mkl_cblas.h>)
     #  include <mkl_cblas.h>
     #else
     #  error \"No supported CBLAS header\"
     #endif
     int main() {
       double a[1] = {1.0}, b[1] = {1.0}, c[1] = {0.0};
       cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                   1, 1, 1, 1.0, a, 1, b, 1, 0.0, c, 1);
       return c[0] == 1.0 ? 0 : 1;
     }"
    SL2CFOAMXX_BLAS_HAS_CBLAS)
  set(CMAKE_REQUIRED_INCLUDES "${_saved_required_includes}")
  set(CMAKE_REQUIRED_LIBRARIES "${_saved_required_libraries}")
  set(CMAKE_REQUIRED_LINK_OPTIONS "${_saved_required_link_options}")
  unset(_saved_required_includes)
  unset(_saved_required_libraries)
  unset(_saved_required_link_options)
  unset(_sl2cfoamxx_blas_check_libraries)
  if(NOT SL2CFOAMXX_BLAS_HAS_CBLAS)
    message(FATAL_ERROR
      "BLAS vendor '${BLA_VENDOR}' was found, but its headers and libraries do "
      "not provide a usable CBLAS API with 32-bit integers. Select a matching "
      "CBLAS-capable installation with BLA_VENDOR and CMAKE_PREFIX_PATH.")
  endif()
  set(SL2CFOAMXX_BLAS_PROVIDER
      "${BLA_VENDOR} (${BLAS_LIBRARIES})")
endif()

# ==============================================================================
# Host library and public API
# ==============================================================================

set(_host_sources
  "${CMAKE_SOURCE_DIR}/src/api/Capabilities.cpp"
  "${CMAKE_SOURCE_DIR}/src/api/Tensor.cpp"
  "${CMAKE_SOURCE_DIR}/src/api/FullTensorContainer.cpp"
  "${CMAKE_SOURCE_DIR}/src/api/Configuration.cpp"
  "${CMAKE_SOURCE_DIR}/src/api/InputValidation.cpp"
  "${CMAKE_SOURCE_DIR}/src/api/Amplitude.cpp"
  "${CMAKE_SOURCE_DIR}/src/api/AmplitudeCache.cpp"
  "${CMAKE_SOURCE_DIR}/src/api/CoherentBoundary.cpp"
  "${CMAKE_SOURCE_DIR}/src/api/CoherentAmplitude.cpp"
  "${CMAKE_SOURCE_DIR}/src/api/FullTensor.cpp"
  "${CMAKE_SOURCE_DIR}/src/api/FullTensorCpu.cpp"
  "${CMAKE_SOURCE_DIR}/src/api/FullTensorGpu.cpp"
  "${CMAKE_SOURCE_DIR}/src/api/ProgressReporter.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/AmplitudeInputSupport.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/AmplitudeInputSupport_HalfSjC.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/AmplitudePackedAutoSlab.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/AmplitudePackedBoosters.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/AmplitudePackedCoherentI1Batch.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/AmplitudePackedContraction.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/AmplitudePackedContractionGeneral.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbit.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbitCostSelected.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbitCostSelected_HalfSjC.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/AmplitudePackedSJTensors_HalfSjC.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/AmplitudePackedXQSlab.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/AmplitudePackedXSlab.cpp"
  "${CMAKE_SOURCE_DIR}/src/amplitude/coherent_intertwiner.cpp"
  "${CMAKE_SOURCE_DIR}/src/full_tensor/CpuContraction.cpp"
  "${CMAKE_SOURCE_DIR}/src/full_tensor/FullTensorExecutionPlanner.cpp"
  "${CMAKE_SOURCE_DIR}/src/full_tensor/FullTensorInputSupport_HalfSjC.cpp"
  "${CMAKE_SOURCE_DIR}/src/full_tensor/FullTensorPackedSJTensors_HalfSjC.cpp"
  "${CMAKE_SOURCE_DIR}/src/full_tensor/FullTensorPackedBoosters_HalfSjC.cpp"
  "${CMAKE_SOURCE_DIR}/src/full_tensor/FullTensorQReflectionOrbitPlan.cpp"
  "${CMAKE_SOURCE_DIR}/src/core/HostMemoryBudget.cpp"
  "${CMAKE_SOURCE_DIR}/src/core/OutputPresentation.cpp")

# A shared public library deliberately forms the final link boundary so an
# application only needs to link sl2cfoamxx itself.
add_library(sl2cfoamxx SHARED ${_host_sources})
add_library(sl2cfoamxx::sl2cfoamxx ALIAS sl2cfoamxx)

if(APPLE)
  # The required Homebrew GCC driver already emits @loader_path for Darwin
  # shared libraries.  Repeating it through INSTALL_RPATH makes both ld and
  # install_name_tool diagnose a duplicate LC_RPATH.
  set(_sl2cfoamxx_install_rpath "")
else()
  set(_sl2cfoamxx_install_rpath "$ORIGIN")
endif()

set_target_properties(sl2cfoamxx PROPERTIES
  VERSION "${PROJECT_VERSION}"
  # 1.x uses a major-version ABI name. Pre-1.0 clients retain their 0.x ABI.
  SOVERSION "${PROJECT_VERSION_MAJOR}"
  # Only declarations marked SL2CFOAMXX_API belong to the shared-library ABI.
  # This keeps private recurrence/contraction helpers out of ELF and Mach-O
  # dynamic symbol tables without changing their C++ namespaces.
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_RPATH_USE_ORIGIN ON
  INSTALL_RPATH "${_sl2cfoamxx_install_rpath}")

target_compile_features(sl2cfoamxx PUBLIC cxx_std_20)
target_compile_definitions(sl2cfoamxx PRIVATE
  SL2CFOAMXX_COMPILED_WITH_CUDA=$<BOOL:${ENABLE_CUDA}>)
target_include_directories(sl2cfoamxx
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE
    "${CMAKE_SOURCE_DIR}/src")

target_link_libraries(sl2cfoamxx PRIVATE
  sl2cfoamxx_blas
  Threads::Threads
  OpenMP::OpenMP_CXX)

# GCC/Clang visibility flags cannot change symbols that arrive from prebuilt
# static archives. On ELF, localize those archive members as well (including
# the private CUDA implementation archive and static BLAS variants).
target_link_options(sl2cfoamxx PRIVATE
  $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs=ALL>)

if(ENABLE_NATIVE_ARCH)
  target_compile_options(sl2cfoamxx PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()
# Host formulas deliberately use infinities and finite-value checks while
# evaluating logarithmic special functions.  -ffast-math invalidates those
# semantics, so the option below applies only to the CUDA kernels.
target_compile_options(sl2cfoamxx PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-psabi>)

target_compile_options(sl2cfoamxx PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RelWithDebInfo>>:-O3;-g>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0;-g>)

if(APPLE)
  target_compile_definitions(sl2cfoamxx PRIVATE
    B4FULL_USE_ACCELERATE=1
    ACCELERATE_NEW_LAPACK=1)
endif()

# ==============================================================================
# Production CUDA library
# ==============================================================================

if(ENABLE_CUDA)
  if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH
        "Host compiler used by nvcc" FORCE)
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "ENABLE_CUDA=ON, but nvcc was not found. Set CUDACXX or CMAKE_CUDA_COMPILER")
  endif()

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  find_path(CUTENSOR_INCLUDE_DIR
    NAMES cutensor.h
    HINTS "${CUTENSOR_ROOT}" ENV CUTENSOR_ROOT
    PATH_SUFFIXES include)
  find_library(CUTENSOR_LIBRARY
    NAMES cutensor libcutensor
    HINTS "${CUTENSOR_ROOT}" ENV CUTENSOR_ROOT
    # NVIDIA bundles may contain several CUDA-major variants. Selecting the
    # first available major can pull in a different cuBLAS/runtime dependency.
    PATH_SUFFIXES lib lib64 "lib/${CUDAToolkit_VERSION_MAJOR}")

  if(NOT CUTENSOR_INCLUDE_DIR OR NOT CUTENSOR_LIBRARY)
    message(FATAL_ERROR
      "The production GPU algorithms require an external cuTENSOR install. "
      "Set CUTENSOR_ROOT to a root containing include/cutensor.h and lib/libcutensor.")
  endif()

  add_library(sl2cfoamxx_cutensor UNKNOWN IMPORTED GLOBAL)
  set_target_properties(sl2cfoamxx_cutensor PROPERTIES
    IMPORTED_LOCATION "${CUTENSOR_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_INCLUDE_DIR}")
  get_filename_component(CUTENSOR_LIBRARY_DIR "${CUTENSOR_LIBRARY}" DIRECTORY)

  set(_gpu_sources
    "${CMAKE_SOURCE_DIR}/src/gpu/CudaIntrospection.cu"
    "${CMAKE_SOURCE_DIR}/src/gpu/CoherentGpuBackend.cu"
    "${CMAKE_SOURCE_DIR}/src/gpu/CoherentGpuRecurrence.cu"
    "${CMAKE_SOURCE_DIR}/src/gpu/FullTensorGpuDeviceSixJ.cu"
    "${CMAKE_SOURCE_DIR}/src/gpu/FullTensorSemiFusedGpuBackend.cu"
    "${CMAKE_SOURCE_DIR}/src/gpu/FullTensorProductionGpuBridge.cu")

  foreach(_source IN LISTS _gpu_sources)
    if(NOT EXISTS "${_source}")
      message(FATAL_ERROR "Required production CUDA source is missing: ${_source}")
    endif()
  endforeach()

  add_library(heavy_compute_cuda STATIC ${_gpu_sources})
  add_library(sl2cfoamxx::cuda ALIAS heavy_compute_cuda)

  set_target_properties(heavy_compute_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")

  target_include_directories(heavy_compute_cuda PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/include"
    "${CUTENSOR_INCLUDE_DIR}")

  target_link_libraries(heavy_compute_cuda PRIVATE
    CUDA::cudart
    CUDA::cublas
    sl2cfoamxx_cutensor
    ${CMAKE_DL_LIBS})

  target_compile_definitions(heavy_compute_cuda PRIVATE
    HEAVY_COMPUTE_USE_CUDA=1)

  target_compile_options(heavy_compute_cuda PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-O3;-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)

  if(ENABLE_FAST_MATH)
    target_compile_options(heavy_compute_cuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  endif()

  # Override target-wide fast math for the numerically sensitive 6j recurrence.
  set_source_files_properties(
    "${CMAKE_SOURCE_DIR}/src/gpu/CoherentGpuRecurrence.cu"
    "${CMAKE_SOURCE_DIR}/src/gpu/FullTensorGpuDeviceSixJ.cu"
    PROPERTIES COMPILE_OPTIONS
      "--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false")

  set_source_files_properties(
    "${CMAKE_SOURCE_DIR}/src/gpu/FullTensorSemiFusedGpuBackend.cu"
    PROPERTIES COMPILE_OPTIONS
      "--fmad=true;--expt-relaxed-constexpr")

  set_source_files_properties(
    "${CMAKE_SOURCE_DIR}/src/api/CoherentAmplitude.cpp"
    "${CMAKE_SOURCE_DIR}/src/api/FullTensorGpu.cpp"
    PROPERTIES COMPILE_DEFINITIONS HEAVY_COMPUTE_USE_CUDA=1)

  target_link_libraries(sl2cfoamxx PRIVATE heavy_compute_cuda)
endif()

# Compile immutable build metadata into the shared library. Keep these
# definitions private: version() intentionally reports the library that was
# loaded at runtime, which need not be the same copy as the caller's headers.
set(_sl2cfoamxx_cuda_architectures_string "")
set(_sl2cfoamxx_cuda_toolkit_version "")
set(_sl2cfoamxx_cutensor_version "")
set(_sl2cfoamxx_cuda_fast_math_enabled 0)
# Convert a potentially multi-library CMake list into one compile-definition
# string. This is a provider/link description, not merely the BLA_VENDOR
# selector (which is often the intentionally generic value "All").
string(REPLACE ";" "," _sl2cfoamxx_blas_provider_string
       "${SL2CFOAMXX_BLAS_PROVIDER}")
string(REPLACE "\\" "\\\\" _sl2cfoamxx_blas_provider_string
       "${_sl2cfoamxx_blas_provider_string}")
string(REPLACE "\"" "\\\"" _sl2cfoamxx_blas_provider_string
       "${_sl2cfoamxx_blas_provider_string}")
if(ENABLE_CUDA)
  string(REPLACE ";" "," _sl2cfoamxx_cuda_architectures_string
         "${CUDA_ARCHITECTURES}")
  set(_sl2cfoamxx_cuda_toolkit_version "${CUDAToolkit_VERSION}")
  if(ENABLE_FAST_MATH)
    set(_sl2cfoamxx_cuda_fast_math_enabled 1)
  endif()

  # cuTENSOR exposes a conventional three-component version in cutensor.h.
  # If a future distribution moves those macros, retain an explicit unknown
  # value rather than making an otherwise usable CUDA configuration fail.
  foreach(_component MAJOR MINOR PATCH)
    file(STRINGS "${CUTENSOR_INCLUDE_DIR}/cutensor.h"
      _sl2cfoamxx_cutensor_${_component}_line
      REGEX "^#[ \t]*define[ \t]+CUTENSOR_${_component}[ \t]+[0-9]+")
    if(_sl2cfoamxx_cutensor_${_component}_line)
      list(GET _sl2cfoamxx_cutensor_${_component}_line 0
        _sl2cfoamxx_cutensor_${_component}_line)
      string(REGEX REPLACE
        ".*CUTENSOR_${_component}[ \t]+([0-9]+).*" "\\1"
        _sl2cfoamxx_cutensor_${_component}
        "${_sl2cfoamxx_cutensor_${_component}_line}")
    else()
      set(_sl2cfoamxx_cutensor_${_component} "")
    endif()
  endforeach()
  if(NOT _sl2cfoamxx_cutensor_MAJOR STREQUAL "" AND
     NOT _sl2cfoamxx_cutensor_MINOR STREQUAL "" AND
     NOT _sl2cfoamxx_cutensor_PATCH STREQUAL "")
    set(_sl2cfoamxx_cutensor_version
        "${_sl2cfoamxx_cutensor_MAJOR}.${_sl2cfoamxx_cutensor_MINOR}.${_sl2cfoamxx_cutensor_PATCH}")
  else()
    set(_sl2cfoamxx_cutensor_version "unknown")
  endif()
endif()

target_compile_definitions(sl2cfoamxx PRIVATE
  SL2CFOAMXX_VERSION_STRING="${PROJECT_VERSION}"
  SL2CFOAMXX_COMPILER_ID_STRING="${CMAKE_CXX_COMPILER_ID}"
  SL2CFOAMXX_COMPILER_VERSION_STRING="${CMAKE_CXX_COMPILER_VERSION}"
  SL2CFOAMXX_BUILD_TYPE_STRING="$<CONFIG>"
  SL2CFOAMXX_BLAS_VENDOR_STRING="${BLA_VENDOR}"
  SL2CFOAMXX_BLAS_PROVIDER_STRING="${_sl2cfoamxx_blas_provider_string}"
  SL2CFOAMXX_NATIVE_ARCH_ENABLED=$<BOOL:${ENABLE_NATIVE_ARCH}>
  SL2CFOAMXX_CUDA_FAST_MATH_ENABLED=${_sl2cfoamxx_cuda_fast_math_enabled}
  SL2CFOAMXX_CUDA_TOOLKIT_VERSION_STRING="${_sl2cfoamxx_cuda_toolkit_version}"
  SL2CFOAMXX_CUDA_ARCHITECTURES_STRING="${_sl2cfoamxx_cuda_architectures_string}"
  SL2CFOAMXX_CUTENSOR_VERSION_STRING="${_sl2cfoamxx_cutensor_version}")

# The optional language wrapper links only the supported shared-library API.
if(SL2CFOAMXX_BUILD_PYTHON)
  add_subdirectory(python)
endif()
if(SL2CFOAMXX_BUILD_JULIA)
  add_subdirectory(julia)
endif()
if(SL2CFOAMXX_BUILD_MATHEMATICA)
  add_subdirectory(mathematica)
endif()

# ==============================================================================
# Optional public-API integration test
# ==============================================================================

if(BUILD_TESTING)
  add_executable(test_public_api tests/test_public_api.cpp)
  target_link_libraries(test_public_api PRIVATE
    sl2cfoamxx::sl2cfoamxx OpenMP::OpenMP_CXX)
  target_compile_options(test_public_api PRIVATE -Wno-psabi)
  target_compile_definitions(test_public_api PRIVATE
    SL2CFOAMXX_TEST_WITH_CUDA=$<BOOL:${ENABLE_CUDA}>)
  add_test(NAME public_api COMMAND test_public_api)
  set_tests_properties(public_api PROPERTIES
    LABELS "cpu;public-api;integration" TIMEOUT 300)
  if(ENABLE_CUDA)
    set_tests_properties(public_api PROPERTIES
      LABELS "cpu;cuda;public-api;integration" RESOURCE_LOCK sl2cfoamxx_gpu)
  endif()
endif()

# ==============================================================================
# Installation
# ==============================================================================

set(_sl2cfoamxx_cmake_install_dir
    "${CMAKE_INSTALL_LIBDIR}/cmake/sl2cfoamxx")

install(TARGETS sl2cfoamxx
  EXPORT sl2cfoamxxTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

install(FILES "${CMAKE_SOURCE_DIR}/include/sl2cfoamxx.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(FILES
  "${CMAKE_SOURCE_DIR}/include/sl2cfoamxx/sl2cfoamxx.hpp"
  "${CMAKE_SOURCE_DIR}/include/sl2cfoamxx/capabilities.hpp"
  "${CMAKE_SOURCE_DIR}/include/sl2cfoamxx/configuration.hpp"
  "${CMAKE_SOURCE_DIR}/include/sl2cfoamxx/error.hpp"
  "${CMAKE_SOURCE_DIR}/include/sl2cfoamxx/export.hpp"
  "${CMAKE_SOURCE_DIR}/include/sl2cfoamxx/types.hpp"
  "${CMAKE_SOURCE_DIR}/include/sl2cfoamxx/tensor.hpp"
  "${CMAKE_SOURCE_DIR}/include/sl2cfoamxx/full_tensor.hpp"
  "${CMAKE_SOURCE_DIR}/include/sl2cfoamxx/progress.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sl2cfoamxx")

configure_package_config_file(
  "${CMAKE_SOURCE_DIR}/cmake/sl2cfoamxxConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/sl2cfoamxxConfig.cmake"
  INSTALL_DESTINATION "${_sl2cfoamxx_cmake_install_dir}")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/sl2cfoamxxConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  # A compatible 1.x package must retain the public major-version ABI.
  COMPATIBILITY SameMajorVersion)

install(EXPORT sl2cfoamxxTargets
  FILE sl2cfoamxxTargets.cmake
  NAMESPACE sl2cfoamxx::
  DESTINATION "${_sl2cfoamxx_cmake_install_dir}")

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/sl2cfoamxxConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/sl2cfoamxxConfigVersion.cmake"
  DESTINATION "${_sl2cfoamxx_cmake_install_dir}")

# ==============================================================================
# Concise diagnostics
# ==============================================================================

message(STATUS "sl2cfoam++ production configuration")
message(STATUS "  build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler     : ${CMAKE_CXX_COMPILER}")
message(STATUS "  OpenMP           : required (${OpenMP_CXX_VERSION})")
message(STATUS "  BLAS/CBLAS       : ${SL2CFOAMXX_BLAS_PROVIDER}")
message(STATUS "  CUDA             : ${ENABLE_CUDA}")
message(STATUS "  Python API       : ${SL2CFOAMXX_BUILD_PYTHON}")
message(STATUS "  Julia API        : ${SL2CFOAMXX_BUILD_JULIA}")
message(STATUS "  Mathematica API  : ${SL2CFOAMXX_BUILD_MATHEMATICA}")
if(ENABLE_CUDA)
  message(STATUS "  nvcc             : ${CMAKE_CUDA_COMPILER}")
  message(STATUS "  CUDA toolkit     : ${CUDAToolkit_VERSION}")
  message(STATUS "  CUDA arches      : ${CUDA_ARCHITECTURES}")
  message(STATUS "  cuTENSOR         : ${CUTENSOR_LIBRARY}")
endif()
message(STATUS "  tests            : ${BUILD_TESTING}")
