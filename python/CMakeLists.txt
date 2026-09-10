# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later


# Python bindings deliberately consume the public C++ shared target. The
# nanobind source distribution includes its robin_map headers, unlike GitHub's
# automatically generated archives, so no submodule or pip bootstrap is needed.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

set(SL2CFOAMXX_NANOBIND_SOURCE_DIR "" CACHE PATH
    "Use an existing nanobind 2.13.0 source tree, including ext/robin_map (offline builds)")
set(_sl2cfoamxx_nanobind_version "2.13.0")
set(_sl2cfoamxx_nanobind_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/nanobind")
if(SL2CFOAMXX_NANOBIND_SOURCE_DIR)
  set(_sl2cfoamxx_nanobind_source_dir "${SL2CFOAMXX_NANOBIND_SOURCE_DIR}")
else()
  set(_sl2cfoamxx_nanobind_source_dir "${PROJECT_SOURCE_DIR}/third-party/nanobind")
endif()

set(NB_TEST OFF)
set(NB_CREATE_INSTALL_RULES OFF)
set(NB_USE_SUBMODULE_DEPS ON)
if(NOT EXISTS "${_sl2cfoamxx_nanobind_source_dir}/CMakeLists.txt")
  if(SL2CFOAMXX_NANOBIND_SOURCE_DIR)
    message(FATAL_ERROR
      "SL2CFOAMXX_NANOBIND_SOURCE_DIR does not contain nanobind sources: "
      "${SL2CFOAMXX_NANOBIND_SOURCE_DIR}")
  endif()
  include(FetchContent)
  # Prefer extraction timestamps when supported (CMake 3.24+).
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()
  FetchContent_Declare(sl2cfoamxx_nanobind
    URL "https://files.pythonhosted.org/packages/07/e5/a6fc2f024eaef17c68b94b5f8f56098d31859a7840a11a3dd45da2d3908a/nanobind-2.13.0.tar.gz"
    URL_HASH SHA256=c7b04d6a6a4cd57985571e605539399b51331ae455d7fce576a5e2fcb89b1dcf
    TLS_VERIFY TRUE
    TIMEOUT 120
    INACTIVITY_TIMEOUT 30
    SOURCE_DIR "${_sl2cfoamxx_nanobind_source_dir}"
    BINARY_DIR "${_sl2cfoamxx_nanobind_binary_dir}"
    DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/third-party/downloads")
  FetchContent_MakeAvailable(sl2cfoamxx_nanobind)
else()
  # A fresh build directory can reuse the source tree without contacting the
  # network. An explicit source directory is likewise never downloaded over.
  add_subdirectory("${_sl2cfoamxx_nanobind_source_dir}"
                   "${_sl2cfoamxx_nanobind_binary_dir}" EXCLUDE_FROM_ALL)
endif()

if(NOT EXISTS "${_sl2cfoamxx_nanobind_source_dir}/ext/robin_map/include/tsl/robin_map.h")
  message(FATAL_ERROR
    "The nanobind source tree must include ext/robin_map. Use the nanobind "
    "source distribution or initialize its Git submodules.")
endif()
file(STRINGS "${_sl2cfoamxx_nanobind_source_dir}/include/nanobind/nanobind.h"
     _sl2cfoamxx_nanobind_version_definitions REGEX "^#define NB_VERSION_(MAJOR|MINOR|PATCH) ")
foreach(_component IN ITEMS MAJOR MINOR PATCH)
  string(REGEX MATCH "NB_VERSION_${_component} ([0-9]+)" _version_match
         "${_sl2cfoamxx_nanobind_version_definitions}")
  set(_sl2cfoamxx_nanobind_${_component} "${CMAKE_MATCH_1}")
endforeach()
set(_sl2cfoamxx_nanobind_actual_version
    "${_sl2cfoamxx_nanobind_MAJOR}.${_sl2cfoamxx_nanobind_MINOR}.${_sl2cfoamxx_nanobind_PATCH}")
if(NOT _sl2cfoamxx_nanobind_actual_version VERSION_EQUAL _sl2cfoamxx_nanobind_version)
  message(FATAL_ERROR
    "This wrapper requires nanobind ${_sl2cfoamxx_nanobind_version}; found "
    "${_sl2cfoamxx_nanobind_actual_version} in ${_sl2cfoamxx_nanobind_source_dir}.")
endif()

nanobind_add_module(_sl2cfoamxx NB_STATIC NOMINSIZE
  bindings.cpp
  tensor_bindings.cpp)
target_link_libraries(_sl2cfoamxx PRIVATE sl2cfoamxx::sl2cfoamxx)
target_compile_options(_sl2cfoamxx PRIVATE -Wno-psabi)

set(_sl2cfoamxx_python_package_dir "${CMAKE_CURRENT_BINARY_DIR}/sl2cfoamxx")
set_target_properties(_sl2cfoamxx PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${_sl2cfoamxx_python_package_dir}"
  BUILD_RPATH_USE_ORIGIN ON)
# Keep the import path stable for multi-configuration generators as well.
foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER "${_config}" _config_upper)
  set_target_properties(_sl2cfoamxx PROPERTIES
    "LIBRARY_OUTPUT_DIRECTORY_${_config_upper}" "${_sl2cfoamxx_python_package_dir}")
endforeach()

# Use a relative install location so --prefix and DESTDIR both work. The
# extension finds the one installed C++ library through a relative RPATH.
set(SL2CFOAMXX_PYTHON_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages"
    CACHE STRING "Python module installation directory relative to the installation prefix")
if(IS_ABSOLUTE "${SL2CFOAMXX_PYTHON_INSTALL_DIR}" OR
   IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
  message(FATAL_ERROR
    "Python bindings require relative SL2CFOAMXX_PYTHON_INSTALL_DIR and "
    "CMAKE_INSTALL_LIBDIR values for a relocatable installation.")
endif()
set(_sl2cfoamxx_python_install_package_dir
    "${SL2CFOAMXX_PYTHON_INSTALL_DIR}/sl2cfoamxx")
file(RELATIVE_PATH _sl2cfoamxx_python_relative_libdir
     "${CMAKE_INSTALL_PREFIX}/${_sl2cfoamxx_python_install_package_dir}"
     "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
if(APPLE)
  set(_sl2cfoamxx_python_origin "@loader_path")
else()
  set(_sl2cfoamxx_python_origin "$ORIGIN")
endif()
set_target_properties(_sl2cfoamxx PROPERTIES
  INSTALL_RPATH "${_sl2cfoamxx_python_origin}/${_sl2cfoamxx_python_relative_libdir}")
install(TARGETS _sl2cfoamxx
  LIBRARY DESTINATION "${_sl2cfoamxx_python_install_package_dir}")

# Pure Python files and optional maintained stubs accompany the extension in
# both the build and installation trees. configure_file tracks source edits.
file(GLOB _sl2cfoamxx_python_package_files CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/sl2cfoamxx/*.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/sl2cfoamxx/*.pyi"
  "${CMAKE_CURRENT_SOURCE_DIR}/sl2cfoamxx/py.typed")
foreach(_package_file IN LISTS _sl2cfoamxx_python_package_files)
  get_filename_component(_package_name "${_package_file}" NAME)
  configure_file("${_package_file}"
    "${_sl2cfoamxx_python_package_dir}/${_package_name}" COPYONLY)
endforeach()
install(FILES ${_sl2cfoamxx_python_package_files}
  DESTINATION "${_sl2cfoamxx_python_install_package_dir}")

if(BUILD_TESTING)
  execute_process(COMMAND "${Python_EXECUTABLE}" -c "import numpy"
    RESULT_VARIABLE _sl2cfoamxx_python_numpy_status
    OUTPUT_QUIET ERROR_QUIET)
  if(_sl2cfoamxx_python_numpy_status EQUAL 0)
    add_test(NAME python_public_api
      COMMAND "${CMAKE_COMMAND}" -E env
              "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
              "${Python_EXECUTABLE}"
              "${PROJECT_SOURCE_DIR}/tests/test_python_api.py")
    set_tests_properties(python_public_api PROPERTIES
      LABELS "cpu;python;public-api;integration"
      TIMEOUT 180)
    foreach(_backend IN ITEMS CPU GPU)
      string(TOLOWER "${_backend}" _backend_lower)
      if(_backend STREQUAL "GPU")
        set(_backend_label cuda)
      else()
        set(_backend_label cpu)
      endif()
      set(_test "python_progress_${_backend_lower}")
      add_test(NAME "${_test}"
        COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
                "${Python_EXECUTABLE}"
                "${PROJECT_SOURCE_DIR}/tests/test_python_api.py"
                progress --backend "${_backend}")
      set_tests_properties("${_test}" PROPERTIES
        LABELS "${_backend_label};python;progress;public-api;integration"
        SKIP_RETURN_CODE 77
        TIMEOUT 180)
      if(ENABLE_CUDA AND _backend STREQUAL "GPU")
        set_tests_properties("${_test}" PROPERTIES RESOURCE_LOCK sl2cfoamxx_gpu)
      endif()
    endforeach()
  else()
    message(STATUS "  Python API tests : skipped (NumPy unavailable for ${Python_EXECUTABLE})")
  endif()
endif()

message(STATUS "  Python executable: ${Python_EXECUTABLE}")
message(STATUS "  nanobind sources : ${_sl2cfoamxx_nanobind_source_dir}")
message(STATUS "  Python package   : ${_sl2cfoamxx_python_package_dir}")
message(STATUS "  Python install   : ${SL2CFOAMXX_PYTHON_INSTALL_DIR}")
