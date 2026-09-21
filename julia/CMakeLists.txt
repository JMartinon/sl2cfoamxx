# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later


# Julia calls this private C-compatible bridge using its built-in foreign
# function interface. All scientific work remains in the public C++ target;
# neither CxxWrap nor Julia development headers are required.
find_program(Julia_EXECUTABLE NAMES julia DOC "Julia interpreter for the Julia API")
if(NOT Julia_EXECUTABLE)
  message(FATAL_ERROR "The Julia API requires Julia 1.10 or newer. Set -DJulia_EXECUTABLE=/path/to/julia.")
endif()
execute_process(COMMAND "${Julia_EXECUTABLE}" --startup-file=no --history-file=no
  -e "print(VERSION)"
  RESULT_VARIABLE _sl2cfoamxx_julia_status
  OUTPUT_VARIABLE _sl2cfoamxx_julia_version OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE _sl2cfoamxx_julia_error)
if(NOT _sl2cfoamxx_julia_status EQUAL 0 OR
   _sl2cfoamxx_julia_version VERSION_LESS 1.10)
  message(FATAL_ERROR "Julia 1.10 or newer is required: ${_sl2cfoamxx_julia_error}")
endif()

add_library(sl2cfoamxx_julia SHARED bridge.cpp)
target_link_libraries(sl2cfoamxx_julia PRIVATE sl2cfoamxx::sl2cfoamxx Threads::Threads)
target_compile_options(sl2cfoamxx_julia PRIVATE -Wno-psabi)

set(_sl2cfoamxx_julia_package_dir "${CMAKE_CURRENT_BINARY_DIR}/SL2CfoamXX")
set_target_properties(sl2cfoamxx_julia PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${_sl2cfoamxx_julia_package_dir}/lib"
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
  BUILD_RPATH_USE_ORIGIN ON)
foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER "${_config}" _config_upper)
  set_target_properties(sl2cfoamxx_julia PROPERTIES
    "LIBRARY_OUTPUT_DIRECTORY_${_config_upper}" "${_sl2cfoamxx_julia_package_dir}/lib")
endforeach()

set(SL2CFOAMXX_JULIA_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/julia" CACHE STRING
    "Julia package parent directory relative to the installation prefix")
if(IS_ABSOLUTE "${SL2CFOAMXX_JULIA_INSTALL_DIR}" OR IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
  message(FATAL_ERROR "Julia bindings require relative SL2CFOAMXX_JULIA_INSTALL_DIR and CMAKE_INSTALL_LIBDIR for relocation.")
endif()
set(_sl2cfoamxx_julia_install_package_dir "${SL2CFOAMXX_JULIA_INSTALL_DIR}/SL2CfoamXX")
file(RELATIVE_PATH _sl2cfoamxx_julia_relative_libdir
  "${CMAKE_INSTALL_PREFIX}/${_sl2cfoamxx_julia_install_package_dir}/lib"
  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
if(APPLE)
  set(_sl2cfoamxx_julia_origin "@loader_path")
else()
  set(_sl2cfoamxx_julia_origin "$ORIGIN")
endif()
set_target_properties(sl2cfoamxx_julia PROPERTIES
  INSTALL_RPATH "${_sl2cfoamxx_julia_origin}/${_sl2cfoamxx_julia_relative_libdir}")
install(TARGETS sl2cfoamxx_julia
  LIBRARY DESTINATION "${_sl2cfoamxx_julia_install_package_dir}/lib")

set(_sl2cfoamxx_julia_package_files
  Project.toml
  src/SL2CfoamXX.jl
  src/enums.jl
  src/ffi.jl
  src/boundary_angles.jl
  src/configuration.jl
  src/capabilities.jl
  src/tensor.jl
  src/computations.jl)
foreach(_relative IN LISTS _sl2cfoamxx_julia_package_files)
  set(_file "${CMAKE_CURRENT_SOURCE_DIR}/SL2CfoamXX/${_relative}")
  configure_file("${_file}" "${_sl2cfoamxx_julia_package_dir}/${_relative}" COPYONLY)
  get_filename_component(_directory "${_relative}" DIRECTORY)
  install(FILES "${_file}" DESTINATION "${_sl2cfoamxx_julia_install_package_dir}/${_directory}")
endforeach()

if(BUILD_TESTING)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/depot")
  foreach(_backend IN ITEMS CPU GPU)
    string(TOLOWER "${_backend}" _backend_lower)
    add_test(NAME "julia_public_api_${_backend_lower}"
      COMMAND "${CMAKE_COMMAND}" -E env
        "JULIA_LOAD_PATH=${CMAKE_CURRENT_BINARY_DIR}:@stdlib"
        "JULIA_DEPOT_PATH=${CMAKE_CURRENT_BINARY_DIR}/depot"
        "JULIA_NUM_THREADS=1"
        "${Julia_EXECUTABLE}" --startup-file=no --history-file=no
        "${PROJECT_SOURCE_DIR}/tests/test_julia_api.jl" "${_backend}")
    if(_backend STREQUAL "GPU")
      set(_labels "cuda;julia;public-api;integration;progress")
    else()
      set(_labels "cpu;julia;public-api;integration;progress")
    endif()
    set_tests_properties("julia_public_api_${_backend_lower}" PROPERTIES
      LABELS "${_labels}" SKIP_RETURN_CODE 77 TIMEOUT 240)
    if(ENABLE_CUDA AND _backend STREQUAL "GPU")
      set_tests_properties("julia_public_api_${_backend_lower}" PROPERTIES
        RESOURCE_LOCK sl2cfoamxx_gpu)
    endif()
  endforeach()
endif()

message(STATUS "  Julia executable : ${Julia_EXECUTABLE} (${_sl2cfoamxx_julia_version})")
message(STATUS "  Julia package    : ${_sl2cfoamxx_julia_package_dir}")
message(STATUS "  Julia install    : ${SL2CFOAMXX_JULIA_INSTALL_DIR}")
