# LibraryLink uses an SDK supplied with the user's Wolfram installation.
# No Wolfram runtime or proprietary headers are copied into this repository.
set(Wolfram_ROOT "" CACHE PATH "Wolfram installation root (or macOS app bundle)")
file(GLOB _sl2cfoamxx_wolfram_candidates LIST_DIRECTORIES TRUE
  "/Applications/Wolfram*.app" "/Applications/Mathematica*.app"
  "/usr/local/Wolfram/Wolfram/*" "/usr/local/Wolfram/Mathematica/*"
  "/usr/local/Wolfram/WolframEngine/*")
list(SORT _sl2cfoamxx_wolfram_candidates COMPARE NATURAL ORDER DESCENDING)
set(_sl2cfoamxx_wolfram_hints "${Wolfram_ROOT}" "$ENV{Wolfram_ROOT}"
  ${_sl2cfoamxx_wolfram_candidates})
find_path(WolframLibrary_INCLUDE_DIR WolframLibrary.h
  HINTS ${_sl2cfoamxx_wolfram_hints}
  PATH_SUFFIXES SystemFiles/IncludeFiles/C Contents/SystemFiles/IncludeFiles/C
  DOC "Directory containing the installed WolframLibrary.h SDK header")
if(NOT WolframLibrary_INCLUDE_DIR)
  message(FATAL_ERROR "The Mathematica API needs the installed LibraryLink SDK. Set -DWolfram_ROOT=/path/to/Wolfram or -DWolframLibrary_INCLUDE_DIR=/path/to/IncludeFiles/C.")
endif()
find_program(Wolfram_EXECUTABLE NAMES WolframKernel MathKernel
  HINTS ${_sl2cfoamxx_wolfram_hints}
  PATH_SUFFIXES Executables Contents/MacOS
  DOC "Licensed Wolfram kernel used to run Mathematica integration tests")

# Building requires headers, but not kernel activation. Runtime tests require
# a valid license and report a kernel startup failure instead of a false pass.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
  set(_sl2cfoamxx_system_id MacOSX-ARM64)
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  set(_sl2cfoamxx_system_id MacOSX-x86-64)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  set(_sl2cfoamxx_system_id Linux-x86-64)
else()
  set(_sl2cfoamxx_system_id "")
endif()
set(SL2CFOAMXX_WOLFRAM_SYSTEM_ID "${_sl2cfoamxx_system_id}" CACHE STRING
    "Wolfram $SystemID for the target platform")
if(NOT SL2CFOAMXX_WOLFRAM_SYSTEM_ID)
  message(FATAL_ERROR "Set SL2CFOAMXX_WOLFRAM_SYSTEM_ID to the target kernel's $SystemID.")
endif()

add_library(sl2cfoamxx_mathematica SHARED bridge.cpp)
target_link_libraries(sl2cfoamxx_mathematica PRIVATE sl2cfoamxx::sl2cfoamxx Threads::Threads)
target_include_directories(sl2cfoamxx_mathematica SYSTEM PRIVATE "${WolframLibrary_INCLUDE_DIR}")
target_compile_options(sl2cfoamxx_mathematica PRIVATE -Wno-psabi)
set(_sl2cfoamxx_mathematica_package "${CMAKE_CURRENT_BINARY_DIR}/SL2CfoamXX")
set(_sl2cfoamxx_mathematica_resources "LibraryResources/${SL2CFOAMXX_WOLFRAM_SYSTEM_ID}")
set_target_properties(sl2cfoamxx_mathematica PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${_sl2cfoamxx_mathematica_package}/${_sl2cfoamxx_mathematica_resources}"
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON BUILD_RPATH_USE_ORIGIN ON)
foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER "${_config}" _config_upper)
  set_target_properties(sl2cfoamxx_mathematica PROPERTIES
    "LIBRARY_OUTPUT_DIRECTORY_${_config_upper}" "${_sl2cfoamxx_mathematica_package}/${_sl2cfoamxx_mathematica_resources}")
endforeach()

set(SL2CFOAMXX_MATHEMATICA_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/wolfram/Applications" CACHE STRING
    "Mathematica package parent directory relative to the installation prefix")
if(IS_ABSOLUTE "${SL2CFOAMXX_MATHEMATICA_INSTALL_DIR}" OR IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
  message(FATAL_ERROR "Mathematica bindings require relative package and library install directories for relocation.")
endif()
set(_sl2cfoamxx_mathematica_install_package "${SL2CFOAMXX_MATHEMATICA_INSTALL_DIR}/SL2CfoamXX")
file(RELATIVE_PATH _sl2cfoamxx_mathematica_relative_libdir
  "${CMAKE_INSTALL_PREFIX}/${_sl2cfoamxx_mathematica_install_package}/${_sl2cfoamxx_mathematica_resources}"
  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
if(APPLE)
  set(_sl2cfoamxx_mathematica_origin "@loader_path")
else()
  set(_sl2cfoamxx_mathematica_origin "$ORIGIN")
endif()
set_target_properties(sl2cfoamxx_mathematica PROPERTIES
  INSTALL_RPATH "${_sl2cfoamxx_mathematica_origin}/${_sl2cfoamxx_mathematica_relative_libdir}")
install(TARGETS sl2cfoamxx_mathematica
  LIBRARY DESTINATION "${_sl2cfoamxx_mathematica_install_package}/${_sl2cfoamxx_mathematica_resources}")

set(_sl2cfoamxx_mathematica_files Kernel/init.m Kernel/SL2CfoamXX.wl Kernel/Tensor.wl)
foreach(_relative IN LISTS _sl2cfoamxx_mathematica_files)
  set(_file "${CMAKE_CURRENT_SOURCE_DIR}/SL2CfoamXX/${_relative}")
  configure_file("${_file}" "${_sl2cfoamxx_mathematica_package}/${_relative}" COPYONLY)
  get_filename_component(_directory "${_relative}" DIRECTORY)
  install(FILES "${_file}" DESTINATION "${_sl2cfoamxx_mathematica_install_package}/${_directory}")
endforeach()

if(BUILD_TESTING)
  if(Wolfram_EXECUTABLE)
    foreach(_backend IN ITEMS CPU GPU)
      string(TOLOWER "${_backend}" _backend_lower)
      add_test(NAME "mathematica_public_api_${_backend_lower}"
        COMMAND "${CMAKE_COMMAND}" -E env
          "SL2CFOAMXX_MATHEMATICA_PACKAGE_DIR=${_sl2cfoamxx_mathematica_package}"
          "SL2CFOAMXX_MATHEMATICA_TEST_MODE=api"
          "SL2CFOAMXX_MATHEMATICA_TEST_BACKEND=${_backend}"
          "${Wolfram_EXECUTABLE}" -noinit -noprompt -script
          "${PROJECT_SOURCE_DIR}/tests/test_mathematica_api.wl")
      if(_backend STREQUAL "GPU")
        set(_labels "cuda;mathematica;public-api;integration;progress")
      else()
        set(_labels "cpu;mathematica;public-api;integration;progress")
      endif()
      set_tests_properties("mathematica_public_api_${_backend_lower}" PROPERTIES
        LABELS "${_labels}" SKIP_RETURN_CODE 77 TIMEOUT 300)
      if(ENABLE_CUDA AND _backend STREQUAL "GPU")
        set_tests_properties("mathematica_public_api_${_backend_lower}" PROPERTIES RESOURCE_LOCK sl2cfoamxx_gpu)
      endif()
    endforeach()
    add_test(NAME mathematica_library_lifecycle
      COMMAND "${CMAKE_COMMAND}" -E env
        "SL2CFOAMXX_MATHEMATICA_PACKAGE_DIR=${_sl2cfoamxx_mathematica_package}"
        "SL2CFOAMXX_MATHEMATICA_TEST_MODE=lifecycle"
        "${Wolfram_EXECUTABLE}" -noinit -noprompt -script
        "${PROJECT_SOURCE_DIR}/tests/test_mathematica_api.wl")
    set_tests_properties(mathematica_library_lifecycle PROPERTIES
      LABELS "cpu;mathematica;public-api;integration;lifecycle" TIMEOUT 90)
  else()
    message(WARNING "Wolfram kernel not found; Mathematica runtime tests are not registered. Set Wolfram_EXECUTABLE to a licensed kernel.")
  endif()
endif()
message(STATUS "  LibraryLink SDK  : ${WolframLibrary_INCLUDE_DIR}")
message(STATUS "  Wolfram kernel   : ${Wolfram_EXECUTABLE}")
message(STATUS "  Wolfram SystemID : ${SL2CFOAMXX_WOLFRAM_SYSTEM_ID}")
message(STATUS "  Mathematica pkg  : ${_sl2cfoamxx_mathematica_package}")
