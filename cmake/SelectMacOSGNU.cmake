# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

# Compiler selection must precede project(), which otherwise finds AppleClang.
function(sl2cfoamxx_select_macos_gnu)
  if(NOT CMAKE_HOST_APPLE OR DEFINED CMAKE_CXX_COMPILER
     OR NOT "$ENV{CXX}" STREQUAL ""
     OR CMAKE_TOOLCHAIN_FILE OR NOT "$ENV{CMAKE_TOOLCHAIN_FILE}" STREQUAL ""
     OR CMAKE_PROJECT_INCLUDE_BEFORE OR CMAKE_PROJECT_sl2cfoamxx_INCLUDE_BEFORE
     OR CMAKE_PROJECT_TOP_LEVEL_INCLUDES)
    return()
  endif()
  if(DEFINED CMAKE_SYSTEM_NAME AND NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    return()
  endif()

  # Respect the Homebrew selected on PATH, including a nonstandard prefix.
  # PATHS supplies the usual installations when brew is not on PATH.
  find_program(_brew NAMES brew PATHS /opt/homebrew/bin /usr/local/bin
    NO_CMAKE_FIND_ROOT_PATH NO_CACHE)
  set(_prefixes)
  if(_brew)
    execute_process(COMMAND "${_brew}" --prefix
      RESULT_VARIABLE _brew_result OUTPUT_VARIABLE _brew_prefix
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET TIMEOUT 10)
    if(_brew_result EQUAL 0 AND IS_ABSOLUTE "${_brew_prefix}")
      list(APPEND _prefixes "${_brew_prefix}")
    endif()
  endif()
  list(APPEND _prefixes /opt/homebrew /usr/local)
  list(REMOVE_DUPLICATES _prefixes)

  foreach(_prefix IN LISTS _prefixes)
    # opt paths follow the active formula; gcc@* may be keg-only (not in bin).
    file(GLOB _default_candidates LIST_DIRECTORIES FALSE
      "${_prefix}/opt/gcc/bin/g++-[0-9]*")
    file(GLOB _versioned_candidates LIST_DIRECTORIES FALSE
      "${_prefix}/opt/gcc@*/bin/g++-[0-9]*")
    file(GLOB _linked_candidates LIST_DIRECTORIES FALSE
      "${_prefix}/bin/g++-[0-9]*")
    set(_candidates ${_default_candidates} ${_versioned_candidates} ${_linked_candidates})
    set(_best_compiler "")
    set(_best_major -1)
    foreach(_candidate IN LISTS _candidates)
      get_filename_component(_name "${_candidate}" NAME)
      if(NOT _name MATCHES "^g\\+\\+-[0-9]+$")
        continue()
      endif()
      # Clang also defines __GNUC__; never accept that compatibility alias.
      execute_process(COMMAND "${_candidate}" -dM -E -x c++ /dev/null
        RESULT_VARIABLE _probe_result OUTPUT_VARIABLE _macros
        ERROR_QUIET TIMEOUT 10)
      if(_probe_result EQUAL 0 AND NOT _macros MATCHES "#define __clang__ "
         AND _macros MATCHES "#define __GNUC__ ([0-9]+)")
        set(_major "${CMAKE_MATCH_1}")
        if(_major GREATER _best_major)
          set(_best_compiler "${_candidate}")
          set(_best_major "${_major}")
        endif()
      endif()
    endforeach()
    if(_best_compiler)
      set(CMAKE_CXX_COMPILER "${_best_compiler}" CACHE FILEPATH "C++ compiler")
      message(STATUS "Using Homebrew GNU C++: ${_best_compiler}")
      return()
    endif()
  endforeach()

  message(FATAL_ERROR
    "No Homebrew GNU C++ compiler was found. Install it with 'brew install gcc', "
    "then configure a fresh build directory. To use another GNU installation, "
    "set -DCMAKE_CXX_COMPILER=/path/to/g++ or CXX. Apple's /usr/bin/g++ is Clang "
    "and is not supported.")
endfunction()

sl2cfoamxx_select_macos_gnu()
