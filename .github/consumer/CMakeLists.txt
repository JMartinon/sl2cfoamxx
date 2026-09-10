# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.22)
project(sl2cfoamxx_installed_consumer LANGUAGES CXX)

# This project deliberately has no add_subdirectory() of the library and no
# private source/header paths. It consumes only its installed CMake target.
find_package(sl2cfoamxx 1.0 CONFIG REQUIRED)
enable_testing()

foreach(example IN ITEMS simple_example coherent_boundary_example)
  add_executable("${example}" "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/${example}.cpp")
  target_link_libraries("${example}" PRIVATE sl2cfoamxx::sl2cfoamxx)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${example}-data")
  add_test(NAME "${example}" COMMAND "${example}")
  set_tests_properties("${example}" PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${example}-data"
    TIMEOUT 180)
endforeach()
