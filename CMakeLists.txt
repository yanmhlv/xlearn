# Copyright (c) 2018 by contributors.
# Authour: Chao Ma (mctt90@gmail.com)
#
# This is the root CMakeLists.txt file of the xLearn project, which
# build all subdirectories (packages) in the order of
# inter-package dependence.
#
# you can build the xLearn using the following commands:
#
# $> cd xLearn
# $> cmake --preset dev
# $> cmake --build --preset dev
#
# The configurations on offer are listed by ``cmake --list-presets``. To
# install the tree that a preset built:
#
# $> cmake --install build/dev
#
cmake_minimum_required(VERSION 3.23)

project ("xLearn" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(XLEARN_BUILD_TESTS "Build the tests and benchmarks" ON)
option(XLEARN_BUILD_PYTHON "Build the xlearn._core Python extension" OFF)
option(XLEARN_NATIVE_ARCH "Tune the build for the building machine" ON)

include(GNUInstallDirs)

if(XLEARN_BUILD_PYTHON)
  # The static libraries below are linked into the extension module.
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

#-------------------------------------------------------------------------------
# The dependencies below carry no compile flags of ours, so without a build
# type they would be built unoptimized -- which Google Benchmark reports as
# "Library was built as DEBUG" and which taints every timing it produces.
#-------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

#-------------------------------------------------------------------------------
# Third-party dependencies. They are declared before our own compile flags
# below so that each is built with its own: Highway in particular relies on
# choosing its instruction sets itself.
#-------------------------------------------------------------------------------
include(FetchContent)

find_package(Threads REQUIRED)

# The portable SIMD layer behind src/base/simd.h.
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG 1.4.0
  GIT_SHALLOW TRUE
)

if(XLEARN_BUILD_TESTS)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.18.0
    GIT_SHALLOW TRUE
  )

  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
    GIT_SHALLOW TRUE
  )

  FetchContent_MakeAvailable(highway googletest benchmark)
  enable_testing()
else()
  FetchContent_MakeAvailable(highway)
endif()

# The extension module links it into a shared library.
set_property(TARGET hwy PROPERTY POSITION_INDEPENDENT_CODE ON)

#-------------------------------------------------------------------------------
# What every xLearn target compiles with: the project root on the include path,
# because headers are included by their path from there, and almost all
# warnings.
#
# -march=native bakes in whatever the building machine supports, so a wheel
# built with it crashes anywhere older. Distributable builds turn it off.
#-------------------------------------------------------------------------------
add_library(xlearn_flags INTERFACE)
add_library(xlearn::flags ALIAS xlearn_flags)

target_include_directories(xlearn_flags INTERFACE "${PROJECT_SOURCE_DIR}")

if(NOT MSVC)
  target_compile_options(xlearn_flags INTERFACE
    -Wall -Wno-sign-compare -Wno-strict-aliasing -Wno-comment)
  if(XLEARN_NATIVE_ARCH)
    target_compile_options(xlearn_flags INTERFACE -march=native)
  endif()
endif()

#-------------------------------------------------------------------------------
# Declare where our project will be installed. A wheel build gets its prefix
# from the build backend instead.
#-------------------------------------------------------------------------------
if(NOT XLEARN_BUILD_PYTHON AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/xLearn"
      CACHE PATH "Install path prefix" FORCE)
endif()

#-------------------------------------------------------------------------------
# Declaring a package: a static library named after the directory holding it,
# reachable from every other package as xlearn::<name> and carrying the
# headers of that directory along, so that dependents and the install tree
# both find them.
#
# A wheel holds the extension module alone, so it installs nothing from here.
#-------------------------------------------------------------------------------
function(xlearn_add_library name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;DEPS")

  add_library(xlearn_${name} STATIC ${arg_SOURCES})
  add_library(xlearn::${name} ALIAS xlearn_${name})
  target_link_libraries(xlearn_${name} PUBLIC xlearn::flags ${arg_DEPS})

  file(GLOB headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
  target_sources(xlearn_${name} PUBLIC
    FILE_SET HEADERS BASE_DIRS "${PROJECT_SOURCE_DIR}" FILES ${headers})

  if(NOT XLEARN_BUILD_PYTHON)
    install(TARGETS xlearn_${name} FILE_SET HEADERS)
  endif()
endfunction()

#-------------------------------------------------------------------------------
# Declaring a test or a benchmark.
#
# A test binary is one ctest entry rather than one entry per test case: the
# tests write scratch files into their working directory under fixed names, so
# the cases within a binary have to keep running in sequence.
#-------------------------------------------------------------------------------
function(xlearn_add_test name)
  if(NOT XLEARN_BUILD_TESTS)
    return()
  endif()
  get_filename_component(package "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  add_executable(${name} ${name}.cc)
  target_link_libraries(${name} PRIVATE GTest::gtest_main ${ARGN})
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/test/${package}")
  add_test(NAME ${name}
           COMMAND ${name}
           WORKING_DIRECTORY "$<TARGET_FILE_DIR:${name}>")
endfunction()

function(xlearn_add_benchmark name)
  if(NOT XLEARN_BUILD_TESTS)
    return()
  endif()
  add_executable(${name} ${name}.cc)
  target_link_libraries(${name} PRIVATE benchmark::benchmark ${ARGN})
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bench")
endfunction()

#-------------------------------------------------------------------------------
# Declare packages in xLearn project.
#-------------------------------------------------------------------------------
add_subdirectory(demo)
add_subdirectory(scripts)
add_subdirectory(src/base)
add_subdirectory(src/data)
add_subdirectory(src/reader)
add_subdirectory(src/score)
add_subdirectory(src/loss)
add_subdirectory(src/solver)
if(XLEARN_BUILD_PYTHON)
add_subdirectory(src/python)
endif()
