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
# $> mkdir build
# $> cd build
# $> cmake ..
# $> make
#
# Thus you check out the xLearn project and build it in a
# subdirectory ``build". If you want further to install the
# built project, you can type the command:
#
# $> make install
#
cmake_minimum_required(VERSION 3.20)

project ("xLearn")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
# xlearn_api_shared links it into a shared library.
set_property(TARGET hwy PROPERTY POSITION_INDEPENDENT_CODE ON)

enable_testing()

#-------------------------------------------------------------------------------
# Take almost all warnings;
# Do not generate debug symbols;
# Optimization level 3;
# Optimization on architecture
#-------------------------------------------------------------------------------
if(NOT WIN32)
add_definitions("-Wall -Wno-sign-compare -O3
-march=native -Wno-strict-aliasing -Wno-comment")
else(WIN32)
add_definitions("/WX- /MT")
endif()

#-------------------------------------------------------------------------------
# Declare where our project will be installed.
#-------------------------------------------------------------------------------
set(CMAKE_INSTALL_PREFIX "xLearn")

#-------------------------------------------------------------------------------
# For Mac OSX
#-------------------------------------------------------------------------------
#set(CMAKE_MACOSX_RPATH TRUE)

#-------------------------------------------------------------------------------
# Set include paths and library paths.
#-------------------------------------------------------------------------------
include_directories(
  "${PROJECT_SOURCE_DIR}"
  "${PROJECT_BINARY_DIR}"
)

#-------------------------------------------------------------------------------
# Declaring a test or a benchmark.
#
# A test binary is one ctest entry rather than one entry per test case: the
# tests write scratch files into their working directory under fixed names, so
# the cases within a binary have to keep running in sequence.
#-------------------------------------------------------------------------------
function(xlearn_add_test name)
  add_executable(${name} ${name}.cc)
  target_link_libraries(${name} GTest::gtest_main ${ARGN})
  add_test(NAME ${name}
           COMMAND ${name}
           WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endfunction()

function(xlearn_add_benchmark name)
  add_executable(${name} ${name}.cc)
  target_link_libraries(${name} benchmark::benchmark ${ARGN})
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bench)
endfunction()

link_directories(
  "${PROJECT_BINARY_DIR}/src/base"
  "${PROJECT_BINARY_DIR}/src/data"
  "${PROJECT_BINARY_DIR}/src/reader"
  "${PROJECT_BINARY_DIR}/src/score"
  "${PROJECT_BINARY_DIR}/src/loss"
  "${PROJECT_BINARY_DIR}/src/solver"
  "${PROJECT_BINARY_DIR}/src/c_api"
)

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
add_subdirectory(src/distributed)
add_subdirectory(src/c_api)
add_subdirectory(python-package)
#add_subdirectory(R-package)
