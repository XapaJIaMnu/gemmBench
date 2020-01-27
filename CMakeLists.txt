cmake_minimum_required(VERSION 3.5)
project(gemmBench)

execute_process(COMMAND git submodule update --init --recursive --no-fetch
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(DNNL_BUILD_TESTS OFF CACHE BOOL "Disable dnnl tests")
set(INTGEMM_DONT_BUILD_TESTS ON CACHE BOOL "Disable intgemm tests")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_subdirectory(libs/intgemm)
add_subdirectory(libs/mkl-dnn)

#FBGEMM
set(FBGEMM_BUILD_TESTS OFF CACHE BOOL "Disable fbgemm tests")
set(FBGEMM_BUILD_BENCHMARKS OFF CACHE BOOL "Disable fbgemm benchmark")
add_subdirectory(libs/FBGEMM)

include_directories(libs/intgemm)
include_directories(libs/dnnl/include)
include_directories(${CMAKE_BINARY_DIR}/libs/intgemm)
include_directories(libs/eigen/)
include_directories(libs/FBGEMM/include)

configure_file(${CMAKE_SOURCE_DIR}/gemmBench.sh ${CMAKE_BINARY_DIR}/gemmBench COPYONLY)

add_executable(bench bench.cpp libs/intgemm/intgemm.cc)
target_link_libraries(bench dnnl fbgemm)
