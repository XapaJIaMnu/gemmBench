cmake_minimum_required(VERSION 3.5)
project(gemmBench)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_subdirectory(libs/intgemm)
add_subdirectory(libs/mkl-dnn)

include_directories(libs/intgemm)
include_directories(libs/dnnl/include)
include_directories(${CMAKE_BINARY_DIR}/libs/intgemm)
include_directories(libs/eigen/)

configure_file(${CMAKE_SOURCE_DIR}/gemmBench.sh ${CMAKE_BINARY_DIR}/gemmBench COPYONLY)

add_executable(bench bench.cpp libs/intgemm/intgemm.cc)
target_link_libraries(bench dnnl)
