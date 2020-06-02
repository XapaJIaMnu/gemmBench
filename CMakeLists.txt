cmake_minimum_required(VERSION 3.5)
project(gemmBench)

option(WITH_MKL "Include Intel MKL" OFF)

execute_process(COMMAND git submodule update --init --recursive --no-fetch
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
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

set(LIBRARIES
  dnnl
  fbgemm
  )

if(WITH_MKL)
  find_path(MKL_ROOT include/mkl.h
    PATHS $ENV{MKLROOT} /opt/intel/mkl
    DOC "Path to MKL root directory")

  # Find MKL includes.
  find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS ${MKL_ROOT}/include/)
  if(MKL_INCLUDE_DIR)
    message(STATUS "Found MKL include directory: ${MKL_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "MKL include directory not found")
  endif()

  # Find MKL libraries.
  find_library(MKL_CORE_LIBRARY NAMES mkl_core PATHS ${MKL_ROOT}/lib/intel64)
  if(MKL_CORE_LIBRARY)
    get_filename_component(MKL_LIBRARY_DIR ${MKL_CORE_LIBRARY} DIRECTORY)
    message(STATUS "Found MKL library directory: ${MKL_LIBRARY_DIR}")
  else()
    message(FATAL_ERROR "MKL library directory not found")
  endif()

  include_directories(${MKL_INCLUDE_DIR})
  add_definitions(-DWITH_MKL -DMKL_ILP64)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
  set(MKL_LIBRARIES
    ${MKL_LIBRARY_DIR}/libmkl_core.a
    ${MKL_LIBRARY_DIR}/libmkl_gnu_thread.a
    ${MKL_LIBRARY_DIR}/libmkl_intel_ilp64.a
    )
  list(APPEND LIBRARIES -Wl,--start-group ${MKL_LIBRARIES} -Wl,--end-group)
  list(APPEND LIBRARIES gomp)
endif()

add_executable(bench bench.cpp libs/intgemm/intgemm.cc)
target_link_libraries(bench ${LIBRARIES})
