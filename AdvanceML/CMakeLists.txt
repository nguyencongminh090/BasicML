cmake_minimum_required(VERSION 3.21)
project(AdvanceML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(ADVANCEML_NATIVE_ARCH "Build with -march=native to enable AVX-512 VNNI on this machine" ON)
option(ADVANCEML_BUILD_TESTS "Build the Catch2 test suite" ON)

add_library(advanceml
    src/version.cpp
)
target_include_directories(advanceml PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(advanceml PUBLIC cxx_std_23)

if(ADVANCEML_NATIVE_ARCH)
    target_compile_options(advanceml PUBLIC -march=native)
endif()

find_package(dnnl CONFIG REQUIRED)
target_link_libraries(advanceml PUBLIC DNNL::dnnl)

if(ADVANCEML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
