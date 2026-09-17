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
    src/detail/dnnl_layout.cpp
    src/tensor.cpp
    src/ops.cpp
    src/nn/module.cpp
    src/nn/linear.cpp
    src/nn/relu.cpp
    src/nn/sigmoid.cpp
    src/nn/leaky_relu.cpp
    src/nn/gelu.cpp
    src/nn/softmax.cpp
    src/nn/sequential.cpp
    src/nn/loss.cpp
    src/nn/conv2d.cpp
    src/nn/max_pool2d.cpp
    src/nn/avg_pool2d.cpp
    src/nn/flatten.cpp
    src/nn/batch_norm2d.cpp
    src/nn/batch_norm1d.cpp
    src/nn/global_avg_pool2d.cpp
    src/nn/global_max_pool2d.cpp
    src/nn/dropout.cpp
    src/nn/tanh.cpp
    src/nn/identity.cpp
    src/nn/prelu.cpp
    src/nn/elu.cpp
    src/nn/selu.cpp
    src/nn/softplus.cpp
    src/nn/swish.cpp
    src/nn/mish.cpp
    src/nn/hardtanh.cpp
    src/nn/hardsigmoid.cpp
    src/optim/optimizer.cpp
    src/optim/sgd.cpp
    src/optim/momentum.cpp
    src/optim/nesterov.cpp
    src/optim/adagrad.cpp
    src/optim/adadelta.cpp
    src/optim/rmsprop.cpp
    src/optim/muon.cpp
    src/optim/adam.cpp
    src/optim/adamw.cpp
    src/datasets/mnist.cpp
    src/metrics/accuracy.cpp
)
target_include_directories(advanceml PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_include_directories(advanceml PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(advanceml PUBLIC cxx_std_23)

# Lets GCC vectorize loops that call sqrt/exp/log (AdamW, softmax) by dropping errno updates;
# unlike -ffast-math it keeps IEEE semantics (NaN/Inf handling, no reassociation).
target_compile_options(advanceml PRIVATE -fno-math-errno)

if(ADVANCEML_NATIVE_ARCH)
    target_compile_options(advanceml PUBLIC -march=native)
endif()

find_package(dnnl CONFIG REQUIRED)
target_link_libraries(advanceml PUBLIC DNNL::dnnl)

if(ADVANCEML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

add_executable(train_cnn_mnist examples/train_cnn_mnist.cpp)
target_link_libraries(train_cnn_mnist PRIVATE advanceml)
