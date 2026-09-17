include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")

add_executable(advanceml_tests
    smoke_test.cpp
    autograd_test.cpp
    autograd_engine_test.cpp
    nn_test.cpp
    optim_test.cpp
    cnn_test.cpp
    datasets_test.cpp
    metrics_test.cpp
    loss_parity_test.cpp
    optim_parity_test.cpp
    layer_parity_test.cpp
    activation_parity_test.cpp
)
target_link_libraries(advanceml_tests PRIVATE advanceml Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(advanceml_tests)
