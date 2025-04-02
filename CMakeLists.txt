cmake_minimum_required(VERSION 3.31)
project(NNN)

set(CMAKE_CXX_STANDARD 20)

add_library(NNN STATIC
        src/matrix.cpp
        include/matrix.hpp
        src/layer.cpp
        include/layer.hpp
        src/neural_network.cpp
        include/neural_network.hpp
        src/activation_function.cpp
        include/activation_function.hpp
        src/loss_function.cpp
        include/loss_function.hpp
)
target_include_directories(NNN PRIVATE include)

add_executable(test_matrix tests/test_matrix.cpp)
target_include_directories(test_matrix PRIVATE include external)
target_link_libraries(test_matrix PRIVATE NNN)

add_executable(test_layer tests/test_layer.cpp)
target_include_directories(test_layer PRIVATE include external)
target_link_libraries(test_layer PRIVATE NNN)

add_executable(test_neural_network tests/test_neural_network.cpp)
target_include_directories(test_neural_network PRIVATE include external)
target_link_libraries(test_neural_network PRIVATE NNN)

add_executable(test_activation_function tests/test_activation_function.cpp)
target_include_directories(test_activation_function PRIVATE include external)
target_link_libraries(test_activation_function PRIVATE NNN)

add_executable(test_loss_function tests/test_loss_function.cpp)
target_include_directories(test_loss_function PRIVATE include external)
target_link_libraries(test_loss_function PRIVATE NNN)
