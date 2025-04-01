cmake_minimum_required(VERSION 3.31)
project(NNN)

set(CMAKE_CXX_STANDARD 20)

add_library(NNN STATIC
        src/matrix.cpp
        include/matrix.hpp
        src/layer.cpp
        include/layer.hpp
)
target_include_directories(NNN PRIVATE include)

add_executable(test_matrix tests/test_matrix.cpp)
target_include_directories(test_matrix PRIVATE include external)
target_link_libraries(test_matrix PRIVATE NNN)

add_executable(test_layer tests/test_layer.cpp)
target_include_directories(test_layer PRIVATE include external)
target_link_libraries(test_layer PRIVATE NNN)
