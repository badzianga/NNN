cmake_minimum_required(VERSION 3.31)
project(NNN)

set(CMAKE_CXX_STANDARD 20)

add_executable(NNN main.cpp
        src/matrix.cpp
        include/matrix.hpp
        src/layer.cpp
        include/layer.hpp
)
target_include_directories(NNN PRIVATE include)

add_executable(test_matrix tests/test_matrix.cpp
        src/matrix.cpp
        include/matrix.hpp
)
target_include_directories(test_matrix PRIVATE include external)

add_executable(test_layer tests/test_layer.cpp
        src/matrix.cpp
        include/matrix.hpp
        src/layer.cpp
        include/layer.hpp
)
target_include_directories(test_layer PRIVATE include external)
