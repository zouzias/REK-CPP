cmake_minimum_required(VERSION 2.8)
project(REK_CPP)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH "bin")

set(SOURCE_FILES
        algorithms/RekSolver.hpp
        samplers/AliasSampler.hpp)

add_executable(rek_cpp ${SOURCE_FILES} main.cpp)
add_executable(test_dense ${SOURCE_FILES} test_dense.cpp)
add_executable(test_sampler ${SOURCE_FILES} test_sampler.cpp)
add_executable(test_sparse ${SOURCE_FILES} test_sparse.cpp)
add_executable(test_sparse_colmajor ${SOURCE_FILES} test_sparse_colmajor.cpp)
