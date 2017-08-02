cmake_minimum_required(VERSION 2.8)
project(REK_CPP)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH "bin")

set(SOURCE_FILES
        src/algorithms/RekSolver.hpp
        src/samplers/AliasSampler.hpp)

add_executable(rek_cpp ${SOURCE_FILES} src/main.cpp)
add_executable(test_dense ${SOURCE_FILES} src/test_dense.cpp)
add_executable(test_sampler ${SOURCE_FILES} src/test_sampler.cpp)
add_executable(test_sparse ${SOURCE_FILES} src/test_sparse.cpp)
add_executable(test_sparse_colmajor ${SOURCE_FILES} src/test_sparse_colmajor.cpp)
