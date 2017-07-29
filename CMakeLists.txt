cmake_minimum_required(VERSION 3.8)
project(REK_CPP)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH "bin")

set(SOURCE_FILES
        algorithms/REKSolver.hpp
        samplers/AliasSampler.hpp)

add_executable(rek_cpp ${SOURCE_FILES} main.cpp)
add_executable(test_dense ${SOURCE_FILES} testDense.cpp)
add_executable(test_sampler ${SOURCE_FILES} testSampler.cpp)
add_executable(test_sparse ${SOURCE_FILES} testSparse.cpp)
