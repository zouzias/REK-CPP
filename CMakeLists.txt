cmake_minimum_required(VERSION 3.0)
project(rek_cpp)

# enable c++ language
enable_language(CXX)

# project version
set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)
set(VERSION_PATCH 0)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH "bin")

FIND_PACKAGE (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(src)

set(SOURCE_FILES
        src/algorithms/Solver.hpp
        src/samplers/AliasSampler.hpp)

add_executable(rek_cpp ${SOURCE_FILES} src/main.cpp)
target_link_libraries (rek_cpp Eigen3::Eigen)

# Test executables
add_executable(test_dense ${SOURCE_FILES} test/test_dense.cpp)
target_link_libraries (test_dense Eigen3::Eigen)
add_executable(test_sampler ${SOURCE_FILES} test/test_sampler.cpp)
target_link_libraries (test_sampler Eigen3::Eigen)
add_executable(test_sparse ${SOURCE_FILES} test/test_sparse.cpp)
target_link_libraries (test_sparse Eigen3::Eigen)
add_executable(test_sparse_colmajor ${SOURCE_FILES} test/test_sparse_colmajor.cpp)
target_link_libraries (test_sparse_colmajor Eigen3::Eigen)


# activate ctest
include(CTest)
enable_testing()

add_test(test_sparse ${PROJECT_BINARY_DIR}/bin/test_sparse)
add_test(test_sampler ${PROJECT_BINARY_DIR}/bin/test_sampler)
add_test(test_sparse_colmajor ${PROJECT_BINARY_DIR}/bin/test_sparse_colmajor)
add_test(test_dense ${PROJECT_BINARY_DIR}/bin/test_dense)