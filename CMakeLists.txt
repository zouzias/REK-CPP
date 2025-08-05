cmake_minimum_required(VERSION 3.10)
project(rek_cpp)

enable_language(CXX)

# project version
set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)
set(VERSION_PATCH 0)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH "bin")

FIND_PACKAGE (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(src)

add_executable(rek_cpp src/main.cpp)
target_link_libraries (rek_cpp Eigen3::Eigen)

add_executable(test_dense  test/test_dense.cpp)
target_link_libraries (test_dense Eigen3::Eigen)

add_executable(test_sampler test/test_sampler.cpp)
target_link_libraries (test_sampler Eigen3::Eigen)

add_executable(test_sparse  test/test_sparse.cpp)
target_link_libraries (test_sparse Eigen3::Eigen)

add_executable(test_sparse_colmajor test/test_sparse_colmajor.cpp)
target_link_libraries (test_sparse_colmajor Eigen3::Eigen)

include (CTest)
# Setup valgrind
set(MemoryCheckCommand /usr/bin/valgrind)
set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --leak-check=full")
set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --track-fds=yes")
set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --trace-children=yes")
set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --track-origins=yes")
set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --error-exitcode=1")
enable_testing()

add_test(test_sparse ${PROJECT_BINARY_DIR}/bin/test_sparse)
add_test(test_sampler ${PROJECT_BINARY_DIR}/bin/test_sampler)
add_test(test_sparse_colmajor ${PROJECT_BINARY_DIR}/bin/test_sparse_colmajor)
add_test(test_dense ${PROJECT_BINARY_DIR}/bin/test_dense)