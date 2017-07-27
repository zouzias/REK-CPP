cmake_minimum_required(VERSION 3.8)
project(REK_CPP)

set(CMAKE_CXX_STANDARD 11)

set(EXECUTABLE_OUTPUT_PATH "bin")

set(SOURCE_FILES
        algorithms/REKSolver.cpp
        algorithms/REKSolver.hpp
        matrix/impl/DenseMatrix.cpp
        matrix/impl/DenseMatrix.hpp
        matrix/impl/SparseMatrix.cpp
        matrix/impl/SparseMatrix.hpp
        matrix/DoubleMatrix.cpp
        matrix/DoubleMatrix.hpp
        samplers/AliasSampler.cpp
        samplers/AliasSampler.hpp
        vector/impl/DenseVector.cpp
        vector/impl/DenseVector.hpp
        vector/impl/SparseVector.cpp
        vector/impl/SparseVector.hpp
        vector/DoubleVector.cpp
        vector/DoubleVector.hpp)

add_executable(rek_cpp ${SOURCE_FILES} main.cpp)
add_executable(test_dense ${SOURCE_FILES} testDense.cpp)
add_executable(test_sampler ${SOURCE_FILES} testSampler.cpp)
add_executable(test_sparse ${SOURCE_FILES} testSparse.cpp)
add_executable(test_vector ${SOURCE_FILES} testVector.cpp)
