#pragma once

#include<iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "../samplers/AliasSampler.hpp"

using namespace Eigen;
typedef Matrix<double, Dynamic, 1> RowVector;

class RekSolver {
    private:

    RowVector solve(SparseMatrix<double, RowMajor> &A,
                                     SparseMatrix<double, ColMajor> &AColMajor,
                                     const Matrix<double, Dynamic, 1> &b,
                                     long MaxIterations) const {
        double val;
        int i_k, j_k;
        RowVector x(A.cols());
        RowVector z(b);
        RowVector rowNorms(A.rows());
        RowVector columnNorms(A.cols());

        x.setZero();

        for (int i = 0; i < A.rows(); i++)
            rowNorms(i) = A.row(i).squaredNorm();

        for (int j = 0; j < AColMajor.cols(); j++)
            columnNorms(j) = AColMajor.col(j).squaredNorm();

        AliasSampler rowSampler(rowNorms);
        AliasSampler colSampler(columnNorms);

        // Initialize Alias samplers, O(n)
        rowSampler.initSampler();
        colSampler.initSampler();

        for (int k = 0; k < MaxIterations; k++) {

            i_k = rowSampler.walkerSample();
            j_k = colSampler.walkerSample();

            val = -AColMajor.col(j_k).dot(z) / columnNorms(j_k);
            z += val * AColMajor.col(j_k);
            val = A.row(i_k).dot(x);
            val = (b(i_k) - z(i_k) - val) / rowNorms(i_k);

            // TODO: .toDense() seems to be required here!
            // FIXME: replace .toDense() with iteration over spark i_k row
            x += val * A.row(i_k).toDense();
        }

        return x;
    }

    public:
        RekSolver() = default;


        RowVector solve(Matrix<double, Dynamic, Dynamic> &A,
                                         const RowVector &b,
                                         double MaxSeconds) const {
            Matrix<double, Dynamic, 1> vector(A.cols());
            return vector;
        }

        /**
         * Returns the solution to Ax=b using the Randomized Extended Kaczmarz method
         *
         * @param A Input dense matrix
         * @param b Right hand side vector
         * @param MaxIterations Maximum number of iterations
         * @return Returns an approximate solution to ||Ax - b||_2
         */
        RowVector solve(Matrix<double, Dynamic, Dynamic> &A,
                                         const RowVector &b,
                                         long MaxIterations) const {
            double val;
            long i_k, j_k;
            RowVector x(A.cols());
            RowVector z(b);
            RowVector rowNorms(A.rows());
            RowVector columnNorms(A.cols());

            x.setZero();
            for (int i = 0; i < A.rows(); i++)
                rowNorms(i) = A.row(i).squaredNorm();

            for (int j = 0; j < A.cols(); j++)
                columnNorms(j) = A.col(j).squaredNorm();

            AliasSampler rowSampler(rowNorms);
            AliasSampler colSampler(columnNorms);

            // Initialize Alias samplers, O(n)
            rowSampler.initSampler();
            colSampler.initSampler();

            for (int k = 0; k < MaxIterations; k++) {
                i_k = rowSampler.walkerSample();
                j_k = colSampler.walkerSample();

                // Extended Kaczmarz
                // i_k = k % A.rows();
                // j_k = k % A.cols();

                val = -z.dot(A.col(j_k)) / columnNorms(j_k);

                z += val * A.col(j_k);

                val = x.dot(A.row(i_k));
                val = (b(i_k) - z(i_k) - val) / rowNorms(i_k);

                x += val * A.row(i_k);
            }
            return x;
        }

    /**
    * Returns the solution to Ax=b using the Randomized Extended Kaczmarz method
    *
    * @param A Input sparse matrix in row major format
    * @param b Right hand side vector
    * @param MaxIterations Maximum number of iterations
    * @return Returns an approximate solution to ||Ax - b||_2
    */
    Matrix<double, Dynamic, 1> solve(SparseMatrix<double, RowMajor> &A,
                                     const Matrix<double, Dynamic, 1> &b,
                                     long MaxIterations) const {
        SparseMatrix<double, ColMajor> AColMajor(A.rows(), A.cols());

        AColMajor.reserve(A.nonZeros());
        for (int k = 0; k < A.outerSize(); ++k)
            for (SparseMatrix<double, RowMajor>::InnerIterator it(A, k); it; ++it) {
                AColMajor.insert(it.row(), it.col()) = it.value();
            }

        return solve(A, AColMajor, b, MaxIterations);
    };

    /**
    * Returns the solution to Ax=b using the Randomized Extended Kaczmarz method
    *
    * @param A Input sparse matrix in column major format
    * @param b Right hand side vector
    * @param MaxIterations Maximum number of iterations
    * @return Returns an approximate solution to ||Ax - b||_2
    */
    Matrix<double, Dynamic, 1> solve(SparseMatrix<double, ColMajor> &AColMajor,
                                     const RowVector &b,
                                     long MaxIterations) const {
        SparseMatrix<double, RowMajor> A(AColMajor.rows(), AColMajor.cols());

        A.reserve(AColMajor.nonZeros());
        for (int k = 0; k< AColMajor.outerSize(); ++k)
            for (SparseMatrix<double, ColMajor>::InnerIterator it(AColMajor, k); it; ++it)
                A.insert(it.row(), it.col()) = it.value();

        return solve(A, AColMajor, b, MaxIterations);
    };
};