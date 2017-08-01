#pragma once

#include<iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "../samplers/AliasSampler.hpp"

using namespace Eigen;
typedef Matrix<double, Dynamic, 1> RowVector;

class REKSolver {
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

            // Extended Kaczmarz
            // i_k = k % A.rows();
            // j_k = k % A.cols();

            val = -AColMajor.col(j_k).dot(z) / columnNorms(j_k);     // val = - dot(z, A(:, j_k)) / colProbs(j_k)
            z += val * AColMajor.col(j_k);                           // z = z + val * A(:, j_k);
            val = A.row(i_k).dot(x);                                 // val = dot(x, A(i_k, :))
            val = (b(i_k) - z(i_k) - val) / rowNorms(i_k);           // val = (b(i_k) - z(i_k) - val) / rowProbs(i_k)

            // TODO: .toDense() seems to be required here!
            x += val * A.row(i_k).toDense();                                // x = x + val * A(i_k, :);
        }

        return x;
    }

    public:
        REKSolver() = default;


        RowVector solve(Matrix<double, Dynamic, Dynamic> &A,
                                         const RowVector &b,
                                         double MaxSeconds) const {
            Matrix<double, Dynamic, 1> vector(A.cols());
            return vector;
        }

        /**
         *
         * @param A
         * @param b
         * @param MaxIterations
         * @return
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

                val = -z.dot(A.col(j_k)) / columnNorms(j_k);     // val = - dot(z, A(:, j_k)) / colProbs(j_k)

                z += val * A.col(j_k);                             // z = z + val * A(:, j_k);

                val = x.dot(A.row(i_k));                                // val = dot(x, A(i_k, :))
                val = (b(i_k) - z(i_k) - val) / rowNorms(i_k);  // val = (b(i_k) - z(i_k) - val) / roProbs(i_k)

                x += val * A.row(i_k);                                // x = x + val * A(i_k, :);
            }
            return x;
        }

    /**
     *
     * @param A
     * @param b
     * @param MaxIterations
     * @return
     */
    Matrix<double, Dynamic, 1> solve(SparseMatrix<double, RowMajor> &A,
                                     const Matrix<double, Dynamic, 1> &b,
                                     long MaxIterations) const {
        SparseMatrix<double, ColMajor> AColMajor(A.rows(), A.cols());
        AColMajor.reserve(A.nonZeros());

        // TODO: Fix this with sparse iter
        for (int i = 0 ; i < A.rows(); i++)
            for (int j = 0; j < A.cols(); j++)
                if (A.coeff(i,j))
                    AColMajor.insert(i,j) = A.coeff(i, j);

        return solve(A, AColMajor, b, MaxIterations);
    };

    Matrix<double, Dynamic, 1> solve(SparseMatrix<double, ColMajor> &AColMajor,
                                     const RowVector &b,
                                     long MaxIterations) const {
        SparseMatrix<double, RowMajor> A(AColMajor.rows(), AColMajor.cols());

        // TODO: Fix this with sparse iter
        A.reserve(AColMajor.nonZeros());
        for (int i = 0 ; i < A.rows(); i++)
            for (int j = 0; j < A.cols(); j++)
                if (AColMajor.coeff(i,j))
                    A.insert(i,j) = AColMajor.coeff(i, j);

        return solve(A, AColMajor, b, MaxIterations);
    };
};