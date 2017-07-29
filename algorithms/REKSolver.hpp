#pragma once

#include<iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "../samplers/AliasSampler.hpp"

using namespace Eigen;


typedef Matrix<double, Dynamic, 1> RowVector;

class REKSolver{

public:
	REKSolver() = default;

    RowVector solve(MatrixXd& A, const RowVector& b, double MaxSeconds) const {
        RowVector vector(A.cols());
        return vector;
    }

    RowVector solve(MatrixXd& A, const RowVector& b, long MaxIterations) const {
        double val;
        long i_k, j_k;
        RowVector x(A.cols());
        RowVector z(b);
        RowVector rowNorms(A.rows());
        RowVector columnNorms(A.cols());

        x.setZero();
        for (int i = 0 ; i < A.rows(); i++)
            rowNorms(i) = A.row(i).squaredNorm();

        for (int j = 0 ; j < A.cols(); j++)
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

            val = - z.dot(A.col(j_k)) / columnNorms(j_k);     // val = - dot(z, A(:, j_k)) / colProbs(j_k)

            z += val * A.col(j_k);                             // z = z + val * A(:, j_k);

            val = x.dot(A.row(i_k));                                // val = dot(x, A(i_k, :))
            val = (b(i_k) - z(i_k) - val) / rowNorms(i_k);  // val = (b(i_k) - z(i_k) - val) / roProbs(i_k)

            x += val * A.row(i_k);                                // x = x + val * A(i_k, :);
        }
        return x;
    }
};