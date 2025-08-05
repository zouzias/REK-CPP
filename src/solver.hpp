#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "sampler.hpp"

namespace rek {

using RowVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

class Solver {
  /** Periodically check for convergence */
  constexpr static int blockSize_ = 1000;

  constexpr static double tolerance_ = 10e-5;

  bool hasConverged(Eigen::SparseMatrix<double, Eigen::RowMajor> &A,
                    Eigen::SparseMatrix<double, Eigen::ColMajor> &AColMajor,
                    const RowVector &x, const RowVector &b, const RowVector &z,
                    double tolerance) const {
    const RowVector residual = A * x - b + z;
    const bool condOne = residual.norm() < tolerance;

    // Early return
    if (!condOne) return false;

    const RowVector zA = z.transpose() * AColMajor;
    return zA.norm() < tolerance;
  }

  // FIXME: Is this necessary? Duplicate with hasConverged()
  bool hasConvergedDense(
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &A,
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
          &AColMajor,
      const RowVector &x, const RowVector &b, const RowVector &z,
      double tolerance) const {
    const RowVector residual = A * x - b + z;
    const bool condOne = residual.norm() < tolerance;

    // Early return
    if (!condOne) return false;

    const RowVector zA = z.transpose() * AColMajor;
    return zA.norm() < tolerance;
  }

  RowVector solve(Eigen::SparseMatrix<double, Eigen::RowMajor> &A,
                  Eigen::SparseMatrix<double, Eigen::ColMajor> &AColMajor,
                  const RowVector &b, long MaxIterations,
                  double tolerance = tolerance_) const {
    const long m = A.rows(), n = AColMajor.cols();
    RowVector x(A.cols());
    RowVector z(b);
    RowVector rowNorms(A.rows());
    RowVector columnNorms(A.cols());

    x.setZero();

    for (long i = 0; i < m; i++) {
      rowNorms(i) = A.row(i).squaredNorm();
    }

    for (long j = 0; j < n; j++) {
      columnNorms(j) = AColMajor.col(j).squaredNorm();
    }

    sample::AliasSampler rowSampler(rowNorms);
    sample::AliasSampler colSampler(columnNorms);

    // Initialize Alias samplers, O(n)
    rowSampler.initSampler();
    colSampler.initSampler();

    for (long k = 0; k < MaxIterations; k++) {
      // Check for convergence every blockSize_ iterations
      if ((k + 1) % blockSize_ == 0 &&
          hasConverged(A, AColMajor, x, b, z, tolerance))
        break;

      const long i_k = rowSampler.walkerSample();
      const long j_k = colSampler.walkerSample();

      double val = -AColMajor.col(j_k).dot(z) / columnNorms(j_k);
      z += val * AColMajor.col(j_k);
      val = A.row(i_k).dot(x);
      val = (b(i_k) - z(i_k) - val) / rowNorms(i_k);

      // TODO: .toDense() seems to be required here!
      // FIXME: replace .toDense() with iteration over sparse i_k row
      x += val * A.row(i_k).toDense();
    }

    return x;
  }

  /**
   * Returns the solution to Ax=b using the Randomized Extended Kaczmarz method
   *
   * @param AColMajor Input dense matrix
   * @param b Right hand side vector
   * @param maxIterations Maximum number of iterations
   * @param tolerance Accuracy tolerance, default tolerance_
   * @return Returns an approximate solution to ||Ax - b||_2
   */
  RowVector solve(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::ColMajor> &AColMajor,
                  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor> &ARowMajor,
                  const RowVector &b, long maxIterations,
                  double tolerance = tolerance_) const {
    const long m = ARowMajor.rows(), n = AColMajor.cols();
    RowVector x(AColMajor.cols());
    RowVector z(b);
    RowVector rowNorms(AColMajor.rows());
    RowVector columnNorms(AColMajor.cols());

    x.setZero();

    for (long i = 0; i < m; i++) {
      rowNorms(i) = ARowMajor.row(i).squaredNorm();
    }

    for (long j = 0; j < n; j++) {
      columnNorms(j) = AColMajor.col(j).squaredNorm();
    }

    for (long k = 0; k < maxIterations; k++) {
      // Check for convergence every blockSize_ iterations
      if ((k + 1) % blockSize_ == 0 &&
          hasConvergedDense(ARowMajor, AColMajor, x, b, z, tolerance)) {
        break;
      }

      // Extended Kaczmarz
      const long i_k = k % m;
      const long j_k = k % n;

      double val = -z.dot(AColMajor.col(j_k)) / columnNorms(j_k);
      z += val * AColMajor.col(j_k);
      val = x.dot(ARowMajor.row(i_k));
      val = (b(i_k) - z(i_k) - val) / rowNorms(i_k);

      x += val * ARowMajor.row(i_k);
    }
    return x;
  }

 public:
  Solver() = default;

  RowVector solve(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::ColMajor> &AColMajor,
                  const RowVector &b, long MaxIterations) const {
    // Duplicate input matrix to row major storage
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        ARowMajor(AColMajor);
    return solve(AColMajor, ARowMajor, b, MaxIterations);
  }

  /**
   * Returns the solution to Ax=b using the Randomized Extended Kaczmarz method
   *
   * @param A Input sparse matrix in row major format
   * @param b Right hand side vector
   * @param maxIterations Maximum number of iterations
   * @param tolerance Accuracy tolerance, default tolerance_
   * @return Returns an approximate solution to ||Ax - b||_2
   */
  RowVector solve(Eigen::SparseMatrix<double, Eigen::RowMajor> &A,
                  const Eigen::Matrix<double, Eigen::Dynamic, 1> &b,
                  long maxIterations, double tolerance = tolerance_) const {
    Eigen::SparseMatrix<double, Eigen::ColMajor> AColMajor(A.rows(), A.cols());

    // Copy row major to column major sparse matrix
    AColMajor.reserve(A.nonZeros());
    for (long k = 0; k < A.outerSize(); ++k)
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, k);
           it; ++it) {
        AColMajor.insert(it.row(), it.col()) = it.value();
      }

    return solve(A, AColMajor, b, maxIterations, tolerance);
  };

  /**
   * Returns the solution to Ax=b using the Randomized Extended Kaczmarz method
   *
   * @param A Input sparse matrix in column major format
   * @param b Right hand side vector
   * @param maxIterations Maximum number of iterations
   * @param tolerance Accuracy tolerance, default tolerance_
   * @return Returns an approximate solution to ||Ax - b||_2
   */
  RowVector solve(Eigen::SparseMatrix<double, Eigen::ColMajor> &AColMajor,
                  const RowVector &b, long maxIterations,
                  double tolerance = tolerance_) const {
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(AColMajor.rows(),
                                                   AColMajor.cols());

    // Copy column major to row major sparse matrix
    A.reserve(AColMajor.nonZeros());
    for (long k = 0; k < AColMajor.outerSize(); ++k)
      for (Eigen::SparseMatrix<double, Eigen::ColMajor>::InnerIterator it(
               AColMajor, k);
           it; ++it)
        A.insert(it.row(), it.col()) = it.value();

    return solve(A, AColMajor, b, maxIterations, tolerance);
  };
};

}  // namespace rek