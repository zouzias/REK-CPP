#ifndef REKSOLVER_HPP_
#define REKSOLVER_HPP_


#include<iostream>
#include <eigen3/Eigen/Dense>


using namespace Eigen;


typedef Matrix<double, Dynamic, Dynamic> MatrixXd;

typedef Matrix<double, Dynamic, 1> RowVector;


class REKSolver{

public:
	REKSolver(){}

    RowVector solve(MatrixXd& A, const RowVector& b, double MaxSeconds) const;
    RowVector solve(MatrixXd& A, const RowVector& b, long MaxIterations) const;
};

#endif /* REKSOLVER_HPP_ */
