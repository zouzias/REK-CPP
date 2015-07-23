/*
 * REKSolver.hpp
 *
 *  Created on: May 4, 2015
 *      Author: zouzias
 */

#ifndef REKSOLVER_HPP_
#define REKSOLVER_HPP_

#include<iostream>
#include "../matrix/DoubleMatrix.hpp"

class REKSolver{

public:
	REKSolver(){}

	DoubleVector& solve(const DoubleMatrix& A, const DoubleVector& b, double MaxSeconds);
	DoubleVector& solve(const DoubleMatrix& A, const DoubleVector& b, long MaxIterations);
};

#endif /* REKSOLVER_HPP_ */
