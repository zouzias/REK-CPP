//============================================================================
// Name        : REK-CPP.cpp
// Author      : Anastasios Zouzias
// Version     :
// Copyright   : Apache 2.0 License
// Description : Hello World in C, Ansi-style
//============================================================================

#include <iostream>

#include "samplers/AliasSampler.hpp"
#include "algorithms/REKSolver.hpp"
#include "matrix/impl/DenseMatrix.hpp"
#include "vector/impl/DenseVector.hpp"

using namespace std;

int main(void) {
	// A is an (m, n ) random matrix
	unsigned int m= 100, n = 10;
	DoubleMatrix& A = *new DenseMatrix(m,n);
	A.random();

	// xopt is a random n-vector
	DoubleVector& xopt = *new DenseVector(n);
	xopt.random();

	// b = A * x
	DoubleVector& b = A.times(xopt);

	REKSolver solver = REKSolver();

	long ITERS = 1000000;
	DoubleVector& x = solver.solve(A, b, ITERS);

	// Error must be smaller than 0.5
	x.minus(xopt);
	cout<< "Error is " << x.DNRM2() << endl;
	assert( x.DNRM2() <= 0.5);
	cout<< "Success..." << endl;

	delete &A;
	delete &xopt;

	return 0;
}
