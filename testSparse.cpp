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
#include "matrix/impl/SparseMatrix.hpp"
#include "vector/impl/DenseVector.hpp"

using namespace std;

int main(void) {
	unsigned int m= 100, n = 10;

	DoubleMatrix& A = *new SparseMatrix(m,n);
	DoubleVector& xopt = *new DenseVector(n);
	xopt.random();
	A.random();
	DoubleVector& b = A.times(xopt);

	REKSolver solver = REKSolver();

	long ITERS = 1000000;
	DoubleVector& x = solver.solve(A, b, ITERS);

	// Error must be smaller than 0.5
	x.minus(xopt);
	cout<< "Error is " << x.DNRM2() << endl;
	assert( x.DNRM2() <= 0.5);
	cout<< "Success..." << endl;

	return 0;
}
