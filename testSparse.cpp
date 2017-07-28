#include <iostream>

#include "algorithms/REKSolver.hpp"

using namespace Eigen;

int main() {
	unsigned int m= 100, n = 10;

	MatrixXd A(m, n);
	RowVector xopt(n);
	xopt.setRandom();
	A.setRandom();
	RowVector b = A * xopt;

	REKSolver solver = REKSolver();

	long ITERS = 1000000;
	RowVector x = solver.solve(A, b, ITERS);

	// Error must be smaller than 0.5
	x -= xopt;
	std::cout<< "Error is " << x.norm() << std::endl;
	assert( x.norm() <= 0.5);
	std::cout<< "Success..." << std::endl;

	return 0;
}
