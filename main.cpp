#include <iostream>

#include "samplers/AliasSampler.hpp"
#include "algorithms/REKSolver.hpp"

using namespace Eigen;

int main(void) {
	unsigned int m= 100, n = 10;

	MatrixXd A(m, n);
	RowVector xopt(n);
	xopt.setRandom();
	A.setRandom();
	RowVector b = A * xopt;

	REKSolver solver = REKSolver();

	long ITERS = 50000;

	RowVector x = solver.solve(A, b, ITERS);

	std::cout << "(x , xopt)" << std::endl;
	for (unsigned int j = 0 ; j < A.cols(); j++){
		std::cout << x(j) << " , " << xopt(j) << std::endl;
	}

	RowVector residual = x - xopt;
	std::cout << "Least Squares error: " << residual.norm() << std::endl;

    return 0;
}
