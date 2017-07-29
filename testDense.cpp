#include <iostream>
#include "algorithms/REKSolver.hpp"

using namespace Eigen;

int main() {

    srand((unsigned int) time(0));
    unsigned int m = 100, n = 10;

    MatrixXd A(m, n);
    RowVector xopt(n);
    xopt.setRandom();
    A.setRandom();
    RowVector b = A * xopt;

    REKSolver solver = REKSolver();

    long ITERS = 1000000;
    RowVector x = solver.solve(A, b, ITERS);

    // Error must be smaller than 0.5
    RowVector residual = (x - xopt) / std::sqrt(n);
    std::cout<< "Error is " << residual.norm() << std::endl;
    assert( residual.norm() <= 0.01);
    std::cout<< "Success..." << std::endl;

    return 0;
}
