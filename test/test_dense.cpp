#include <iostream>
#include "algorithms/Solver.hpp"

using namespace rek;

int main() {

    srand((unsigned int) time(0));
    unsigned int m = 100, n = 10;
    long ITERS = 10000;

    Matrix<double, Dynamic, Dynamic> A(m, n);
    RowVector xopt(n);
    xopt.setRandom();
    A.setRandom();
    RowVector b = A * xopt;

    auto solver = Solver();
    RowVector x = solver.solve(A, b, ITERS);

    // Error must be smaller than 0.5
    RowVector residual = (x - xopt) / std::sqrt(n);
    std::cout<< "Error is " << residual.norm() << std::endl;
    assert( residual.norm() <= 0.01);
    std::cout<< "Success..." << std::endl;

    return 0;
}
