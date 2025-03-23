#include <iostream>

#include "solver.hpp"

using namespace rek;

int main() {
  const unsigned int m = 100, n = 10;
  const long ITERS = 100000;

  Matrix<double, Dynamic, Dynamic> A(m, n);
  RowVector xopt(n);
  xopt.setRandom();
  A.setRandom();

  RowVector b = A * xopt;

  auto solver = Solver();
  RowVector x = solver.solve(A, b, ITERS);

  std::cout << "(x , xopt)" << std::endl;
  for (unsigned int j = 0; j < A.cols(); j++) {
    std::cout << x(j) << " , " << xopt(j) << std::endl;
  }

  const RowVector residual = x - xopt;
  std::cout << "Least Squares error: " << residual.norm() << std::endl;

  return 0;
}
