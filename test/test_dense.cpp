#include <iostream>

#include "solver.hpp"

using namespace Eigen;
using namespace rek;

int main() {
  srand((unsigned int)time(nullptr));
  unsigned int m = 100, n = 10;
  long ITERS = 10000;

  Matrix<double, Dynamic, Dynamic> A(m, n);
  rek::RowVector xopt(n);
  xopt.setRandom();
  A.setRandom();
  rek::RowVector b = A * xopt;

  auto solver = Solver();
  rek::RowVector x = solver.solve(A, b, ITERS);

  // Error should be small
  rek::RowVector residual = (x - xopt) / std::sqrt(n);
  std::cout << "Error is " << residual.norm() << std::endl;
  assert(residual.norm() <= 0.01);
  std::cout << "Success..." << std::endl;

  return 0;
}
