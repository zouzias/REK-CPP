## Randomized Extended Kaczmarz (C++)

![build](https://github.com/zouzias/REK-CPP/actions/workflows/cmake.yml/badge.svg)
![valgrind](https://github.com/zouzias/REK-CPP/actions/workflows/valgrind.yml/badge.svg)
![clang-format](https://github.com/zouzias/REK-CPP/actions/workflows/clang-format-check.yml/badge.svg)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

### Overview

The Randomized Extended Kaczmarz (REK) algorithm is a randomized iterative method for solving
least-squares/linear regression problems, including inconsistent and rank-deficient systems.
This repository provides a header-only C++ implementation built on top of [Eigen](https://eigen.tuxfamily.org/),
supporting both dense and sparse matrices.

- [<b>Randomized Extended Kaczmarz for Solving Least-Squares.</b>](http://dx.doi.org/10.1137/120889897)
SIAM. J. Matrix Anal. & Appl., 34(2), 773–793. (21 pages)
Authors: [Anastasios Zouzias](https://github.com/zouzias) and Nikolaos Freris

### Requirements

- A C++14-compatible compiler (e.g., GCC or Clang)
- [CMake](https://cmake.org/) >= 3.19
- [Eigen](https://eigen.tuxfamily.org/) >= 3.3 (`libeigen3-dev` on Ubuntu/Debian)

### Build

Clone the repository, then build with CMake:

```bash
make build
```

This configures and builds the project into `build/`, producing the following binaries under `build/bin`:

- `rek_cpp` — a simple demo that runs REK on a dense random least-squares instance
- `test_dense`, `test_sparse`, `test_sparse_colmajor`, `test_sampler` — unit tests

Run the demo directly:

```bash
./build/bin/rek_cpp
```

Run the unit test suite:

```bash
make test
```

### Usage

```c++
#include "solver.hpp"

int main() {
  const unsigned int m = 100, n = 10;
  const long ITERS = 100000;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(m, n);
  rek::RowVector xopt(n);
  xopt.setRandom();
  A.setRandom();

  rek::RowVector b = A * xopt;

  auto solver = rek::Solver();
  rek::RowVector x = solver.solve(A, b, ITERS);

  std::cout << "(x , xopt)" << std::endl;
  for (unsigned int j = 0; j < A.cols(); j++) {
    std::cout << x(j) << " , " << xopt(j) << std::endl;
  }

  const rek::RowVector residual = x - xopt;
  std::cout << "Least Squares error: " << residual.norm() << std::endl;

  return 0;
}
```

`rek::Solver::solve` is overloaded to accept dense (`Eigen::Matrix`) or sparse
(`Eigen::SparseMatrix`, row- or column-major) inputs for `A`, alongside the right-hand
side vector `b` and a maximum number of iterations.

### Implementation Details

REK-CPP implements REK with two additional technical features. First, it utilizes level-1 BLAS
routines for all operations of REK, and second, it stores the transpose of `A` explicitly for more
efficient memory access of both the rows and columns of `A`.

The sampling operations of REK are implemented using the so-called "alias method" for generating
samples from any given discrete distribution [Vos91]. In particular, the alias method, assuming access
to a uniform random variable on [0,1] in constant time and linear-time preprocessing, generates one sample
of a given distribution in constant time. We use an implementation of W. D. Smith.

### Project layout

```
src/          Header-only library (solver.hpp, sampler.hpp) and demo (main.cpp)
test/         Unit tests (dense, sparse, sparse column-major, sampler)
CMake/        CMake find-modules
dev-scripts/  Helper scripts (e.g., valgrind runs)
```

### Credits and acknowledgments

Credits go to Warren D. Smith for implementing the aliasing method [Vos91] in C.

[Vos91] M. D. Vose. A Linear Algorithm for Generating Random Numbers with a given Distribution.
IEEE Trans. Softw. Eng., 17(9):972–975, September 1991.

### License

This project is licensed under the [Apache License 2.0](LICENSE).
