## Randomized Extended Kaczmarz (C++) ![build](https://github.com/zouzias/REK-CPP/actions/workflows/cmake.yml/badge.svg)

### Overview 
The Randomized Extended Kaczmarz algorithm is a randomized algorithm for solving least-squares/linear regression problems.

- [<b>Randomized Extended Kaczmarz for Solving Least-Squares.</b>](http://dx.doi.org/10.1137/120889897)
SIAM. J. Matrix Anal. & Appl., 34(2), 773–793. (21 pages) 
Authors: [Anastasios Zouzias](https://github.com/zouzias) and Nikolaos Freris 

### Installation

Clone the project. Type

```bash
./build.sh
./build/bin/test_{dense|sparse|sparse_colmajor}
```

The above code runs a simple instance of least-squares for a gaussian random matrix A and gaussian vector b.

## Usage

```c++
using namespace Eigen;

unsigned int m= 100, n = 10;
Matrix<double, Dynamic, Dynamic> A(m, n);
RowVector xopt(n);

xopt.setRandom();
A.setRandom();

RowVector b = A * xopt;

auto solver = rek::Solver();

long ITERS = 50000;

RowVector x = solver.solve(A, b, ITERS);

std::cout << "(x , xopt)" << std::endl;
for (unsigned int j = 0 ; j < A.cols(); j++){
    std::cout << x(j) << " , " << xopt(j) << std::endl;
}

RowVector residual = x - xopt;
std::cout << "Least Squares error: " << residual.norm() << std::endl;
```

### Implementation Details
REK-CPP is an implementation of REK with two additional technical features. First, REK-CPP utilizes level-1 BLAS routines for 
all operations of REK and second REK-CPP additionally stores explicitly the transpose of A for more efficient
memory access of both the rows and columns of A. 

The sampling operations of REK are implemented using the so-called ``alias method'' for generating samples 
from any given discrete distribution [Vos91]. In particular, the alias method, assuming access 
to a uniform random variable on [0,1] in constant time and linear time preprocessing, generates one sample
of a given distribution in constant time. We use an implementation of W. D. Smith.

### Credits and acknowledgments

Credits go to Warren D. Smith for implementing the aliasing method [Vos91] in C.
<br><br>
[Vos91] M. D. Vose. A Linear Algorithm for Generating Random Numbers with a given Distribution. 
<br>
IEEE Trans. Softw. Eng., 17(9):972–975, September 1991.
