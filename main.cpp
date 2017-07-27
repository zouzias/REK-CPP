#include <iostream>

#include "samplers/AliasSampler.hpp"
#include "algorithms/REKSolver.hpp"


#include "matrix/impl/DenseMatrix.hpp"
#include "vector/impl/DenseVector.hpp"

using namespace std;

int main(void) {
	unsigned int m= 100, n = 10;

	DoubleMatrix& A = *new DenseMatrix(m,n);
	DoubleVector& xopt = *new DenseVector(n);
	xopt.random();
	A.random();
	DoubleVector& b = A.times(xopt);

	REKSolver solver = REKSolver();

	long ITERS = 50000;
	DoubleVector& x = solver.solve(A, b, ITERS);

	cout<<"(x , xopt)"<<endl;
	for (unsigned int j = 0 ; j < A.numCols(); j++){
		cout<< x.get(j) << " , " << xopt.get(j) <<endl;
	}

	x.minus(xopt);
	cout<< "LS error: " << x.DNRM2() << endl;

  delete &A;
  delete &xopt;
  delete &b;
  delete &x;

	return 0;
}
