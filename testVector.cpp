#include <iostream>

#include "vector/impl/DenseVector.hpp"
#include "vector/impl/SparseVector.hpp"
#include "vector/DoubleVector.hpp"

using namespace std;

int main(void) {

	unsigned int m= 1000, n = 500;

	DoubleVector* x = new SparseVector(m);
  DoubleVector* y = new DenseVector(n);

  for(unsigned int i = 0 ; i < m; i++)
    x->set(i, 1.0);

  for(unsigned int j = 0 ; j < n; j++)
    y->set(j, 1.0);


  // Test size
  cout<< "Testing size" << endl;
	assert( x->size() == m);
  assert( y->size() == n);
  cout<< "Success..." << endl;

  cout<< "Testing norm" << endl;
	assert( x->DNRM2() == m);
  assert( y->DNRM2() == n);
  cout<< "Success..." << endl;

  cout<< "Testing norm/daxpy/minus" << endl;
  assert( x->DNRM2() == m);
  x->DAXPY(1.0, *x);
	assert( x->DNRM2() == 4 * m);
  x->scale(0.5);
  assert( x->DNRM2() == m);
  x->minus(*x);
  assert( x->DNRM2() == 0.0);
  cout<< "Success..." << endl;

	return 0;
}
