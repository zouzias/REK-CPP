
#ifndef DENSEVECTOR_HPP_
#define DENSEVECTOR_HPP_

#include "../DoubleVector.hpp"
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

class DenseVector : public DoubleVector {
private:

std::vector<double> _vector;

public:

	DenseVector(): DoubleVector(DEFAULT_SIZE), _vector(DEFAULT_SIZE){};

	DenseVector(unsigned int size): DoubleVector(size), _vector(size){
		_size = size;
	};

	DenseVector(const DoubleVector& vec): _vector(vec.size()){
		_size = vec.size();
		for (unsigned int i = 0; i < _vector.size(); i++){
			_vector[i] = vec.get(i);
		}
	};

	~DenseVector(){};

	void random();

	unsigned int size() const;

	double get(unsigned int index) const;

	void set(unsigned int index, double value);

	void scale(double alpha);

	void minus(const DoubleVector& x);
};

#endif
