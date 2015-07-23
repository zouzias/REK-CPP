#include "SparseVector.hpp"

void SparseVector::random(){
	/* initialize random seed */
	srand (time(NULL));

	for(unsigned int i = 0; i < _size; i++)
		_values[i] = (double)rand() / RAND_MAX;
};

unsigned int SparseVector::size() const{
	return _size;
};

double SparseVector::get(unsigned int index) const{
	if ( _values.find(index) != _values.end())
		return _values.at(index);
	else
		return 0.0;
};

void SparseVector::set(unsigned int index, double value){
	_values[index] = value;
};

void SparseVector::scale(double alpha){
	for(values_iter iter = _values.begin(); iter != _values.end(); iter++)
			iter->second *= alpha;
};

void SparseVector::minus(const DoubleVector& x){
	for(unsigned int i = 0; i < _size; i++)
		_values[i] = get(i) - x.get(i);
};
