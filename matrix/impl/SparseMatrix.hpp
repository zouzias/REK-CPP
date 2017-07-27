#ifndef SPARSEMATRIX_HPP_
#define SPARSEMATRIX_HPP_

#include<vector>
#include "../DoubleMatrix.hpp"
#include "../../vector/impl/SparseVector.hpp"
#include "../../vector/impl/DenseVector.hpp"

class SparseMatrix : public DoubleMatrix{

protected:
	std::vector<SparseVector*> _rows;   // Stored By rows
	std::vector<SparseVector*> _columns;  // Stored by columns

public:

	SparseMatrix(int m, int n): DoubleMatrix(m, n){
		// Init rows
		for(int i = 0; i < m; i++)
			_rows.push_back(new SparseVector(n));

		// Init columns
		for(int j = 0; j < n; j++)
			_columns.push_back(new SparseVector(m));
	};

	~SparseMatrix(){
			for(unsigned int i = 0; i < _rows.size(); i++)
				delete _rows[i];

			for(unsigned int j = 0; j < _columns.size(); j++)
				delete _columns[j];
	};

	virtual void random(){
		/* initialize random seed: */
		srand (time(NULL));

		for (unsigned int i = 0, m = _rows.size(); i < m; i++){
			for (unsigned int j = 0, n = _columns.size(); j < n; j++) {
				double val = (double)rand() / RAND_MAX;
				_rows[i]->set(j, val);
				_columns[j]->set(i, val);
			}
		}
	}

	virtual int numRows() const{
		return _rows.size();
	}

	virtual int numCols() const{
		return _columns.size();
	}

	virtual DoubleVector& getRow(unsigned int i) const{
		return *_rows[i];
	}

	virtual DoubleVector& getColumn(unsigned int j) const{
		return *_columns[j];
	}

	virtual DoubleVector& rowNorms() const{
		DoubleVector& rowNRMs = *new DenseVector(_rows.size());
		for (unsigned int i = 0, m = _rows.size(); i < m; i++)
			rowNRMs.set(i, getRow(i).DNRM2());

		return rowNRMs;
	}

	virtual DoubleVector& columnNorms() const{
		DoubleVector& colNRMs = *new DenseVector(_columns.size());
		for (unsigned int j = 0, n = _columns.size(); j < n; j++)
			colNRMs.set(j, getColumn(j).DNRM2());

		return colNRMs;
	}

	virtual DoubleVector& times(const DoubleVector& x) const{
		DoubleVector& Ax = *new DenseVector(numRows());
		for(unsigned int j = 0, n = numCols(); j < n; j++)
			Ax.DAXPY(x.get(j), this->getColumn(j));

		return Ax;
	}

	virtual double normF() const{
		DoubleVector& rowNorms = this->rowNorms();
		double frobenius = rowNorms.DNRM2();
		delete &rowNorms;

		return frobenius;
	}

	virtual double get(unsigned int i,unsigned int j) const{
		return _rows[i]->get(j);
	}

	virtual void set(unsigned int i, unsigned int j, double s){
		_rows[i]->set(j, s);
		_columns[j]->set(i, s);
	}

};

#endif /* SPARSEMATRIX_HPP_ */
