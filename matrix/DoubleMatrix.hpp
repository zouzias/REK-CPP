#ifndef DOUBLEMATRIX_HPP_
#define DOUBLEMATRIX_HPP_

#include "../vector/DoubleVector.hpp"
class DoubleVector;

class DoubleMatrix{

protected:
  unsigned int m, n;
  const static int DEFAULT_SIZE = 10;

public:

    DoubleMatrix(){
      this->m = DEFAULT_SIZE;
      this->n = DEFAULT_SIZE;
    };

    DoubleMatrix(unsigned int m, unsigned int n){
      this->m = m;
      this->n = n;
    };

    virtual ~DoubleMatrix(){};

	   /**
     * Fill in the entries with random values
     *
     */
    virtual void random() = 0;

    /**
     * Get the number of rows
     *
     * @return
     */
    virtual int numRows()  const = 0;

    /**
     * Get the number of columns
     *
     * @return
     */
    virtual int numCols()  const = 0;

    /**
     * Get the i-th row
     *
     * @param i
     * @return
     */
    virtual DoubleVector& getRow(unsigned int i)  const = 0;

    /**
     * Get the j-th column
     *
     * @param j
     * @return
     */
    virtual DoubleVector& getColumn(unsigned int j) const = 0;

    /**
     * Compute a distribution over the rows. Namely, the i-th row A_i has probability mass
     *  ||A_i||_2^2 / ||A||_F^2 where ||A||_F^2 is the Frobenius norm of A
     *
     * @return A vector of size equal to the number of rows containing the above probability distribution.
     */
    virtual DoubleVector& rowNorms()  const = 0;

    /**
     * Compute a distribution over the columns. Namely, the j-th column A(j) has probability mass
     *  ||A(j)||_2^2 / ||A||_F^2 where ||A||_F^2 is the Frobenius norm of A
     *
     * @return A vector of size equal to the number of column containing the above probability distribution.
     */
    virtual DoubleVector& columnNorms()  const = 0;

    /**
     * Computes y = A * x
     *
     * @param x
     * @return y
     */
    virtual DoubleVector& times(const DoubleVector& x) const = 0; // y = A * x

    /**
     * Frobenius norm of a matrix A, i.e., normF(A)= sqrt(sum_{i,j} A(i,j)^2)
     *
     * @return
     */
    virtual double normF() const = 0; // Frobenius norm of matrix

    /**
     * Get (i,j) entry of matrix
     *
     * @param i
     * @param j
     * @return
     */
    virtual double get(unsigned int i, unsigned int j) const = 0;

    /**
     * Set (i,j)-th entry of matrix to s
     *
     * @param i
     * @param j
     * @param s
     */
     virtual void set(unsigned int i, unsigned int j, double s) = 0;

};

#endif /* DOUBLEMATRIX_HPP_ */
