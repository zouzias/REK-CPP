#include <iostream>
#include <random>

#include "algorithms/RekSolver.hpp"

using namespace Eigen;

typedef Matrix<double, Dynamic, 1> RowVector;

SparseMatrix<double, RowMajor> randomSparseMatrix(int m, int n, double threshold){
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<Eigen::Triplet<double> > tripletList;
    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j) {
            auto v_ij = dist(gen);                         //generate random number
            if(v_ij < threshold)
                tripletList.emplace_back(Eigen::Triplet<double>(i, j, v_ij));
        }

    SparseMatrix<double, RowMajor> mat(m, n);
    mat.resizeNonZeros((int)tripletList.size());
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}

int main() {
    srand((unsigned int) time(0));
    unsigned int m = 100, n = 10;

    SparseMatrix<double, RowMajor> A = randomSparseMatrix(m, n, 0.5);
    std::cout << "A is: " << std::endl;
    std::cout << A << std::endl;
    RowVector xopt(n);
	xopt.setRandom();
	RowVector b = A * xopt;

	RekSolver solver = RekSolver();

	long ITERS = 10000;
    std::cout << "b : " << b << std::endl;

    RowVector x = solver.solve(A, b, ITERS);

	// Error must be smaller than 0.5
    RowVector residual = (x - xopt) / std::sqrt(n);
	std::cout<< "Least squares error is " << residual.norm() << std::endl;
	assert( residual.norm() <= 0.5);
	std::cout<< "Success..." << std::endl;

	return 0;
}