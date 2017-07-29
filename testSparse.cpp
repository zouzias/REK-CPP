#include <iostream>
#include <random>

#include "algorithms/REKSolver.hpp"

using namespace Eigen;

SparseMatrix<double> randomSparseMatrix(long m, long n, double threshold){
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dist(0.0,1.0);


    std::vector<Eigen::Triplet<double> > tripletList;
    for(int i=0;i<m;++i)
        for(int j=0;j<n;++j) {
            auto v_ij = dist(gen);                         //generate random number
            if(v_ij < threshold)
                tripletList.emplace_back(Eigen::Triplet<double>(i, j, v_ij));
        }

    SparseMatrix<double> mat(m, n);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}

int main() {
    srand((unsigned int) time(0));
    unsigned int m = 100, n = 10;

	// SparseMatrix<double, RowMajor> A = randomSparseMatrix(m, n);
    MatrixXd A(m, n);
    RowVector xopt(n);
	xopt.setRandom();
    A.setRandom();
	RowVector b = A * xopt;

	REKSolver solver = REKSolver();

	long ITERS = 1000000;
	RowVector x = solver.solve(A, b, ITERS);

	// Error must be smaller than 0.5
    RowVector residual = (x - xopt) / std::sqrt(n);
	std::cout<< "Error is " << residual.norm() << std::endl;
	assert( residual.norm() <= 0.5);
	std::cout<< "Success..." << std::endl;

	return 0;
}
