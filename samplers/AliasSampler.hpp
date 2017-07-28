#ifndef ALIASSAMPLER_HPP_
#define ALIASSAMPLER_HPP_

#include<vector>
#include <cassert>
#include <cstdlib>
#include <eigen3/Eigen/Dense>

// Needed for Walker sampling
#define real double
#ifndef uint
#define uint unsigned int
#endif

using namespace Eigen;

typedef Matrix<double, Dynamic, 1> RowVector;


class AliasSampler{

private:

  unsigned int N;
  std::vector<uint> A;
  std::vector<uint> B;
  std::vector<double> Y;

public:

  ~AliasSampler(){};

  AliasSampler(const std::vector<double> probs): A(probs.size() + 2), B(probs.size() + 2), Y(probs.size() + 2){
    uint j;
    double sum = 0;

    this->N = (unsigned int)probs.size();
    for (j = 0; j < N; j++)
        sum += probs[j];

    sum = 1 / sum;

    // Normalize it now
    for (j = 0; j < N; j++)
        Y[j + 1] = probs[j] * sum;

  };

  AliasSampler(const RowVector& probs): A(probs.size() + 2), B(probs.size() + 2), Y(probs.size() + 2){

    uint j;
    double sum = 0;

    this->N = (unsigned int)probs.size();

    for (j = 0; j < N; j++)
        sum += probs(j, 1);

    sum = 1 / sum;

    // Normalize it now
    for (j = 0; j < N; j++)
        Y[j + 1] = probs(j, 1) * sum;

  };

  std::vector<uint>* sample(unsigned int numSamples){
    std::vector<uint>* samples = new std::vector<uint>(numSamples);
    // Sample from Alias Sampler
    for (unsigned int k = 0; k < numSamples; k++)
        (*samples)[k] = walkerSample();

    return samples;
  };

  void initSampler(){
    uint i, j, k;
    assert(1 <= N);
    for (i = 1; i <= N; i++) {
        A[i] = i;
        B[i] = i; /* initial destins=stay there */
        assert(Y[i] >= 0.0);
        Y[i] = Y[i] * N; /* scale probvec */
    }
    B[0] = 0;
    Y[0] = 0.0;
    B[N + 1] = N + 1;
    Y[N + 1] = 2.0; /* sentinels */
    i = 0;
    j = N + 1;
    for (;;) {
        do {
            i++;
        } while (Y[B[i]] < 1.0); /* find i so X[B[i]] needs more */
        do {
            j--;
        } while (Y[B[j]] >= 1.0); /* find j so X[B[j]] wants less */
        if (i >= j)
            break;
        k = B[i];
        B[i] = B[j];
        B[j] = k; /* swap B[i], B[j] */
    }

    i = j;
    j++;
    while (i > 0) {
        while (Y[B[j]] <= 1.0)
            j++;
        /* find j so X[B[j]] needs more */
        assert(Y[B[i]] < 1.0); /* meanwhile X[B[i]] wants less */
        if (j > N)
            break;
        assert(j <= N);
        assert(Y[B[j]] > 1.0);
        Y[B[j]] -= 1.0 - Y[B[i]]; /* B[i] will donate to B[j] to fix up */
        A[B[i]] = B[j];
        if (Y[B[j]] < 1.0) { /* X[B[j]] now wants less so readjust ordering */
            assert(i < j);
            k = B[i];
            B[i] = B[j];
            B[j] = k; /* swap B[j], B[i] */
            j++;
        } else
            i--;
    }
  };

  uint walkerSample (){
    uint i;
    real r;
    /* Let i = random uniform integer from {1,2,...N};  */
    i = 1 + (uint) ((N - 1) * drand48());
    r = drand48();
    if (r > Y[i])
        i = A[i];

    return i - 1;
  }
};

#endif /* ALIASMETHOD_HPP_ */
