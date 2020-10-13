#include <iostream>
#include <vector>

#include "samplers/AliasSampler.hpp"

using namespace std;

int main() {

  unsigned int n = 100, numSamples = 500;
  double std = 10.0;
  vector<double> prob(n);

  for (unsigned int i = 0; i < n; i++) {
    prob[i] = std * drand48();
  }

  rek::sample::AliasSampler alias(prob);
  alias.initSampler();
  vector<uint> samples = alias.sample(numSamples);

  cout << "Checking if random samples are in range [0, " << n - 1 << "]"
       << endl;
  for (unsigned int sample : samples) {
    assert(sample >= 0);
    assert(sample < n);
  }
  cout << "Success..." << endl;

  return 0;
}
