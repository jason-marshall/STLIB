// -*- C++ -*-

#include "stlib/numerical/random/normal/NormalGeneratorZigguratVoss.h"

#include "stlib/ads/algorithm/statistics.h"

#include <iostream>
#include <vector>


using namespace stlib;

int
main()
{
  typedef numerical::NormalGeneratorZigguratVoss<> NormalGenerator;
  typedef NormalGenerator::DiscreteUniformGenerator UniformGenerator;

  // The uniform generator.
  UniformGenerator uniform;

  // Construct using the uniform generator.
  NormalGenerator f(&uniform);

  {
    // Copy constructor.
    NormalGenerator g(f);
  }
  {
    // Assignment operator.
    NormalGenerator g(&uniform);
    g = f;
  }
  {
    // Seed.
    NormalGenerator g(&uniform);
    g.seed(1);
  }

  // Check the mean and variance.
  // CONTINUE: Numerically check the values instead of printing them.
  const int Size = 1000000;
  std::vector<double> data(Size);
  for (int n = 0; n != Size; ++n) {
    data[n] = f();
  }
  double mean, variance;
  ads::computeMeanAndVariance(data.begin(), data.end(), &mean, &variance);
  std::cout << "Size = " << Size << ", mean = " << mean
            << ", variance = " << variance << "\n";

  return 0;
}
