// -*- C++ -*-

#include "stlib/numerical/random/exponential/ExponentialGeneratorInversion.h"

#include "stlib/ads/algorithm/statistics.h"

#include <iostream>
#include <vector>


using namespace stlib;

int
main()
{
  typedef numerical::ExponentialGeneratorInversion<> ExponentialGenerator;
  typedef ExponentialGenerator::DiscreteUniformGenerator UniformGenerator;

  // The uniform generator.
  UniformGenerator uniform;

  // Construct using the uniform generator.
  ExponentialGenerator f(&uniform);

  {
    // Copy constructor.
    ExponentialGenerator g(f);
  }
  {
    // Assignment operator.
    ExponentialGenerator g(&uniform);
    g = f;
  }
  {
    // Seed.
    ExponentialGenerator g(&uniform);
    g.seed(1);
  }

  // Check the mean and variance.
  // CONTINUE: Numerically check the values instead of printing them.
  const int Size = 1000000;
  std::vector<double> data(Size);
  for (int n = 0; n != Size; ++n) {
    data[n] = f(2);
  }
  double mean, variance;
  ads::computeMeanAndVariance(data.begin(), data.end(), &mean, &variance);
  std::cout << "Size = " << Size << ", mean = " << mean
            << ", variance = " << variance << "\n";

  return 0;
}
