// -*- C++ -*-

#include "stlib/numerical/random/exponential/ExponentialGeneratorZiggurat.h"

#include "stlib/ads/algorithm/statistics.h"

#include <iostream>
#include <vector>


using namespace stlib;

int
main()
{
  typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;
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

  double t = 0;
  std::size_t count = 0;
  while (t < 1000) {
    t += f(1);
    if (t < 1000) {
      ++count;
    }
  }
  std::cout << "count = " << count << '\n';

  t = 0;
  count = 0;
  while (t < 1000) {
    double e = f(1);
    if (2 < e) {
      t += 2;
    }
    else {
      t += e;
      if (t < 1000) {
        ++count;
      }
    }
  }
  std::cout << "count = " << count << '\n';

  t = 0;
  count = 0;
  while (t < 1000) {
    double e = f(1);
    if (1 < e) {
      t += 1;
    }
    else {
      t += e;
      if (t < 1000) {
        ++count;
      }
    }
  }
  std::cout << "count = " << count << '\n';

  t = 0;
  count = 0;
  while (t < 1000) {
    double e = f(1);
    if (0.1 < e) {
      t += 0.1;
    }
    else {
      t += e;
      if (t < 1000) {
        ++count;
      }
    }
  }
  std::cout << "count = " << count << '\n';

  t = 0;
  count = 0;
  while (t < 1000) {
    double e = f(1);
    if (0.0001 < e) {
      t += 0.0001;
    }
    else {
      t += e;
      if (t < 1000) {
        ++count;
      }
    }
  }
  std::cout << "count = " << count << '\n';

  return 0;
}
