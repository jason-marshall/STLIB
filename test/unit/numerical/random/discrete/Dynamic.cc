// -*- C++ -*-

#include "stlib/numerical/random/discrete/DiscreteGeneratorDynamic.h"

#include "stlib/ads/algorithm/statistics.h"

#include <iostream>
#include <vector>

USING_STLIB_EXT_VECTOR_IO_OPERATORS;
using namespace stlib;

int
main()
{
  typedef numerical::DiscreteGeneratorDynamic<> Generator;

  std::vector<double> pmf(10, 1.0);

  Generator::DiscreteUniformGenerator uniform;
  Generator f(&uniform);

  {
    // Copy constructor.
    Generator g(f);
  }
  {
    // Assignment operator.
    Generator g(0);
    g = f;
  }
  {
    // Seed.
    Generator g(&uniform);
    g.seed(1);
  }

  // Check the mean and variance.
  // CONTINUE: Numerically check the values instead of printing them.
  const std::size_t Size = 100000;

  //-------------------------------------------------------------------------
  std::cout << "First half.\n";
  for (std::size_t i = 0; i != 5; ++i) {
    f.insert(i);
  }

  std::vector<double> data(Size);
  for (std::size_t n = 0; n != Size; ++n) {
    f.addPmf(pmf, 1.0);
    data[n] = f();
  }
  double mean, variance;
  ads::computeMeanAndVariance(data.begin(), data.end(), &mean, &variance);
  std::cout << "Size = " << Size << ", mean = " << mean
            << ", variance = " << variance << "\n";

  std::vector<std::size_t> counts(pmf.size(), std::size_t(0));
  for (std::size_t n = 0; n != Size; ++n) {
    f.addPmf(pmf, 1.0);
    ++counts[f()];
  }
  std::cout << "PMF:\n" << pmf << "\nCounts:\n" << counts << "\n";

  //-------------------------------------------------------------------------
  std::cout << "All.\n";
  for (std::size_t i = 5; i != 10; ++i) {
    f.insert(i);
  }

  for (std::size_t n = 0; n != Size; ++n) {
    f.addPmf(pmf, 1.0);
    data[n] = f();
  }
  ads::computeMeanAndVariance(data.begin(), data.end(), &mean, &variance);
  std::cout << "Size = " << Size << ", mean = " << mean
            << ", variance = " << variance << "\n";

  std::fill(counts.begin(), counts.end(), 0);
  for (std::size_t n = 0; n != Size; ++n) {
    f.addPmf(pmf, 1.0);
    ++counts[f()];
  }
  std::cout << "PMF:\n" << pmf << "\nCounts:\n" << counts << "\n";

  //-------------------------------------------------------------------------
  std::cout << "Second half.\n";
  for (std::size_t i = 0; i != 5; ++i) {
    f.erase(i);
  }

  for (std::size_t n = 0; n != Size; ++n) {
    f.addPmf(pmf, 1.0);
    data[n] = f();
  }
  ads::computeMeanAndVariance(data.begin(), data.end(), &mean, &variance);
  std::cout << "Size = " << Size << ", mean = " << mean
            << ", variance = " << variance << "\n";

  std::fill(counts.begin(), counts.end(), 0);
  for (std::size_t n = 0; n != Size; ++n) {
    f.addPmf(pmf, 1.0);
    ++counts[f()];
  }
  std::cout << "PMF:\n" << pmf << "\nCounts:\n" << counts << "\n";

  return 0;
}
