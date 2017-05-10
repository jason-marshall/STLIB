// -*- C++ -*-

#include "stlib/numerical/random/hypoexponential/HypoexponentialDistributionNormalApproximation.h"

#include <iostream>
#include <vector>


using namespace stlib;

int
main()
{
  typedef numerical::HypoexponentialDistributionNormalApproximation
  Hypoexponential;

  const double Eps = std::numeric_limits<double>::epsilon();

  {
    Hypoexponential x(0.1);
    assert(! x.isValid());

    // Constant.
    std::size_t count = 0;
    while (! x.isValid()) {
      x.insertInverse(1 / 1.);
      ++count;
    }
    assert(count == 170);

    // Decreasing.
    x.clear();
    count = 0;
    while (! x.isValid()) {
      x.insertInverse(count + 1);
      ++count;
    }
    assert(count == 285);

    // Increasing.
    x.clear();
    count = 0;
    while (! x.isValid()) {
      x.insertInverse(1. / std::sqrt(count + 1));
      ++count;
    }
    assert(count == 19367);

    x.clear();
    x.insertInverse(1. / 1);

    assert(std::abs(x.ccdf(1 + std::sqrt(2) * 5.74587239219118) - Eps) / Eps
           < 100 * Eps);
    assert(std::abs(x.ccdf(1) - 0.5) / 0.5   < 100 * Eps);
    assert(std::abs(x.ccdf(-10) - 1) / 1   < 100 * Eps);

    assert(x.isCcdfNonzero(-10));
    assert(x.isCcdfNonzero(0));
    assert(x.isCcdfNonzero(9));
    assert(! x.isCcdfNonzero(10));
    assert(! x.isCcdfNonzero(1e20));

    // Singular.
    x.clear();
    x.setMeanToInfinity();
    assert(x.ccdf(-1) == 1);
    assert(x.ccdf(1) == 1);
    assert(x.ccdf(1e20) == 1);
    assert(x.isCcdfNonzero(-1));
    assert(x.isCcdfNonzero(1));
    assert(x.isCcdfNonzero(1e20));
  }

  return 0;
}
