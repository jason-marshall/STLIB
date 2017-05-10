// -*- C++ -*-

#include "stlib/numerical/random/hypoexponential/HypoexponentialDistributionDistinctDynamicMinimumParameters.h"

#include <iostream>
#include <vector>

using namespace stlib;

int
main()
{
  typedef numerical::HypoexponentialDistributionDistinctDynamicMinimumParameters
  Hypoexponential;

  const double Eps = std::numeric_limits<double>::epsilon();

  // Empty.
  {
    Hypoexponential x(1);
    assert(x.ccdf(0) == 1);
    assert(x.ccdf(-1) == 1);
    assert(x.ccdf(1) == 1);
  }

  //
  // insert()
  //

  // One parameter.
  {
    Hypoexponential x(1);
    // Inserted.
    assert(x.insert(1) == std::numeric_limits<double>::max());
    assert(x.ccdf(0) == 1);
    assert(x.ccdf(-1) == 1);
    assert(x.ccdf(1) == std::exp(-1));
    // Not inserted.
    assert(x.insert(0.5) == 0.5);
    assert(x.ccdf(1) == std::exp(-1));
    std::cout << "One parameter.\n";
    for (double t = 1; t <= 1024; t *= 2) {
      const double ccdf = x.ccdf(t);
      const bool isNonzero = x.isCcdfNonzero(t);
      if (ccdf > Eps) {
        assert(isNonzero);
      }
      else if (ccdf < Eps * Eps) {
        assert(! isNonzero);
      }
      std::cout << t << ' ' << ccdf << ' ' << isNonzero << '\n';
    }
  }

  // Two parameters.
  {
    Hypoexponential x(2);
    // Inserted.
    assert(x.insert(1) == std::numeric_limits<double>::max());
    // Not inserted.
    assert(x.insert(1) == 1);
    assert(x.ccdf(0) == 1);
    assert(x.ccdf(-1) == 1);
    assert(x.ccdf(1) == std::exp(-1));
    // Not inserted.
    {
      const double p = 1 + 10 * std::numeric_limits<double>::epsilon();
      assert(x.insert(p) == p);
    }
    // Inserted.
    assert(x.insert(2) == std::numeric_limits<double>::max());
    assert(std::abs(x.ccdf(1) - (2 * std::exp(-1) - std::exp(-2))) <
           10 * std::numeric_limits<double>::epsilon());
    // Not inserted.
    assert(x.insert(0.5) == 0.5);
    assert(std::abs(x.ccdf(1) - (2 * std::exp(-1) - std::exp(-2))) <
           10 * std::numeric_limits<double>::epsilon());
    assert(x.ccdf(1e6) < std::numeric_limits<double>::epsilon());
    std::cout << "\nTwo parameters.\n";
    for (double t = 1; t <= 1024; t *= 2) {
      const double ccdf = x.ccdf(t);
      const bool isNonzero = x.isCcdfNonzero(t);
      if (ccdf > Eps) {
        assert(isNonzero);
      }
      else if (ccdf < Eps * Eps) {
        assert(! isNonzero);
      }
      std::cout << t << ' ' << ccdf << ' ' << isNonzero << '\n';
    }
  }

  // Ten parameters.
  {
    Hypoexponential x(10);
    for (std::size_t i = 0; i != 10; ++i) {
      assert(x.insert(i + 1) == std::numeric_limits<double>::max());
    }
    std::cout << "\nTen parameters.\n";
    for (double t = 1; t <= 1024; t *= 2) {
      const double ccdf = x.ccdf(t);
      const bool isNonzero = x.isCcdfNonzero(t);
      if (ccdf > Eps) {
        assert(isNonzero);
      }
      else if (ccdf < Eps * Eps) {
        assert(! isNonzero);
      }
      std::cout << t << ' ' << ccdf << ' ' << isNonzero << '\n';
    }
  }

  //
  // insertOrReplace()
  //

  // One parameter.
  {
    Hypoexponential x(1);
    // Inserted.
    assert(x.insertOrReplace(1) == std::numeric_limits<double>::max());
    // Not inserted.
    assert(x.insertOrReplace(1) == 1);
    assert(x.ccdf(0) == 1);
    assert(x.ccdf(-1) == 1);
    assert(x.ccdf(1) == std::exp(-1));
    // Inserted.
    assert(x.insertOrReplace(0.5) == 1);
    assert(x.ccdf(1) == std::exp(-0.5));
  }

  // Two parameters.
  {
    Hypoexponential x(2);
    // Inserted.
    assert(x.insertOrReplace(1) == std::numeric_limits<double>::max());
    // Not inserted.
    assert(x.insertOrReplace(1) == 1);
    assert(x.ccdf(0) == 1);
    assert(x.ccdf(-1) == 1);
    assert(x.ccdf(1) == std::exp(-1));
    // Not inserted.
    {
      const double p = 1 + 10 * std::numeric_limits<double>::epsilon();
      assert(x.insertOrReplace(p) == p);
    }
    // Inserted.
    assert(x.insertOrReplace(2) == std::numeric_limits<double>::max());
    assert(std::abs(x.ccdf(1) - (2 * std::exp(-1) - std::exp(-2))) <
           10 * std::numeric_limits<double>::epsilon());
    // Not inserted.
    assert(x.insertOrReplace(3) == 3);
    // Inserted.
    assert(x.insertOrReplace(0.5) == 2);
    assert(std::abs(x.ccdf(1) - (2 * std::exp(-0.5) - std::exp(-1))) <
           10 * std::numeric_limits<double>::epsilon());
    assert(x.ccdf(1e6) < std::numeric_limits<double>::epsilon());
  }

  return 0;
}
