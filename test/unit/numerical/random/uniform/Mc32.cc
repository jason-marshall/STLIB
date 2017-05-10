// -*- C++ -*-

#include "stlib/numerical/random/uniform/DiscreteUniformGeneratorMc32.h"

#include <iostream>

#include <cassert>

// This file does not actually test the random number generators.  It is just
// for checking that the code will compile.

using namespace stlib;

int
main()
{
  typedef numerical::DiscreteUniformGeneratorMc32<> Generator;
  typedef Generator::result_type Integer;
  {
    // Default constructor.
    Generator f;
    f.seed(1);
    const Integer u = f();
    std::cout << "Random integer = " << u << "\n";

    {
      // Copy constructor.
      Generator g(f);
      assert(g() == f());
    }
    {
      // Assignment operator.
      Generator g;
      g = f;
      assert(g() == f());
    }
  }
  {
    // Construct with a seed.
    Generator f(1);
  }
#if 0
  // It appears that for odd seeds, the period is 2^30.
  // Check the period.
  {
    // Default constructor.
    for (unsigned seed = 1; seed <= 10; ++seed) {
      Generator f(seed);
      unsigned count = 1;
      while (f() != seed) {
        ++count;
      }
      std::cout << count << "\n";
    }
  }
#endif

  return 0;
}
