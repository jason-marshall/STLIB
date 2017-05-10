// -*- C++ -*-

#include "stlib/numerical/random/uniform/DiscreteUniformGeneratorTt800.h"

#include <iostream>

// This file does not actually test the random number generators.  It is just
// for checking that the code will compile.

using namespace stlib;

int
main()
{
  typedef numerical::DiscreteUniformGeneratorTt800 Generator;
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

  return 0;
}
