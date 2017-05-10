// -*- C++ -*-

#include "stlib/numerical/random/uniform/DiscreteUniformGeneratorNr2.h"

#include <iostream>

// This file does not actually test the random number generators.  It is just
// for checking that the code will compile.

using namespace stlib;

int
main()
{
  typedef numerical::DiscreteUniformGeneratorNr2 Generator;
  typedef Generator::result_type Integer;
  {
    // Default constructor.
    Generator f;
    f.seed(1);
    const Integer i = f();
    std::cout << "Random integer = " << i << "\n";

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
