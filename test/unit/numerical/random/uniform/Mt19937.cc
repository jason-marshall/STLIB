// -*- C++ -*-

#include "stlib/numerical/random/uniform/DiscreteUniformGeneratorMt19937.h"

#include <sstream>
#include <cassert>

// This file does not actually test the random number generators.  It is just
// for checking that the code will compile.

using namespace stlib;

int
main()
{
  typedef numerical::DiscreteUniformGeneratorMt19937 Generator;
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
    {
      // I/O.
      std::stringstream s;
      s << f;
      Generator g;
      s >> g;
      assert(g() == f());
    }
  }
  {
    // Construct with a seed.
    Generator f(1);
  }

  return 0;
}
