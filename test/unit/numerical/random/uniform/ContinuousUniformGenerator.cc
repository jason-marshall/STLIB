// -*- C++ -*-

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <iostream>
#include <cmath>
#include <cassert>

// This file does not actually test the random number generators.  It is just
// for checking that the code will compile.

using namespace stlib;

int
main()
{
  //
  // Open interval (0..1).
  //
  {
    typedef numerical::ContinuousUniformGeneratorOpen<> ContinuousGenerator;
    typedef ContinuousGenerator::DiscreteUniformGenerator DiscreteGenerator;
    typedef ContinuousGenerator::result_type Number;

    DiscreteGenerator discrete;
    ContinuousGenerator f(&discrete);
    f.seed(1);
    const Number u = f();
    assert(0 < u && u < 1);

    {
      // Copy constructor.
      ContinuousGenerator g(f);
    }
    {
      // Assignment operator.
      ContinuousGenerator g(&discrete);
      g = f;
    }
  }

  //
  // Open interval (0..1).
  //
  {
    typedef numerical::ContinuousUniformGeneratorClosed<> ContinuousGenerator;
    typedef ContinuousGenerator::DiscreteUniformGenerator DiscreteGenerator;
    typedef ContinuousGenerator::result_type Number;

    DiscreteGenerator discrete;
    ContinuousGenerator f(&discrete);
    f.seed(1);
    const Number u = f();
    assert(0 < u && u < 1);

    {
      // Copy constructor.
      ContinuousGenerator g(f);
    }
    {
      // Assignment operator.
      ContinuousGenerator g(&discrete);
      g = f;
    }
  }

  return 0;
}
