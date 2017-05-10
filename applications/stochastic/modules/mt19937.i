// -*- C++ -*-

%module mt19937

%{
#include "../../../src/stochastic/api.h"
%}

namespace stochastic {

  extern
  unsigned
  generateMt19937State(unsigned seed, unsigned state[]);

}
