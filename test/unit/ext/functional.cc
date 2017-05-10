// -*- C++ -*-

#include "stlib/ext/functional.h"

#include <cassert>
#include <cmath>

using namespace stlib;

struct Square : std::unary_function<float, float> {
  result_type
  operator()(argument_type x) const
  {
    return x * x;
  }
};

int
main()
{
  Square f;
  assert(ext::compose1(f, f)(2) == f(f(2)));

  return 0;
}
