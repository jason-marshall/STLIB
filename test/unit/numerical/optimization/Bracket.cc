// -*- C++ -*-

#include "stlib/numerical/optimization/Bracket.h"

#include <iostream>
#include <functional>

#include <cassert>

using namespace stlib;

class Quadratic :
  public std::unary_function<double, double>
{
public:
  double operator()(const double x) const
  {
    return x * x;
  }
};

int
main()
{
  Quadratic f;
  numerical::Bracket<Quadratic> b(f);

  b.bracket(-2, -1);
  assert(b.isValid());

  b.bracket(-1, 1);
  assert(b.isValid());

  b.bracket(-1, 0);
  assert(b.isValid());

  b.bracket(1, 2);
  assert(b.isValid());

  return 0;
}
