// -*- C++ -*-

#include "stlib/numerical/optimization/Brent.h"
#include "stlib/numerical/equality.h"

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
  numerical::Brent<Quadratic> b(f);
  double x, v;
  v = b.minimize(-2, -1, &x);
  std::cout << "x = " << x << '\n'
            << "v = " << v << '\n';
  assert(numerical::areEqual(x, 0., 1e4));
  assert(numerical::areEqual(v, 0., 1e4));

  v = b.minimize(2, 1, &x);
  std::cout << "x = " << x << '\n'
            << "v = " << v << '\n';
  assert(numerical::areEqual(x, 0., 1e4));
  assert(numerical::areEqual(v, 0., 1e4));

  return 0;
}
