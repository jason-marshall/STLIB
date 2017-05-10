// -*- C++ -*-

#include "stlib/numerical/optimization/FunctionOnLine.h"
#include "stlib/ext/vector.h"

#include <iostream>
#include <functional>

USING_STLIB_EXT_VECTOR_MATH_OPERATORS;
using namespace stlib;

class Quadratic :
  public std::unary_function<std::vector<double>, double>
{
public:

  result_type
  operator()(const argument_type& x) const
  {
    return stlib::ext::dot(x, x) + 1;
  }

  result_type
  operator()(const argument_type& x, argument_type* gradient) const
  {
    for (std::size_t i = 0; i != x.size(); ++i) {
      (*gradient)[i] = 2 * x[i];
    }
    return stlib::ext::dot(x, x) + 1;
  }
};

int
main()
{
  {
    Quadratic function;
    std::vector<double> base(3);
    base[0] = 1;
    base[1] = 0;
    base[2] = 0;
    std::vector<double> tangent(3);
    tangent[0] = 1;
    tangent[1] = 0;
    tangent[2] = 0;
    std::vector<double> gradient(3);
    numerical::FunctionOnLine<Quadratic> f(function, base, tangent);

    std::vector<double> x = base;
    assert(f(0) == function(x));
    function(x, &gradient);
    assert(f.derivative(0) == stlib::ext::dot(gradient, tangent));

    x += tangent;
    assert(f(1) == function(x));
    function(x, &gradient);
    assert(f.derivative(1) == stlib::ext::dot(gradient, tangent));
  }

  return 0;
}
