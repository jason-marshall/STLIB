// -*- C++ -*-

#include "stlib/numerical/optimization/FunctionOfSelectedCoordinates.h"
#include "stlib/ext/vector.h"

#include <iostream>
#include <functional>

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
  Quadratic function;
  std::vector<double> completeX(3);
  completeX[0] = 1;
  completeX[1] = 2;
  completeX[2] = 3;
  std::vector<std::size_t> indices(1, 0);
  numerical::FunctionOfSelectedCoordinates<Quadratic>
  f(function, completeX, indices);


  std::vector<double> completeGradient(3);
  std::vector<double> x(1);
  x[0] = completeX[indices[0]];
  std::vector<double> g(1);
  assert(f(x, &g) == function(completeX, &completeGradient));
  assert(g[0] == completeGradient[indices[0]]);

  x[0] = completeX[indices[0]] = 7;
  assert(f(x, &g) == function(completeX, &completeGradient));
  assert(g[0] == completeGradient[indices[0]]);

  return 0;
}
