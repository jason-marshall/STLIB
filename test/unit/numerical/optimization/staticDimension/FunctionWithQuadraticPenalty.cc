// -*- C++ -*-

#include "stlib/numerical/optimization/staticDimension/FunctionWithQuadraticPenalty.h"

#include "stlib/ext/array.h"

#include <iostream>
#include <functional>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;
using namespace numerical;

typedef double number_type;
typedef std::array<double, 3> point_type;

class Function :
  public std::unary_function<point_type, number_type>
{
public:
  result_type operator()(const argument_type& x) const
  {
    return x[0] + x[1] + x[2];
  }
  void gradient(const argument_type& /*x*/, argument_type& y) const
  {
    std::fill(y.begin(), y.end(), 1);
  }
};

class Constraint :
  public std::unary_function<point_type, number_type>
{
public:
  result_type operator()(const argument_type& x) const
  {
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 3;
  }
  void gradient(const argument_type& x, argument_type& y) const
  {
    y = 2.0 * x;
  }
};

int
main()
{
  {
    Function function;
    Constraint constraint;
    FunctionWithQuadraticPenalty<3, Function, Constraint>
    f(function, constraint);
    std::array<point_type, 4> points;
    std::fill(points[0].begin(), points[0].end(), -1);
    std::fill(points[1].begin(), points[1].end(), 0);
    std::fill(points[2].begin(), points[2].end(), 1);
    std::fill(points[3].begin(), points[3].end(), 2);
    point_type grad;

    std::cout << "penalty parameter = " << f.penalty_parameter() << '\n';
    std::cout << "reduction factor = " << f.reduction_factor() << '\n';
    for (std::size_t i = 0; i != points.size(); ++i) {
      std::cout << "f(" << points[i] << ") = " << f(points[i]) << '\n';
      f.gradient(points[i], grad);
      std::cout << "D f(" << points[i] << ") = " << grad << '\n';
    }

    f.increase_penalty();
    std::cout << "penalty parameter = " << f.penalty_parameter() << '\n';
    std::cout << "reduction factor = " << f.reduction_factor() << '\n';
    for (std::size_t i = 0; i != points.size(); ++i) {
      std::cout << "f(" << points[i] << ") = " << f(points[i]) << '\n';
      f.gradient(points[i], grad);
      std::cout << "D f(" << points[i] << ") = " << grad << '\n';
    }
  }

  return 0;
}
