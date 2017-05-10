// -*- C++ -*-

#include "stlib/numerical/optimization/ObjectiveFunction.h"
#include "stlib/ext/vector.h"

#include <iostream>
#include <limits>

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

// Rosenbrocks classic parabolic valley ("banana") function.
class Banana :
  public std::unary_function<std::vector<double>, double>
{
public:

  result_type
  operator()(const argument_type& x) const
  {
    const result_type a = x[0];
    const result_type b = x[1];
    const result_type c = 100.0 * (b - (a * a)) * (b - (a * a));
    return c + ((1.0 - a) * (1.0 - a));
  }

  result_type
  operator()(const argument_type& x, argument_type* gradient) const
  {
    (*gradient)[0] = -2. * (1. - x[0]) - 400. * x[0] * (-x[0] * x[0] + x[1]);
    (*gradient)[1] = 200. * (-x[0] * x[0] + x[1]);
    const result_type a = x[0];
    const result_type b = x[1];
    const result_type c = 100.0 * (b - (a * a)) * (b - (a * a));
    return c + ((1.0 - a) * (1.0 - a));
  }
};

int
main()
{
  {
    Quadratic function;
    numerical::ObjectiveFunction<Quadratic>
    f(function, std::numeric_limits<std::size_t>::max());
    f.resetNumFunctionCalls();
    std::vector<double> x(3);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    x[2] = 5. / 7.;
    assert(f(x) == function(x));
    std::vector<double> g1(3), g2(3);
    f(x, &g1);
    function(x, &g2);
    assert(g1 == g2);
  }
#if !(defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ <= 4)
  {
    // Verify that an exception is thrown when the maximum number of steps
    // is exceeded.
    Quadratic function;
    numerical::ObjectiveFunction<Quadratic> f(function, 0);
    f.resetNumFunctionCalls();
    std::vector<double> x(3);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    x[2] = 5. / 7.;

    bool caught = false;
    try {
      f(x);
    }
    catch (numerical::OptMaxObjFuncCallsError& error) {
      caught = true;
      std::cout << error.what() << '\n';
    }
    assert(caught);
  }
#endif
  {
    Banana function;
    numerical::ObjectiveFunction<Banana>
    f(function, std::numeric_limits<std::size_t>::max());
    f.resetNumFunctionCalls();
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    assert(f(x) == function(x));
    std::vector<double> g1(3), g2(3);
    f(x, &g1);
    function(x, &g2);
    assert(g1 == g2);
  }

  return 0;
}
