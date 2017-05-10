// -*- C++ -*-

#include "stlib/numerical/optimization/ConjugateGradient.h"
#include "stlib/numerical/equality.h"

#include <iostream>

USING_STLIB_EXT_VECTOR_IO_OPERATORS;
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
    numerical::ConjugateGradient<Quadratic> opt(function);
    std::vector<double> x(3);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    x[2] = 5. / 7.;
    const double value = opt.minimize(&x);
    std::cout << "Minimum point = " << x
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    for (std::size_t i = 0; i != x.size(); ++i) {
      assert(numerical::areEqual(x[i], 0., 1e4));
    }
  }
  {
    Banana function;
    numerical::ConjugateGradient<Banana> opt(function);
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    const double value = opt.minimize(&x);
    std::cout << "\nBanana minimization with DBrent:\n"
              << "Minimum point = " << x << '\n'
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    for (std::size_t i = 0; i != x.size(); ++i) {
      assert(numerical::areEqual(x[i], 1., 1e4));
    }
  }
  {
    Banana function;
    numerical::ConjugateGradient<Banana, numerical::Brent> opt(function);
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    const double value = opt.minimize(&x);
    std::cout << "\nBanana minimization with Brent:\n"
              << "Minimum point = " << x << '\n'
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    for (std::size_t i = 0; i != x.size(); ++i) {
      assert(numerical::areEqual(x[i], 1., 1e4));
    }
  }

  return 0;
}
