// -*- C++ -*-

#include "stlib/numerical/optimization/QuasiNewtonLBFGS.h"
#include "stlib/numerical/equality.h"

#include <iostream>

USING_STLIB_EXT_VECTOR_IO_OPERATORS;
USING_STLIB_EXT_VECTOR_MATH_OPERATORS;
using namespace stlib;

class Quadratic :
  public std::unary_function<std::vector<double>, double>
{
public:

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

class BananaPlusOne :
  public std::unary_function<std::vector<double>, double>
{
public:

  result_type
  operator()(const argument_type& x, argument_type* gradient) const
  {
    (*gradient)[0] = -2. * (1. - x[0]) - 400. * x[0] * (-x[0] * x[0] + x[1]);
    (*gradient)[1] = 200. * (-x[0] * x[0] + x[1]);
    const result_type a = x[0];
    const result_type b = x[1];
    const result_type c = 100.0 * (b - (a * a)) * (b - (a * a));
    return 1 + c + ((1.0 - a) * (1.0 - a));
  }
};

int
main()
{
  {
    Quadratic function;
    numerical::QuasiNewtonLBFGS<Quadratic> opt(function);
    assert(opt.getMaxObjFuncCalls() ==
           std::numeric_limits<std::size_t>::max());
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
#if !(defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ <= 4)
  {
    Quadratic function;
    numerical::QuasiNewtonLBFGS<Quadratic> opt(function);
    std::cout << "Reduce the maximum allowed function calls.\n";
    opt.setMaxObjFuncCalls(2);
    assert(opt.getMaxObjFuncCalls() == 2);
    std::vector<double> x(3);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    x[2] = 5. / 7.;
    double value;
    bool caught = false;
    try {
      value = opt.minimize(&x);
    }
    catch (numerical::OptMaxObjFuncCallsError& error) {
      caught = true;
      std::cout << error.what() << '\n';
    }
    assert(caught);
    std::vector<double> g(x.size());
    std::cout << "Minimum point = " << x
              << "Minimum value = " << value << '\n'
              << "Minimum value = " << opt.function()(x, &g) << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
  }
#endif
  std::cout << "------------------------------------------------------------\n"
            << "Banana minimizations:\n\n";
  {
    Banana function;
    numerical::QuasiNewtonLBFGS<Banana> opt(function);
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    const double value = opt.minimize(&x);
    std::vector<double> d = x;
    d -= 1.;
    std::cout << "Default settings.\n"
              << "Minimum point = " << x << '\n'
              << "Difference from exact minimum point = " << d << '\n'
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    std::vector<double> g(2);
    function(x, &g);
    const double xnorm = std::sqrt(stlib::ext::dot(x, x));
    const double gnorm = std::sqrt(stlib::ext::dot(g, g));
    assert(gnorm < opt.getRelativeGradientTolerance() * std::max(1., xnorm));
  }
  {
    Banana function;
    numerical::QuasiNewtonLBFGS<Banana> opt(function);
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    opt.setRelativeGradientTolerance(1e-2);
    const double value = opt.minimize(&x);
    std::cout << "Increase the relative gradient tolerance.\n"
              << "Gradient tolerance = "
              << opt.getRelativeGradientTolerance() << '\n'
              << "Minimum point = " << x
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    std::vector<double> g(2);
    function(x, &g);
    const double xnorm = std::sqrt(stlib::ext::dot(x, x));
    const double gnorm = std::sqrt(stlib::ext::dot(g, g));
    assert(gnorm < opt.getRelativeGradientTolerance() * std::max(1., xnorm));
  }
  {
    Banana function;
    numerical::QuasiNewtonLBFGS<Banana> opt(function);
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    opt.setRelativeGradientTolerance(1e-8);
    const double value = opt.minimize(&x);
    std::cout << "Reduce the relative gradient tolerance.\n"
              << "Gradient tolerance = "
              << opt.getRelativeGradientTolerance() << '\n'
              << "Minimum point = " << x
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    std::vector<double> g(2);
    function(x, &g);
    const double xnorm = std::sqrt(stlib::ext::dot(x, x));
    const double gnorm = std::sqrt(stlib::ext::dot(g, g));
    assert(gnorm < opt.getRelativeGradientTolerance() * std::max(1., xnorm));
  }
  {
    Banana function;
    numerical::QuasiNewtonLBFGS<Banana> opt(function);
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    opt.setRmsGradientTolerance(1e-3);
    const double value = opt.minimize(&x);
    std::cout << "Use the RMS gradient tolerance.\n"
              << "RMS Gradient tolerance = "
              << opt.getRmsGradientTolerance() << '\n'
              << "Minimum point = " << x
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    std::vector<double> g(2);
    function(x, &g);
    const double xnorm = std::sqrt(stlib::ext::dot(x, x));
    const double gnorm = std::sqrt(stlib::ext::dot(g, g));
    assert(gnorm < opt.getRelativeGradientTolerance() * std::max(1., xnorm) ||
           gnorm / std::sqrt(double(g.size())) <
           opt.getRmsGradientTolerance());
  }
  {
    Banana function;
    numerical::QuasiNewtonLBFGS<Banana> opt(function);
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    opt.setMaxGradientTolerance(1e-3);
    const double value = opt.minimize(&x);
    std::cout << "Use the maximum gradient tolerance.\n"
              << "Maximum Gradient tolerance = "
              << opt.getMaxGradientTolerance() << '\n'
              << "Minimum point = " << x
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    std::vector<double> g(2);
    function(x, &g);
    const double xnorm = std::sqrt(stlib::ext::dot(x, x));
    const double gnorm = std::sqrt(stlib::ext::dot(g, g));
    bool isMaxCriterionSatisfied = true;
    for (std::size_t i = 0; i != g.size(); ++i) {
      if (std::abs(g[i]) >= opt.getMaxGradientTolerance()) {
        isMaxCriterionSatisfied = false;
      }
    }
    assert(gnorm < opt.getRelativeGradientTolerance() * std::max(1., xnorm) ||
           isMaxCriterionSatisfied);
  }
  {
    BananaPlusOne function;
    numerical::QuasiNewtonLBFGS<BananaPlusOne> opt(function);
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    opt.setNumberOfIterationsForRateOfDecrease(2);
    const double value = opt.minimize(&x);
    std::cout << "Rate of decrease tolerance = "
              << opt.getRateOfDecreaseTolerance() << '\n'
              << "Past iterations = "
              << opt.getNumberOfIterationsForRateOfDecrease() << '\n'
              << "Minimum point = " << x
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    for (std::size_t i = 0; i != x.size(); ++i) {
      assert(numerical::areEqual(x[i], 1., 1e12));
    }
  }
  {
    BananaPlusOne function;
    numerical::QuasiNewtonLBFGS<BananaPlusOne> opt(function);
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    opt.setNumberOfIterationsForRateOfDecrease(2);
    opt.setRateOfDecreaseTolerance(1e-3);
    const double value = opt.minimize(&x);
    std::cout << "Rate of decrease tolerance = "
              << opt.getRateOfDecreaseTolerance() << '\n'
              << "Past iterations = "
              << opt.getNumberOfIterationsForRateOfDecrease() << '\n'
              << "Minimum point = " << x
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    for (std::size_t i = 0; i != x.size(); ++i) {
      assert(numerical::areEqual(x[i], 1., 1e14));
    }
  }

  //
  // Exceptions.
  //
#if !(defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ <= 4)
  {
    Banana function;
    numerical::QuasiNewtonLBFGS<Banana> opt(function);
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    opt.setMaxTime(-1);
    bool caught = false;
    try {
      opt.minimize(&x);
    }
    catch (numerical::OptMaxTimeError&) {
      caught = true;
    }
    assert(caught);
  }
  {
    Banana function;
    numerical::QuasiNewtonLBFGS<Banana> opt(function);
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    opt.setMaxObjFuncCalls(1);
    bool caught = false;
    try {
      opt.minimize(&x);
    }
    catch (numerical::OptMaxObjFuncCallsError&) {
      caught = true;
    }
    assert(caught);
  }
  {
    Banana function;
    numerical::QuasiNewtonLBFGS<Banana> opt(function);
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    opt.setMaxObjFuncCalls(1);
    opt.disableMaxComputationExceptions();
    opt.minimize(&x);
  }
  {
    Banana function;
    numerical::QuasiNewtonLBFGS<Banana> opt(function);
    std::vector<double> x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    opt.setMaxObjFuncCalls(1);
    opt.disableExceptions();
    opt.minimize(&x);
  }
#endif

  return 0;
}
