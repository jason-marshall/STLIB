// -*- C++ -*-

#include "stlib/numerical/optimization/staticDimension/QuasiNewton.h"

#include <iostream>
#include <functional>

using namespace stlib;
using namespace numerical;

class Quadratic :
  public std::unary_function<std::array<double, 3>, double>
{
public:
  result_type
  operator()(const argument_type& x) const
  {
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + 1;
  }
  void
  gradient(const argument_type& x, argument_type& gradient) const
  {
    gradient[0] = 2 * x[0];
    gradient[1] = 2 * x[1];
    gradient[2] = 2 * x[2];
  }
};

// Rosenbrocks classic parabolic valley ("banana") function.
class Banana :
  public std::unary_function< std::array<double, 2>, double >
{
private:

  mutable result_type a, b, c;

public:
  result_type
  operator()(const argument_type& x) const
  {
    a = x[0];
    b = x[1];
    c = 100.0 * (b - (a * a)) * (b - (a * a));
    return (c + ((1.0 - a) * (1.0 - a)));
  }
  void
  gradient(const argument_type& x, argument_type& gradient) const
  {
    gradient[0] = - 400.0 * x[0] * (x[1] - x[0] * x[0]) + 2.0 * (x[0] - 1.0);
    gradient[1] = 200.0 * (x[1] - x[0] * x[0]);
  }
};

int
main()
{
  std::cout.precision(16);
  {
    Quadratic function;
    QuasiNewton<3, Quadratic> opt(function);
    std::array<double, 3> x = {{2. / 3., 3. / 5., 5. / 7.}};
    double value;
    std::size_t num_steps;
    opt.find_minimum(x, value, num_steps);
    std::cout << "Minimum point = " << x << '\n'
              << "Minimum value = " << value << '\n'
              << "Number of steps = " << num_steps << '\n'
              << "Number of function calls = " << opt.num_function_calls()
              << '\n' << '\n';
  }

  {
    Banana function;
    QuasiNewton<2, Banana> opt(function);
    std::array<double, 2> x = {{2. / 3., 5. / 7.}};
    double value;
    std::size_t num_steps;
    const double max_step = 1.0;
    const double x_tolerance = 1e-4;
    const double gradient_tolerance = 1e-16;
    opt.find_minimum(x, value, num_steps,
                     max_step, x_tolerance, gradient_tolerance);
    std::cout << "\nBanana:\n"
              << "Minimum point = " << x << '\n'
              << "Minimum value = " << value << '\n'
              << "Number of steps = " << num_steps << '\n'
              << "x tolerance = " << x_tolerance << '\n'
              << "gradient tolerance = " << gradient_tolerance << '\n'
              << "Number of function calls = " << opt.num_function_calls()
              << '\n' << '\n';
  }
  {
    Banana function;
    QuasiNewton<2, Banana> opt(function);
    std::array<double, 2> x = {{2. / 3., 5. / 7.}};
    double value;
    std::size_t num_steps;
    const double max_step = 1.0;
    const double x_tolerance = 1e-16;
    const double gradient_tolerance = 1e-4;
    opt.find_minimum(x, value, num_steps,
                     max_step, x_tolerance, gradient_tolerance);
    std::cout << "\nBanana:\n"
              << "Minimum point = " << x << '\n'
              << "Minimum value = " << value << '\n'
              << "Number of steps = " << num_steps << '\n'
              << "x tolerance = " << x_tolerance << '\n'
              << "gradient tolerance = " << gradient_tolerance << '\n'
              << "Number of function calls = " << opt.num_function_calls()
              << '\n' << '\n';
  }

  {
    Banana function;
    QuasiNewton<2, Banana> opt(function);
    std::array<double, 2> x = {{2. / 3., 5. / 7.}};
    double value;
    std::size_t num_steps;
    const double max_step = 1.0;
    const double x_tolerance = 1e-8;
    const double gradient_tolerance = 1e-16;
    opt.find_minimum(x, value, num_steps,
                     max_step, x_tolerance, gradient_tolerance);
    std::cout << "\nBanana:\n"
              << "Minimum point = " << x << '\n'
              << "Minimum value = " << value << '\n'
              << "Number of steps = " << num_steps << '\n'
              << "x tolerance = " << x_tolerance << '\n'
              << "gradient tolerance = " << gradient_tolerance << '\n'
              << "Number of function calls = " << opt.num_function_calls()
              << '\n' << '\n';
  }
  {
    Banana function;
    QuasiNewton<2, Banana> opt(function);
    std::array<double, 2> x = {{2. / 3., 5. / 7.}};
    double value;
    std::size_t num_steps;
    const double max_step = 1.0;
    const double x_tolerance = 1e-16;
    const double gradient_tolerance = 1e-8;
    opt.find_minimum(x, value, num_steps,
                     max_step, x_tolerance, gradient_tolerance);
    std::cout << "\nBanana:\n"
              << "Minimum point = " << x << '\n'
              << "Minimum value = " << value << '\n'
              << "Number of steps = " << num_steps << '\n'
              << "x tolerance = " << x_tolerance << '\n'
              << "gradient tolerance = " << gradient_tolerance << '\n'
              << "Number of function calls = " << opt.num_function_calls()
              << '\n' << '\n';
  }

  {
    Banana function;
    QuasiNewton<2, Banana> opt(function);
    std::array<double, 2> x = {{2. / 3., 5. / 7.}};
    double value;
    std::size_t num_steps;
    const double max_step = 1.0;
    const double x_tolerance = 1e-16;
    const double gradient_tolerance = 1e-16;
    opt.find_minimum(x, value, num_steps,
                     max_step, x_tolerance, gradient_tolerance);
    std::cout << "\nBanana:\n"
              << "Minimum point = " << x << '\n'
              << "Minimum value = " << value << '\n'
              << "Number of steps = " << num_steps << '\n'
              << "x tolerance = " << x_tolerance << '\n'
              << "gradient tolerance = " << gradient_tolerance << '\n'
              << "Number of function calls = " << opt.num_function_calls()
              << '\n' << '\n';
  }
  /*
  {
    Banana function;
    const double initial_step_size = 0.1;
    const double final_step_size = 0.01;
    QuasiNewton<2, Banana> opt(function, initial_step_size,
   			      final_step_size);
    std::array<double,2> x(2./3., 5./7.);
    double value;
    std::size_t num_steps;
    bool result = opt.find_minimum(x, value, num_steps);
    std::cout << "\nBanana:\n"
         << "Initial step size = " << initial_step_size << '\n'
         << "Final step size = " << final_step_size << '\n'
         << "Result = " << result << '\n'
         << "Minimum point = " << x << '\n'
         << "Minimum value = " << value << '\n'
         << "Number of steps = " << num_steps << '\n'
         << "Number of function calls = " << opt.num_function_calls()
         << '\n' << '\n';
  }
  {
    Banana function;
    const double initial_step_size = 0.1;
    const double final_step_size = 1e-6;
    QuasiNewton<2, Banana> opt(function, initial_step_size,
   			      final_step_size);
    std::array<double,2> x(2./3., 5./7.);
    double value;
    std::size_t num_steps;
    bool result = opt.find_minimum(x, value, num_steps);
    std::cout << "\nBanana:\n"
         << "Initial step size = " << initial_step_size << '\n'
         << "Final step size = " << final_step_size << '\n'
         << "Result = " << result << '\n'
         << "Minimum point = " << x << '\n'
         << "Minimum value = " << value << '\n'
         << "Number of steps = " << num_steps << '\n'
         << "Number of function calls = " << opt.num_function_calls()
         << '\n' << '\n';
  }
  */

  return 0;
}
