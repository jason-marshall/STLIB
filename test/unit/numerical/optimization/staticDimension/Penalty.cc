// -*- C++ -*-

#include "stlib/numerical/optimization/staticDimension/Penalty.h"

#include <iostream>
#include <functional>

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
};

class Constraint :
  public std::unary_function<point_type, number_type>
{
public:
  result_type operator()(const argument_type& x) const
  {
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 3;
  }
};

int
main()
{
  typedef Penalty<3, Function, Constraint> PenaltyMethod;
  typedef PenaltyMethod::point_type point_type;
  {
    std::cout.precision(16);
    Function function;
    Constraint constraint;

    const double initial_step_size = 0.1;
    const std::array<double, 3> fss = {{1e-4, 1e-6, 1e-8}};
    const std::array<std::size_t, 3> maxFunctionCalls = {{1000, 1000000, 1000000}};
    for (std::size_t i = 0; i != fss.size(); ++i) {
      const double final_step_size = fss[i];
      const double max_constraint_error = 100 * fss[i];
      Penalty<3, Function, Constraint>
      method(function, constraint,
             initial_step_size, final_step_size, max_constraint_error,
             maxFunctionCalls[i]);

      point_type x = {{2. / 3., 3. / 7., 7. / 11.}};
      try {
        method.find_minimum(x);
      }
      catch (...) {
        std::cerr << "Error encountered.\nContinuing.\n";
      }
      std::cout << "final step size = " << final_step_size << '\n'
                << "max constraint error = " << max_constraint_error
                << '\n'
                << "penalty parameter = "
                << method.penalty_parameter() << '\n'
                << "x = " << x << '\n'
                << "function(x) = " << function(x) << '\n'
                << "constraint(x) = " << constraint(x) << '\n'
                << '\n';
    }
  }

  return 0;
}
