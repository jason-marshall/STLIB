// -*- C++ -*-

#include "stlib/numerical/optimization/staticDimension/PenaltyQuasiNewton.h"

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
  typedef PenaltyQuasiNewton<3, Function, Constraint> PenaltyMethod;
  typedef PenaltyMethod::point_type point_type;
  {
    std::cout.precision(16);
    Function function;
    Constraint constraint;

    std::array<double, 4> mce = {{1e-2, 1e-4, 1e-6, 1e-8}};
    for (std::size_t i = 0; i != mce.size(); ++i) {
      const double max_constraint_error = mce[i];
      PenaltyQuasiNewton<3, Function, Constraint>
      method(function, constraint, max_constraint_error);

      point_type x = {{2. / 3., 3. / 7., 7. / 11.}};
      double value;
      std::size_t num_iterations;
      try {
        method.find_minimum(x, value, num_iterations);
      }
      catch (...) {
        std::cerr << "Error encountered.\nContinuing.\n";
      }
      std::cout << "max constraint error = " << max_constraint_error
                << '\n'
                << "penalty parameter = "
                << method.penalty_parameter() << '\n'
                << "num iterations = " << num_iterations << '\n'
                << "x = " << x << '\n'
                << "function(x) = " << function(x) << '\n'
                << "constraint(x) = " << constraint(x) << '\n'
                << '\n';
      assert(constraint(x) <= max_constraint_error);
    }
  }

  return 0;
}
