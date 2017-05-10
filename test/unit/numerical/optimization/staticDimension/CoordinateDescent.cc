// -*- C++ -*-

#include "stlib/numerical/optimization/staticDimension/CoordinateDescent.h"

#include <iostream>
#include <functional>

using namespace stlib;
using namespace numerical;

typedef std::array<double, 3> point_type;

class Quadratic :
  public std::unary_function<point_type, double>
{
public:
  double operator()(const point_type& x) const
  {
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + 1;
  }
};

class TetMesh
{
private:

  // Copy constructor not implemented.
  TetMesh(const TetMesh&);

  // Assignment operator not implemented.
  TetMesh&
  operator=(const TetMesh&);

public:
  TetMesh() {}
  double quality(const point_type& x) const
  {
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + 1;
  }
};

class TetMeshQuality :
  public std::unary_function<point_type, double>
{
private:

  TetMesh& _tm;

public:
  TetMeshQuality(TetMesh& tm) :
    _tm(tm) {}
  double
  operator()(const point_type& x) const
  {
    return _tm.quality(x);
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
};

int
main()
{
  {
    Quadratic function;
    CoordinateDescent<3, Quadratic> opt(function);
    point_type x = {{2. / 3., 3. / 5., 5. / 7.}};
    double value;
    std::size_t num_steps;
    bool result = opt.find_minimum(x, value, num_steps);
    std::cout << "Result = " << result << '\n'
              << "Minimum point = " << x << '\n'
              << "Minimum value = " << value << '\n'
              << "Number of steps = " << num_steps << '\n'
              << "Number of function calls = " << opt.num_function_calls()
              << '\n' << '\n';
  }
  {
    TetMesh tm;
    TetMeshQuality function(tm);
    CoordinateDescent<3, TetMeshQuality> opt(function, 0.1, 0.01);
    point_type x = {{2. / 3., 3. / 5., 5. / 7.}};
    double value;
    std::size_t num_steps;
    bool result = opt.find_minimum(x, value, num_steps);
    std::cout << "Result = " << result << '\n'
              << "Minimum point = " << x << '\n'
              << "Minimum value = " << value << '\n'
              << "Number of steps = " << num_steps << '\n'
              << "Number of function calls = " << opt.num_function_calls()
              << '\n' << '\n';
  }
  {
    Banana function;
    const double initial_step_size = 0.1;
    const double final_step_size = 0.01;
    CoordinateDescent<2, Banana> opt(function, initial_step_size,
                                     final_step_size);
    std::array<double, 2> x = {{2. / 3., 5. / 7.}};
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
    CoordinateDescent<2, Banana> opt(function, initial_step_size,
                                     final_step_size);
    std::array<double, 2> x = {{2. / 3., 5. / 7.}};
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

  return 0;
}
