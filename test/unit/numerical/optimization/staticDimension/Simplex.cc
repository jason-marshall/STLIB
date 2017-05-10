// -*- C++ -*-

#include "stlib/numerical/optimization/staticDimension/Simplex.h"

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

int
main()
{
  {
    Quadratic function;
    Simplex<3, Quadratic> simplex(function);
    using point_type = Simplex<3, Quadratic>::point_type;
    bool result = simplex.find_minimum(point_type{{2. / 3., 3. / 5., 5. / 7.}});
    std::cout << "Result = " << result << '\n'
              << "Minimum point = " << simplex.minimum_point() << '\n'
              << "Minimum value = " << simplex.minimum_value() << '\n'
              << "Number of function calls = " << simplex.num_function_calls()
              << '\n';
  }
  {
    TetMesh tm;
    TetMeshQuality function(tm);
    Simplex<3, TetMeshQuality> simplex(function, 0.01, 0.1);
    using point_type = Simplex<3, TetMeshQuality>::point_type;
    bool result = simplex.find_minimum(point_type{{2. / 3., 3. / 5., 5. / 7.}});
    std::cout << "Result = " << result << '\n'
              << "Minimum point = " << simplex.minimum_point() << '\n'
              << "Minimum value = " << simplex.minimum_value() << '\n'
              << "Number of function calls = " << simplex.num_function_calls()
              << '\n';
  }

  return 0;
}
