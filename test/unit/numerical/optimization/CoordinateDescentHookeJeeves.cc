// -*- C++ -*-

#include "stlib/numerical/optimization/CoordinateDescentHookeJeeves.h"
#include "stlib/ext/vector.h"

#include <iostream>
#include <functional>

USING_STLIB_EXT_VECTOR_MATH_OPERATORS;
USING_STLIB_EXT_VECTOR_IO_OPERATORS;
using namespace stlib;
using namespace numerical;

typedef std::vector<double> Vector;

class Quadratic :
  public std::unary_function<Vector, double>
{
public:
  double operator()(const Vector& x) const
  {
    return stlib::ext::dot(x, x) + 1;
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

  double quality(const Vector& x) const
  {
    return stlib::ext::dot(x, x) + 1;
  }
};

class TetMeshQuality :
  public std::unary_function<Vector, double>
{
private:

  const TetMesh& _tm;

public:
  TetMeshQuality(TetMesh& tm) :
    _tm(tm)
  {
  }

  double
  operator()(const Vector& x) const
  {
    return _tm.quality(x);
  }
};

// Rosenbrocks classic parabolic valley ("banana") function.
class Banana :
  public std::unary_function<Vector, double>
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
    CoordinateDescentHookeJeeves<Quadratic> opt(function);
    Vector x(3);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    x[2] = 5. / 7.;
    const double value = opt.minimize(&x);
    std::cout << "Minimum point = " << x
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    const double delta = opt.getFinalStepSize()
                         / opt.getStepSizeReductionFactor();
    Vector y(x.size());
    for (std::size_t d = 0; d != x.size(); ++d) {
      y = x;
      y[d] += delta;
      assert(function(x) <= function(y));
      y[d] -= 2 * delta;
      assert(function(x) <= function(y));
    }
  }
  {
    TetMesh tm;
    TetMeshQuality function(tm);
    CoordinateDescentHookeJeeves<TetMeshQuality> opt(function, 0.1, 0.01);
    Vector x(3);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    x[2] = 5. / 7.;
    const double value = opt.minimize(&x);
    std::cout << "Minimum point = " << x
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    const double delta = opt.getFinalStepSize()
                         / opt.getStepSizeReductionFactor();
    Vector y(x.size());
    for (std::size_t d = 0; d != x.size(); ++d) {
      y = x;
      y[d] += delta;
      assert(function(x) <= function(y));
      y[d] -= 2 * delta;
      assert(function(x) <= function(y));
    }
  }
  {
    Banana function;
    const double initialStepSize = 0.1;
    const double finalStepSize = 0.01;
    CoordinateDescentHookeJeeves<Banana> opt(function, initialStepSize,
        finalStepSize);
    Vector x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    const double value = opt.minimize(&x);
    std::cout << "\nBanana:\n"
              << "Initial step size = " << initialStepSize << '\n'
              << "Final step size = " << finalStepSize << '\n'
              << "Minimum point = " << x << '\n'
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    const double delta = opt.getFinalStepSize()
                         / opt.getStepSizeReductionFactor();
    Vector y(x.size());
    for (std::size_t d = 0; d != x.size(); ++d) {
      y = x;
      y[d] += delta;
      assert(function(x) <= function(y));
      y[d] -= 2 * delta;
      assert(function(x) <= function(y));
    }
  }
  {
    Banana function;
    const double initialStepSize = 0.1;
    const double finalStepSize = 1e-6;
    CoordinateDescentHookeJeeves<Banana> opt(function, initialStepSize,
        finalStepSize);
    Vector x(2);
    x[0] = 2. / 3.;
    x[1] = 3. / 5.;
    const double value = opt.minimize(&x);
    std::cout << "\nBanana:\n"
              << "Initial step size = " << initialStepSize << '\n'
              << "Final step size = " << finalStepSize << '\n'
              << "Minimum point = " << x << '\n'
              << "Minimum value = " << value << '\n'
              << "Number of function calls = " << opt.numFunctionCalls()
              << '\n' << '\n';
    const double delta = opt.getFinalStepSize()
                         / opt.getStepSizeReductionFactor();
    Vector y(x.size());
    for (std::size_t d = 0; d != x.size(); ++d) {
      y = x;
      y[d] += delta;
      assert(function(x) <= function(y));
      y[d] -= 2 * delta;
      assert(function(x) <= function(y));
    }
  }

  return 0;
}
