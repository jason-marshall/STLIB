// -*- C++ -*-

#include "stlib/numerical/derivative/centered_difference.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

// Quadratic function of a single variable.
template < typename T = double >
class Quadratic1 :
  public std::unary_function<T, T>
{
private:
  typedef std::unary_function<T, T> base_type;
  typedef T number_type;

public:
  typedef typename base_type::argument_type argument_type;
  typedef typename base_type::result_type result_type;

  result_type
  operator()(const argument_type x) const
  {
    return 0.5 * x * x;
  }
};

// Derivative of the above functional.
template < typename T = double >
class DQuadratic1 :
  public std::unary_function<T, T>
{
private:
  typedef std::unary_function<T, T> base_type;
  typedef T number_type;

public:
  typedef typename base_type::argument_type argument_type;
  typedef typename base_type::result_type result_type;

  result_type
  operator()(const argument_type x) const
  {
    return x;
  }
};

// Quadratic function in R^N.  f : R^N -> R.
template < int N, typename T = double >
class Quadratic :
  public std::unary_function< std::array<T, N>, T >
{
private:
  typedef std::unary_function< std::array<T, N>, T >
  base_type;
  typedef T number_type;

public:
  typedef typename base_type::argument_type argument_type;
  typedef typename base_type::result_type result_type;

  result_type
  operator()(const argument_type& x) const
  {
    result_type result = 0;
    for (int n = 0; n != N; ++n) {
      result += 0.5 * x[n] * x[n];
    }
    return result;
  }
};

// Gradient of the above functional.
template < int N, typename T = double >
class GQuadratic :
  public std::unary_function< std::array<T, N>, std::array<T, N> >
{
private:
  typedef std::unary_function< std::array<T, N>, std::array<T, N> >
  base_type;
  typedef T number_type;

public:
  typedef typename base_type::argument_type argument_type;
  typedef typename base_type::result_type result_type;

  result_type
  operator()(const argument_type& x) const
  {
    result_type result;
    for (int n = 0; n != N; ++n) {
      result[n] = x[n];
    }
    return result;
  }
};

// f : R -> R^N.
template < int N, typename T = double >
class R1RN :
  public std::unary_function< T, std::array<T, N> >
{
private:
  typedef std::unary_function< T, std::array<T, N> > base_type;
  typedef T number_type;

public:
  typedef typename base_type::argument_type argument_type;
  typedef typename base_type::result_type result_type;

  result_type
  operator()(const argument_type x) const
  {
    result_type y;
    for (int n = 0; n != N; ++n) {
      y[n] = (n + 1) * 0.5 * x * x;
    }
    return y;
  }
};

// Derivative of the above functional.
template < int N, typename T = double >
class DR1RN :
  public std::unary_function< T, std::array<T, N> >
{
private:
  typedef std::unary_function< T, std::array<T, N> > base_type;
  typedef T number_type;

public:
  typedef typename base_type::argument_type argument_type;
  typedef typename base_type::result_type result_type;

  result_type
  operator()(const argument_type x) const
  {
    result_type y;
    for (int n = 0; n != N; ++n) {
      y[n] = (n + 1) * x;
    }
    return y;
  }
};

// f : R^N -> R^M.
template < int N, int M, typename T = double >
class RNRM :
  public std::unary_function< std::array<T, N>, std::array<T, M> >
{
private:
  typedef std::unary_function< std::array<T, N>, std::array<T, M> >
  base_type;
  typedef T number_type;

public:
  typedef typename base_type::argument_type argument_type;
  typedef typename base_type::result_type result_type;

  result_type
  operator()(const argument_type x) const
  {
    result_type y;
    for (int m = 0; m != M; ++m) {
      y[m] = 0;
      for (int n = 0; n != N; ++n) {
        y[m] += (m + 1) * (n + 1) * 0.5 * x[n] * x[n];
      }
    }
    return y;
  }
};

// Gradient of the above functional.
template < int N, int M, typename T = double >
class GRNRM :
  public std::unary_function < std::array<T, N>,
  std::array<std::array<T, M>, N> >
{
private:
  typedef std::unary_function < std::array<T, N>,
          std::array<std::array<T, M>, N> >
          base_type;
  typedef T number_type;

public:
  typedef typename base_type::argument_type argument_type;
  typedef typename base_type::result_type result_type;

  result_type
  operator()(const argument_type x) const
  {
    result_type y;
    for (int n = 0; n != N; ++n) {
      for (int m = 0; m != M; ++m) {
        y[n][m] = (m + 1) * (n + 1) * x[n];
      }
    }
    return y;
  }
};

int
main()
{
  using namespace numerical;

  const double delta = std::sqrt(std::numeric_limits<double>::epsilon());

  {
    Quadratic1<> f;
    // Numerical derivative.
    DerivativeCenteredDifference< Quadratic1<> > dfdx(f);
    // Exact derivative.
    DQuadratic1<> df;
    assert(dfdx(1.0) - df(1.0) < f(1.0) * delta);
    std::cout << "error in f'(1) = " << dfdx(1.0) - df(1.0) << '\n';

    assert(dfdx(10.0) - df(10.0) < f(10.0) * delta);
    std::cout << "error in f'(10) = " << dfdx(10.0) - df(10.0)
              << '\n';

    assert(dfdx(100.0) - df(100.0) < f(100.0) * delta);
    std::cout << "error in f'(100) = " << dfdx(100.0) - df(100.0)
              << '\n';
  }
  {
    Quadratic<3> f;
    // Numerical gradient.
    GradientCenteredDifference< 3, Quadratic<3> > dfdx(f);
    // Exact gradient.
    GQuadratic<3> gf;
    std::array<double, 3> x;
    for (int n = 0; n != 3; ++n) {
      std::fill(x.begin(), x.end(), std::pow(10.0, n));
      std::cout << "error in f'(" << x << ") = "
                << dfdx(x) - gf(x) << '\n';
    }
  }
  {
    R1RN<3> f;
    // Numerical derivative.
    DerivativeCenteredDifference< R1RN<3> > dfdx(f);
    // Exact derivative.
    DR1RN<3> exact_dfdx;
    std::cout << "error in f'(1) = "
              << dfdx(1.0) - exact_dfdx(1.0) << '\n';
    std::cout << "error in f'(10) = "
              << dfdx(10.0) - exact_dfdx(10.0) << '\n';
    std::cout << "error in f'(100) = "
              << dfdx(100.0) - exact_dfdx(100.0) << '\n';
  }
  {
    RNRM<2, 3> f;
    RNRM<2, 3>::argument_type x;
    // Numerical gradient of f.
    GradientCenteredDifference< 2, RNRM<2, 3> > gf(f);
    // Exact gradient of f.
    GRNRM<2, 3> egf;
    std::fill(x.begin(), x.end(), 1);
    std::cout << "error in D f(1,1) = " << gf(x) - egf(x) << '\n';
  }

  return 0;
}
