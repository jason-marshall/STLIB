// -*- C++ -*-

#include "stlib/numerical/derivative/ridders.h"
#include "stlib/numerical/equality.h"

#include <iostream>
#include <vector>

#include <cassert>

using namespace stlib;

// Quadratic function of a single variable.
template<typename T>
class Quadratic1 :
  public std::unary_function<T, T>
{
private:
  typedef std::unary_function<T, T> base_type;
  typedef T Number;

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
template<typename T>
class DQuadratic1 :
  public std::unary_function<T, T>
{
private:
  typedef std::unary_function<T, T> base_type;
  typedef T Number;

public:
  typedef typename base_type::argument_type argument_type;
  typedef typename base_type::result_type result_type;

  result_type
  operator()(const argument_type x) const
  {
    return x;
  }
};

// Cosine.
template<typename T>
class Cosine :
  public std::unary_function<T, T>
{
private:
  typedef std::unary_function<T, T> base_type;
  typedef T Number;

public:
  typedef typename base_type::argument_type argument_type;
  typedef typename base_type::result_type result_type;

  result_type
  operator()(const argument_type x) const
  {
    return std::cos(x);
  }
};

// Derivative of cosine.
template<typename T>
class DerivativeCosine :
  public std::unary_function<T, T>
{
private:
  typedef std::unary_function<T, T> base_type;
  typedef T Number;

public:
  typedef typename base_type::argument_type argument_type;
  typedef typename base_type::result_type result_type;

  result_type
  operator()(const argument_type x) const
  {
    return - std::sin(x);
  }
};

// Quadratic function in R^N.  f : R^N -> R.
template<std::size_t N, typename T>
class QuadraticStatic :
  public std::unary_function<std::array<T, N>, T>
{
private:
  typedef std::unary_function<std::array<T, N>, T> Base;
  typedef T Number;

public:
  typedef typename Base::argument_type argument_type;
  typedef typename Base::result_type result_type;

  result_type
  operator()(const argument_type& x) const
  {
    result_type result = 0;
    for (std::size_t n = 0; n != N; ++n) {
      result += 0.5 * x[n] * x[n];
    }
    return result;
  }
};

// Gradient of the above functional.
template<std::size_t N, typename T>
class GQuadraticStatic :
  public std::unary_function< std::array<T, N>, std::array<T, N> >
{
private:
  typedef std::unary_function< std::array<T, N>, std::array<T, N> >
  Base;
  typedef T Number;

public:
  typedef typename Base::argument_type argument_type;
  typedef typename Base::result_type result_type;

  result_type
  operator()(const argument_type& x) const
  {
    result_type result;
    for (std::size_t n = 0; n != N; ++n) {
      result[n] = x[n];
    }
    return result;
  }
};

// Quadratic function in R^N.  f : R^N -> R.
template<typename T>
class QuadraticDynamic :
  public std::unary_function<std::vector<T>, T>
{
private:
  typedef std::unary_function<std::vector<T>, T> Base;
  typedef T Number;

public:
  typedef typename Base::argument_type argument_type;
  typedef typename Base::result_type result_type;

  result_type
  operator()(const argument_type& x) const
  {
    result_type result = 0;
    for (std::size_t n = 0; n != x.size(); ++n) {
      result += 0.5 * x[n] * x[n];
    }
    return result;
  }
};

// Gradient of the above functional.
template<typename T>
class GQuadraticDynamic :
  public std::unary_function<std::vector<T>, std::vector<T> >
{
private:
  typedef std::unary_function<std::vector<T>, std::vector<T> >
  Base;
  typedef T Number;

public:
  typedef typename Base::argument_type argument_type;
  typedef typename Base::result_type result_type;

  result_type
  operator()(const argument_type& x) const
  {
    return x;
  }
};

// Quadratic function in R^N.  sum_i(i * x_i^2)
template<typename T>
class IndexQuadratic :
  public std::unary_function<std::vector<T>, T>
{
private:
  typedef std::unary_function<std::vector<T>, T> Base;
  typedef T Number;

public:
  typedef typename Base::argument_type argument_type;
  typedef typename Base::result_type result_type;

  result_type
  operator()(const argument_type& x) const
  {
    result_type result = 0;
    for (std::size_t n = 0; n != x.size(); ++n) {
      result += 0.5 * n * x[n] * x[n];
    }
    return result;
  }
};

// Gradient of the above functional.
template<typename T>
class GradientIndexQuadratic :
  public std::unary_function<std::vector<T>, std::vector<T> >
{
private:
  typedef std::unary_function<std::vector<T>, std::vector<T> >
  Base;
  typedef T Number;

public:
  typedef typename Base::argument_type argument_type;
  typedef typename Base::result_type result_type;

  result_type
  operator()(const argument_type& x) const
  {
    result_type y = x;
    for (std::size_t n = 0; n != y.size(); ++n) {
      y[n] *= n;
    }
    return y;
  }
};


// The error tolerance for the specified number type.
template<typename _T>
struct _Delta;

template<>
struct _Delta<double> {
  static
  double
  Value()
  {
    return std::sqrt(std::numeric_limits<double>::epsilon());
  }
};

// Note that Ridder's method does not work well for single-precision numbers.
template<>
struct _Delta<float> {
  static
  float
  Value()
  {
    return std::pow(std::numeric_limits<float>::epsilon(), 1.f / 7.f);
  }
};


template<typename _Number>
void
test()
{
  using namespace numerical;

  {
    const _Number Delta = std::sqrt(std::numeric_limits<_Number>::epsilon());
    Quadratic1<_Number> f;
    // Exact derivative.
    DQuadratic1<_Number> df;
    assert(std::abs(numerical::derivativeRidders(f, 0.) - df(0.)) < Delta);
    assert(std::abs(numerical::derivativeRidders(f, 1.) - df(1.)) <
           Delta * f(1.));
    assert(std::abs(numerical::derivativeRidders(f, 10.) - df(10.)) <
           Delta * f(10.));
    assert(std::abs(numerical::derivativeRidders(f, 100.) - df(100.)) <
           Delta * f(100.));
  }
  {
    Cosine<_Number> f;
    DerivativeCosine<_Number> df;
    _Number d = df(1.);
    std::cout << "Scale, Error\n";
    std::array<_Number, 17> scales = {
      {
        1e-8, 1e-7,  1e-6,  1e-5,  1e-4,  1e-3, 1e-2,  1e-1,  1e0,
        1e1,   1e2,   1e3, 1e4,   1e5,   1e6,   1e7,   1e8
      }
    };
    for (std::size_t i = 0; i != scales.size(); ++i) {
      std::cout << scales[i] << ' '
                << std::abs(numerical::derivativeRidders(f, 1., scales[i])
                            - d) << '\n';
    }
  }
  {
    const std::size_t N = 3;
    const _Number Delta = std::sqrt(std::numeric_limits<_Number>::epsilon());
    const _Number Tol = Delta / std::numeric_limits<_Number>::epsilon();
    QuadraticStatic<N, _Number> f;
    // Exact derivative.
    GQuadraticStatic<N, _Number> df;
    std::array<_Number, N> x, g;

    x.fill(0);
    numerical::gradientRidders(f, x, &g);
    assert(numerical::areEqual(g, df(x), Tol));

    std::array<_Number, 3> arguments = {{1e0, 1e2, 1e4}};

    for (std::size_t i = 0; i != arguments.size(); ++i) {
      x.fill(arguments[i]);
      x[1] *= 2.;
      x[2] *= 3.;
      numerical::gradientRidders(f, x, &g);
      assert(numerical::areEqual(g, df(x), Tol));
    }
  }
  {
    const std::size_t N = 10;
    //const _Number Delta = std::sqrt(std::numeric_limits<_Number>::epsilon());
    const _Number Delta = _Delta<_Number>::Value();
    const _Number Tol = Delta / std::numeric_limits<_Number>::epsilon();
    QuadraticDynamic<_Number> f;
    // Exact derivative.
    GQuadraticDynamic<_Number> df;
    std::vector<_Number> x(N), g(N);

    std::fill(x.begin(), x.end(), _Number(0));
    numerical::gradientRidders(f, x, &g);
    assert(numerical::areEqual(g, df(x), Tol));

    std::array<_Number, 3> arguments = {{1e0, 1e2, 1e4}};

    for (std::size_t i = 0; i != arguments.size(); ++i) {
      std::fill(x.begin(), x.end(), arguments[i]);
      for (std::size_t j = 0; j != x.size(); ++j) {
        x[j] *= j;
      }
      numerical::gradientRidders(f, x, &g);
      assert(numerical::areEqual(g, df(x), Tol));
    }
  }
  {
    const std::size_t N = 10;
    //const _Number Delta = std::sqrt(std::numeric_limits<_Number>::epsilon());
    const _Number Delta = _Delta<_Number>::Value();
    const _Number Tol = Delta / std::numeric_limits<_Number>::epsilon();
    IndexQuadratic<_Number> f;
    // Exact derivative.
    GradientIndexQuadratic<_Number> df;
    std::vector<_Number> x(N), g(N);

    std::fill(x.begin(), x.end(), _Number(0));
    numerical::gradientRidders(f, x, &g);
    assert(numerical::areEqual(g, df(x), Tol));

    std::array<_Number, 3> arguments = {{1e0, 1e2, 1e4}};

    for (std::size_t i = 0; i != arguments.size(); ++i) {
      std::fill(x.begin(), x.end(), arguments[i]);
      for (std::size_t j = 0; j != x.size(); ++j) {
        x[j] *= j;
      }
      numerical::gradientRidders(f, x, &g);
      assert(numerical::areEqual(g, df(x), Tol));
    }
  }

}

int
main()
{
  test<float>();
  test<double>();

  return 0;
}
