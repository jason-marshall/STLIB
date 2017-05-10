// -*- C++ -*-

#if !defined(__numerical_specialFunctions_ErrorFunction_ipp__)
#error This file is an implementation detail of ErrorFunction.
#endif

namespace stlib
{
namespace numerical
{

// These are adapted from the implementation in "Numerical Recipes."

inline
double
erfcChebyshev(const double z)
{
  BOOST_STATIC_CONSTEXPR double coefficients[] = {
    -1.3026537197817094, 6.4196979235649026e-1,
    1.9476473204185836e-2, -9.561514786808631e-3,
    -9.46595344482036e-4, 3.66839497852761e-4, 4.2523324806907e-5,
    -2.0278578112534e-5, -1.624290004647e-6, 1.303655835580e-6,
    1.5626441722e-8, -8.5238095915e-8, 6.529054439e-9,
    5.059343495e-9, -9.91364156e-10, -2.27365122e-10,
    9.6467911e-11, 2.394038e-12, -6.886027e-12, 8.94487e-13,
    3.13092e-13, -1.12708e-13, 3.81e-16, 7.106e-15, -1.523e-15,
    -9.4e-17, 1.21e-16, -2.8e-17
  };
  const std::size_t size = sizeof(coefficients) / sizeof(double);

  assert(z >= 0);
  const double t = 2. / (2. + z);
  const double ty = 4. * t - 2.;
  double d = 0, dd = 0, tmp;
  for (std::size_t j = size - 1; j != 0; --j) {
    tmp = d;
    d = ty * d - dd + coefficients[j];
    dd = tmp;
  }
  return t * std::exp(-z * z + 0.5 * (coefficients[0] + ty * d) - dd);
}

inline
double
erf(const double x)
{
  if (x >= 0) {
    return 1. - erfcChebyshev(x);
  }
  else {
    return erfcChebyshev(-x) - 1.;
  }
}


inline
double
erfc(const double x)
{
  if (x >= 0.) {
    return erfcChebyshev(x);
  }
  else {
    return 2. - erfcChebyshev(-x);
  }
}

} // namespace numerical
}
