// -*- C++ -*-

/*!
  \file numerical/random/poisson.h
  \brief Constants for the inversion method.
*/

#if !defined(__numerical_random_poisson_PoissonGeneratorInversionMaximumMean_h__)
#define __numerical_random_poisson_PoissonGeneratorInversionMaximumMean_h__

namespace stlib
{
namespace numerical {

//! Maximum allowed mean for using the PoissonGeneratorInversion class.
/*!
  If the mean is too large, we will get underflow in computing the
  probability density function.
*/
template<typename T>
class PoissonGeneratorInversionMaximumMean {
public:
   //! Invalid value for an unknown number type.
   enum {Value = -1};
};

template<>
class PoissonGeneratorInversionMaximumMean<double> {
public:
   //! - std::log(std::numeric_limits<double>::min()) == 708.396
   enum {Value = 708};
};

template<>
class PoissonGeneratorInversionMaximumMean<float> {
public:
   //! - std::log(std::numeric_limits<float>::min()) == 87.3365
   /*!
     Here I assume that the arithmetic is actually done in single precision.
     However, floats are typically converted to double's.
   */
   enum {Value = 87};
};

} // namespace numerical
}

#endif
