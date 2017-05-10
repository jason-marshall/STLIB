// -*- C++ -*-

/*!
  \file numerical/random/exponential/HypoexponentialDistributionNormalApproximation.h
  \brief Normal approximation of the hypoexponential distribution.
*/

#if !defined(__numerical_HypoexponentialDistributionNormalApproximation_h__)
#define __numerical_HypoexponentialDistributionNormalApproximation_h__

// GCC defines the error function as an extension.
#ifndef __GNUC__
#include "stlib/numerical/specialFunctions/ErrorFunction.h"
#endif

#include <limits>
#include <vector>

#include <cmath>
#include <cassert>

namespace stlib
{
namespace numerical {

//! Normal approximation of the hypoexponential distribution.
/*!
  See the
  \ref numerical_random_hypoexponential "hypoexponential distribution page"
  for general information.
*/
class HypoexponentialDistributionNormalApproximation {
private:

   //
   // Member data.
   //

   //! The allowed error.
   double _allowedError;
   //! True if the normal approximation is accurate enough.
   bool _isValid;
   //! The mean is the sum of the exponential means.
   double _mean;
   //! The variance is the sum of the exponential variances.
   double _variance;
   //! The square root of the variance.
   double _sigma;
   //! The sum of the third moments.
   /*! This quantity is only maintained until the approximation is determined
     to be valid. */
   double _sumOfThirdMoments;

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   HypoexponentialDistributionNormalApproximation();

public:

   //! Construct from the allowed error.
   HypoexponentialDistributionNormalApproximation(const double allowedError) :
      _allowedError(allowedError),
      _isValid(false),
      _mean(0),
      _variance(0),
      _sigma(0),
      _sumOfThirdMoments(0) {
      assert(allowedError > 0);
   }

   // The default copy constructor and assignment operator are fine.

   //! Destructor.
   ~HypoexponentialDistributionNormalApproximation() {}

   //! Clear the set of parameters.
   void
   clear() {
      _isValid = false;
      _mean = 0;
      _variance = 0;
      _sigma = 0;
      _sumOfThirdMoments = 0;
   }

   //! Return true if the normal approximation is within the allowed error tolerance.
   bool
   isValid() const {
      return _isValid;
   }

   //! Return the complementary CDF.
   double
   ccdf(const double t) const {
      return 0.5 * erfc((t - _mean) / (std::sqrt(2.) * _sigma));
   }

   //! Return true if the complementary CDF may be greater than the machine epsilon.
   /*!
     Below are the definitions of the complementary CDF and the complementary
     error function.
     \f[
     \mathrm{ccdf}(t) =
     \frac{1}{2} \mathrm{erfc}\left(\frac{t - \mu}{\sqrt{2} \sigma}\right)
     \f]
     \f[
     \mathrm{erfc}(x) = \frac{2}{\sqrt{\pi}} \int_x^\infty \mathrm{e}^{-\xi^2}
     \mathrm{d}\xi
     \f]
     Let \f$\epsilon = 2^{-52}\f$ be the machine epsilon.
     The numerical solution of \f$\mathrm{ccdf}(x) = \epsilon\f$
     is \f$x \approx 5.746\f$. We determine if the complementary CDF is
     effectively nonzero with the condition
     \f[
     \frac{t - \mu}{\sqrt{2} \sigma} < x
     \f]
     which we simplify below.
     \f[
     t < \mu + \sqrt{2} \sigma x
     \f]
   */
   bool
   isCcdfNonzero(const double t) const {
      const double sqrt2x = 5.74587239219118 * std::sqrt(2.);
      return t < _mean + sqrt2x * _sigma;
   }

   //! Set the mean to infinity.
   void
   setMeanToInfinity() {
      // Set the variables so that erfc(t) == 1.
      _mean = std::numeric_limits<double>::max();
      _variance = 1;
      _sigma = 1;
      _isValid = true;
   }

   //! Insert a parameter. For the sake of efficiency accept the inverse of its value.
   void
   insertInverse(const double m) {
      // m = 1 / lambda
      // For the exponential distribution the first, second, and third moments
      // are 1/lambda, 1/lambda^2, and 2/lambda^3, respectively.
      _mean += m;
      _variance += m * m;
      _sigma = std::sqrt(_variance);
      // If the approximation has not yet been determined to be valid.
      if (! _isValid) {
         _sumOfThirdMoments += 2 * m * m * m;
         // Check to see if the approximation has become valid.
         // The constant 0.65 is from Zahl, 1966.
         _isValid = 0.65 * _sumOfThirdMoments / (_sigma * _sigma * _sigma) <
                    _allowedError;
      }
   }

   //! Insert a parameter into a distribution that has been determined to be valid.
   void
   insertInverseValid(const double m) {
#ifdef STLIB_DEBUG
      assert(_isValid);
#endif
      // m = 1 / lambda
      // For the exponential distribution the first, second, and third moments
      // are 1/lambda, 1/lambda^2, and 2/lambda^3, respectively.
      _mean += m;
      _variance += m * m;
      _sigma = std::sqrt(_variance);
   }
};

} // namespace numerical
}

#endif
