// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorInversionChopDownUnrolled.h
  \brief Poisson deviates using inversion (chop-down).
*/

#if !defined(__numerical_PoissonGeneratorInversionChopDownUnrolled_h__)
#define __numerical_PoissonGeneratorInversionChopDownUnrolled_h__

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionMaximumMean.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include "stlib/numerical/interpolation/hermite.h"
#include "stlib/ext/functional.h"

#include <vector>

#include <cmath>
#include <cassert>

namespace stlib
{
namespace numerical {

//! Generator for Poisson deviates using the inversion (chop-down) method.
/*!
  \image html random/poisson/same/sameInversionChopDownUnrolled.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInversionChopDownUnrolled.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/different/differentInversionChopDownUnrolled.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInversionChopDownUnrolled.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionInversionChopDownUnrolled.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInversionChopDownUnrolled.pdf "Execution times for a distribution of means." width=0.5\textwidth

  \note This algorithm evaluates the probability density function.  For
  mean \f$\mu\f$, this is
  \f[
  P(n) = \frac{e^{-\mu} \mu^n}{n!}.
  \f]
  If the mean is large enough, evaluating the exponential will cause underflow.
  Typically this means that what should be a small positive number is
  truncated to zero.  In this case, the algorithm gives incorrect results.
  The maximum allowed mean is
  <pre>- std::log(std::numeric_limits<Number>::min())</pre>
  (I check this with an assertion when debugging is enabled.)
  When using double precision floating-point numbers, do not call
  this function with arguments greater than 708.  For single
  precision numbers, the threshhold is 87.
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         typename _Result = std::size_t >
class PoissonGeneratorInversionChopDownUnrolled {
public:

   //! The number type.
   typedef double Number;
   //! The argument type.
   typedef Number argument_type;
   //! The result type.
   typedef _Result result_type;
   //! The discrete uniform generator.
   typedef _Uniform DiscreteUniformGenerator;

   //
   // Member data.
   //

private:

   //! The discrete uniform generator.
   DiscreteUniformGenerator* _discreteUniformGenerator;
   numerical::Hermite<Number> _expNeg;
   std::vector<Number> _inverse;
#ifdef NUMERICAL_POISSON_CACHE_OLD_MEAN
   Number _oldMean, _oldExponential;
#endif

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   PoissonGeneratorInversionChopDownUnrolled();

   //
   // Member functions.
   //
public:

   //! Construct using the uniform generator and the maximum mean.
   explicit
   PoissonGeneratorInversionChopDownUnrolled(DiscreteUniformGenerator* generator,
         const Number maximumMean) :
      _discreteUniformGenerator(generator),
      // Approximate exp(-x) for x in [0..maximumMean).  Choose a patch length of
      // about 1/100.  Add 0.01 to the maximumMean so there is not an error if
      // this functor is called with maximumMean.
      _expNeg(ext::compose1(std::ptr_fun<double, double>(std::exp),
                            std::negate<double>()),
              ext::compose1(std::negate<double>(),
                            ext::compose1(std::ptr_fun<double, double>(std::exp),
                                        std::negate<double>())),
              0.0, maximumMean + 0.01, int(100 * maximumMean) + 1),
      // CONTINUE: choose an approprate size.
      _inverse(2 * PoissonGeneratorInversionMaximumMean<Number>::Value)
#ifdef NUMERICAL_POISSON_CACHE_OLD_MEAN
      , _oldMean(-1)
      , _oldExponential(-1)
#endif
   {
      _inverse[0] = 0;
      _inverse[1] = 0;
      _inverse[2] = 0;
      _inverse[3] = 0;
      for (std::size_t i = 4; i < _inverse.size(); ++i) {
         _inverse[i] = 1.0 / ((i - 3) * (i - 2) * (i - 1) * i);
      }
   }

   //! Copy constructor.
   PoissonGeneratorInversionChopDownUnrolled
   (const PoissonGeneratorInversionChopDownUnrolled& other) :
      _discreteUniformGenerator(other._discreteUniformGenerator),
      _expNeg(other._expNeg),
      _inverse(other._inverse)
#ifdef NUMERICAL_POISSON_CACHE_OLD_MEAN
      , _oldMean(other._oldMean)
      , _oldExponential(other._oldExponential)
#endif
   {}

   //! Assignment operator.
   PoissonGeneratorInversionChopDownUnrolled&
   operator=(const PoissonGeneratorInversionChopDownUnrolled& other) {
      if (this != &other) {
         _discreteUniformGenerator = other._discreteUniformGenerator;
         _expNeg = other._expNeg;
         _inverse = other._inverse;
#ifdef NUMERICAL_POISSON_CACHE_OLD_MEAN
         _oldMean = other._oldMean;
         _oldExponential = other._oldExponential;
#endif
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorInversionChopDownUnrolled() {}

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _discreteUniformGenerator->seed(seedValue);
   }

   //! Return a Poisson deviate with the specifed mean.
   result_type
   operator()(argument_type mean);
};


} // namespace numerical
}

#define __numerical_random_PoissonGeneratorInversionChopDownUnrolled_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorInversionChopDownUnrolled.ipp"
#undef __numerical_random_PoissonGeneratorInversionChopDownUnrolled_ipp__

#endif
