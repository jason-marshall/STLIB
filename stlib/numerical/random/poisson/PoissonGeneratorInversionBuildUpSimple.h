// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorInversionBuildUpSimple.h
  \brief Poisson deviates using inversion (build-up).
*/

#if !defined(__numerical_PoissonGeneratorInversionBuildUpSimple_h__)
#define __numerical_PoissonGeneratorInversionBuildUpSimple_h__

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionMaximumMean.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <cmath>
#include <cassert>
#include <cstddef>

namespace stlib
{
namespace numerical {

//! Generator for Poisson deviates using the inversion (build-up) method.
/*!
  \image html random/poisson/same/sameInversionBuildUpSimple.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInversionBuildUpSimple.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/different/differentInversionBuildUpSimple.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInversionBuildUpSimple.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionInversionBuildUpSimple.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInversionBuildUpSimple.pdf "Execution times for a distribution of means." width=0.5\textwidth

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
class PoissonGeneratorInversionBuildUpSimple {
public:

   //! The number type.
   typedef double Number;
   //! The argument type.
   typedef Number argument_type;
   //! The result type.
   typedef _Result result_type;
   //! The discrete uniform generator.
   typedef _Uniform DiscreteUniformGenerator;

private:

   //
   // Member data.
   //

   //! The discrete uniform generator.
   DiscreteUniformGenerator* _discreteUniformGenerator;

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   PoissonGeneratorInversionBuildUpSimple();

   //
   // Member functions.
   //

public:

   //! Construct using the uniform generator.
   explicit
   PoissonGeneratorInversionBuildUpSimple(DiscreteUniformGenerator* generator) :
      _discreteUniformGenerator(generator) {}

   //! Copy constructor.
   PoissonGeneratorInversionBuildUpSimple
   (const PoissonGeneratorInversionBuildUpSimple& other) :
      _discreteUniformGenerator(other._discreteUniformGenerator) {}

   //! Assignment operator.
   PoissonGeneratorInversionBuildUpSimple&
   operator=(const PoissonGeneratorInversionBuildUpSimple& other) {
      if (this != &other) {
         _discreteUniformGenerator = other._discreteUniformGenerator;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorInversionBuildUpSimple() {}

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

#define __numerical_random_PoissonGeneratorInversionBuildUpSimple_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorInversionBuildUpSimple.ipp"
#undef __numerical_random_PoissonGeneratorInversionBuildUpSimple_ipp__

#endif
