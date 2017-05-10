// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorInversionFromModeChopDown.h
  \brief Inversion from the mode method of generating Poisson deviates.
*/

#if !defined(__numerical_PoissonGeneratorInversionFromModeChopDown_h__)
#define __numerical_PoissonGeneratorInversionFromModeChopDown_h__

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
#include "stlib/numerical/random/poisson/PoissonPdfAtTheMode.h"
#else
#include "stlib/numerical/random/poisson/PoissonPdfCached.h"
#endif

namespace stlib
{
namespace numerical {

//! Inversion from the mode method of generating Poisson deviates.
/*!
  CONTINUE: The mean absolute deviation is bounded above by the standard
  deviation.

  \image html random/poisson/same/sameInversionFromModeChopDown.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInversionFromModeChopDown.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/different/differentInversionFromModeChopDown.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInversionFromModeChopDown.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionInversionFromModeChopDown.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInversionFromModeChopDown.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         typename _Result = std::size_t >
class PoissonGeneratorInversionFromModeChopDown {
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

#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
   PoissonPdfAtTheMode<Number> _pdf;
#else
   PoissonPdfCached<Number> _pdf;
#endif
#ifdef NUMERICAL_POISSON_CACHE_OLD_MEAN
   Number _oldMean, _oldPdf;
#endif

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   PoissonGeneratorInversionFromModeChopDown();

   //! Copy constructor not implemented.
   PoissonGeneratorInversionFromModeChopDown
   (const PoissonGeneratorInversionFromModeChopDown&);

   //! Assignment operator not implemented.
   PoissonGeneratorInversionFromModeChopDown&
   operator=(const PoissonGeneratorInversionFromModeChopDown&);

public:

   //! Construct using the uniform generator and the maximum mean.
   explicit
   PoissonGeneratorInversionFromModeChopDown(DiscreteUniformGenerator* generator,
         const Number maximumMean) :
      _discreteUniformGenerator(generator),
#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
      _pdf(0, static_cast<int>(maximumMean), 100)
#else
      _pdf(static_cast<std::size_t>(maximumMean + 1))
#endif
#ifdef NUMERICAL_POISSON_CACHE_OLD_MEAN
      , _oldMean(-1)
      , _oldPdf(-1)
#endif
   {}

   //! Destructor.
   ~PoissonGeneratorInversionFromModeChopDown() {}

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

#define __numerical_random_PoissonGeneratorInversionFromModeChopDown_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorInversionFromModeChopDown.ipp"
#undef __numerical_random_PoissonGeneratorInversionFromModeChopDown_ipp__

#endif
