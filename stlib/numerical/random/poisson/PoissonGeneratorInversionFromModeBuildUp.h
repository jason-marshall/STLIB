// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorInversionFromModeBuildUp.h
  \brief Inversion from the mode method of generating Poisson deviates.
*/

#if !defined(__numerical_PoissonGeneratorInversionFromModeBuildUp_h__)
#define __numerical_PoissonGeneratorInversionFromModeBuildUp_h__

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
#include "stlib/numerical/random/poisson/PoissonPdfCdfAtTheMode.h"
#else
#include "stlib/numerical/random/poisson/PoissonPdfCached.h"
#include "stlib/numerical/random/poisson/PoissonCdfAtTheMode.h"
#endif

namespace stlib
{
namespace numerical {

//! Inversion from the mode method of generating Poisson deviates.
/*!
  CONTINUE: The mean absolute deviation is bounded above by the standard
  deviation.

  \image html random/poisson/same/sameInversionFromModeBuildUp.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInversionFromModeBuildUp.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/different/differentInversionFromModeBuildUp.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInversionFromModeBuildUp.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionInversionFromModeBuildUp.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInversionFromModeBuildUp.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         typename _Result = std::size_t >
class PoissonGeneratorInversionFromModeBuildUp {
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
   PoissonPdfCdfAtTheMode<Number> _pdfCdf;
#else
   PoissonPdfCached<Number> _pdf;
   PoissonCdfAtTheMode<Number> _cdfAtTheMode;
#endif

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   PoissonGeneratorInversionFromModeBuildUp();

   //! Copy constructor not implemented.
   PoissonGeneratorInversionFromModeBuildUp
   (const PoissonGeneratorInversionFromModeBuildUp&);

   //! Assignment operator not implemented.
   PoissonGeneratorInversionFromModeBuildUp&
   operator=(const PoissonGeneratorInversionFromModeBuildUp&);

public:

   //! Construct using the uniform generator and the maximum mean.
   explicit
   PoissonGeneratorInversionFromModeBuildUp(DiscreteUniformGenerator* generator,
         const Number maximumMean) :
      _discreteUniformGenerator(generator),
#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
      _pdfCdf(0, static_cast<int>(maximumMean), 100)
#else
      _pdf(static_cast<std::size_t>(maximumMean + 1)),
      _cdfAtTheMode(static_cast<std::size_t>(maximumMean + 1))
#endif
   {}

   //! Destructor.
   ~PoissonGeneratorInversionFromModeBuildUp() {}

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

#define __numerical_random_PoissonGeneratorInversionFromModeBuildUp_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorInversionFromModeBuildUp.ipp"
#undef __numerical_random_PoissonGeneratorInversionFromModeBuildUp_ipp__

#endif
