// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorInvAcNormSure.h
  \brief Poisson deviates using inversion, acceptance-complement, normal approximation, and sure approximation.
*/

#if !defined(__numerical_PoissonGeneratorInvAcNormSure_h__)
#define __numerical_PoissonGeneratorInvAcNormSure_h__

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionChopDown.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorAcceptanceComplementWinrand.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorNormal.h"

#include "stlib/numerical/round/round.h"

namespace stlib
{
namespace numerical {

//! Poisson deviates using inversion, acceptance-complement, normal approximation, and sure approximation.
/*!
  This functor returns an integer value that is a random deviate drawn from a
  Poisson distribution with specified mean.  For small means the algorithm
  uses the inversion (chop-down) method (see PoissonGeneratorInversionChopDown);
  for medium means it uses the acceptance-complement method
  (see PoissonGeneratorAcceptanceComplementWinrand); for large means it uses
  normal approximation (see PoissonGeneratorNormal); for very large means it
  rounds to the nearest integer.

  CONTINUE HERE

  \image html random/poisson/same/sameInvAcNormSureSmallArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInvAcNormSureSmallArgument.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/same/sameInvAcNormSureLargeArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInvAcNormSureLargeArgument.pdf "Execution times for the same means." width=0.5\textwidth


  \image html random/poisson/different/differentInvAcNormSureSmallArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInvAcNormSureSmallArgument.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/different/differentInvAcNormSureLargeArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInvAcNormSureLargeArgument.pdf "Execution times for different means." width=0.5\textwidth


  \image html random/poisson/distribution/distributionInvAcNormSureSmallArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInvAcNormSureSmallArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionInvAcNormSureLargeArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInvAcNormSureLargeArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         template<class> class Normal = NORMAL_GENERATOR_DEFAULT,
         typename _Result = double >
class PoissonGeneratorInvAcNormSure {
public:

   //! The number type.
   typedef double Number;
   //! The argument type.
   typedef Number argument_type;
   //! The result type.
   typedef _Result result_type;
   //! The discrete uniform generator.
   typedef _Uniform DiscreteUniformGenerator;
   //! The normal generator.
   typedef Normal<DiscreteUniformGenerator> NormalGenerator;

   //
   // Member data.
   //

private:

   //! The inversion method.
   PoissonGeneratorInversionChopDown<DiscreteUniformGenerator> _inversion;
   //! The acceptance-complement method.
   PoissonGeneratorAcceptanceComplementWinrand
   <DiscreteUniformGenerator, Normal> _acceptanceComplementWinrand;
   //! The normal approximation method.
   PoissonGeneratorNormal<DiscreteUniformGenerator, Normal, result_type>
   _normal;
   //! Normal deviates for means greater than this.
   Number _normalThreshhold;
   //! Sure deviates for means greater than this.
   Number _sureThreshhold;

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   PoissonGeneratorInvAcNormSure();

public:

   //! Construct using the normal generator and the threshhold.
   explicit
   PoissonGeneratorInvAcNormSure(NormalGenerator* normalGenerator,
                                 Number normalThreshhold =
                                    std::numeric_limits<Number>::max(),
                                 Number sureThreshhold =
                                    std::numeric_limits<Number>::max());

   //! Copy constructor.
   PoissonGeneratorInvAcNormSure(const PoissonGeneratorInvAcNormSure& other) :
      _inversion(other._inversion),
      _acceptanceComplementWinrand(other._acceptanceComplementWinrand),
      _normal(other._normal),
      _normalThreshhold(other._normalThreshhold) {}

   //! Assignment operator.
   PoissonGeneratorInvAcNormSure&
   operator=(const PoissonGeneratorInvAcNormSure& other) {
      if (this != &other) {
         _inversion = other._inversion;
         _acceptanceComplementWinrand = other._acceptanceComplementWinrand;
         _normal = other._normal;
         _normalThreshhold = other._normalThreshhold;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorInvAcNormSure() {}

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _acceptanceComplementWinrand.seed(seedValue);
   }

   //! Return a Poisson deviate with the specifed mean.
   result_type
   operator()(argument_type mean);

   //! Set the normal threshhold.
   void
   setNormalThreshhold(const Number t) {
      _normalThreshhold = t;
   }

   //! Set the sure threshhold.
   void
   setSureThreshhold(const Number t) {
      _sureThreshhold = t;
   }

private:

   // Use acceptance-complement for means larger than this.
   Number
   getAcThreshhold() const {
#ifdef NUMERICAL_POISSON_HERMITE_APPROXIMATION
      return 13;
#else
      return 6.5;
#endif
   }
};

} // namespace numerical
}

#define __numerical_random_PoissonGeneratorInvAcNormSure_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorInvAcNormSure.ipp"
#undef __numerical_random_PoissonGeneratorInvAcNormSure_ipp__

#endif
