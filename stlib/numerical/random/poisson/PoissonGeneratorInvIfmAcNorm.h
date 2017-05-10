// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorInvIfmAcNorm.h
  \brief Poisson deviates using inversion, inversion from the mode, acceptance-complement, and normal approximation.
*/

#if !defined(__numerical_PoissonGeneratorInvIfmAcNorm_h__)
#define __numerical_PoissonGeneratorInvIfmAcNorm_h__

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionChopDown.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorInversionFromModeBuildUp.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorAcceptanceComplementWinrand.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorNormal.h"

namespace stlib
{
namespace numerical {

//! Poisson deviates using inversion, inversion from the mode, acceptance-complement, and normal approximation.
/*!
  This functor returns an integer value that is a random deviate drawn from a
  Poisson distribution with specified mean.  For small means the algorithm
  uses the inversion from the mode (build-up) method
  (see PoissonGeneratorInversionFromModeBuildUp);
  for medium means it uses the acceptance-complement method
  (see PoissonGeneratorAcceptanceComplementWinrand); for large means it uses
  the normal approximation (see PoissonGeneratorNormal).

  \image html random/poisson/same/sameInvIfmAcNormSmallArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInvIfmAcNormSmallArgument.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/same/sameInvIfmAcNormLargeArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInvIfmAcNormLargeArgument.pdf "Execution times for the same means." width=0.5\textwidth


  \image html random/poisson/different/differentInvIfmAcNormSmallArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInvIfmAcNormSmallArgument.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/different/differentInvIfmAcNormLargeArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInvIfmAcNormLargeArgument.pdf "Execution times for different means." width=0.5\textwidth


  \image html random/poisson/distribution/distributionInvIfmAcNormSmallArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInvIfmAcNormSmallArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionInvIfmAcNormLargeArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInvIfmAcNormLargeArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         template<class> class Normal = NORMAL_GENERATOR_DEFAULT,
         typename _Result = double >
class PoissonGeneratorInvIfmAcNorm {
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
   //! The inversion from the mode method.
   PoissonGeneratorInversionFromModeBuildUp<DiscreteUniformGenerator>
   _inversionFromTheMode;
   //! The acceptance-complement method.
   PoissonGeneratorAcceptanceComplementWinrand
   <DiscreteUniformGenerator, Normal> _acceptanceComplementWinrand;
   //! The normal approximation method.
   PoissonGeneratorNormal<DiscreteUniformGenerator, Normal, result_type>
   _normal;
   //! The normal deviates for means greater than this.
   Number _normalThreshhold;

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   PoissonGeneratorInvIfmAcNorm();

public:

   //! Construct using the normal generator and the threshhold.
   explicit
   PoissonGeneratorInvIfmAcNorm(NormalGenerator* normalGenerator,
                                Number normalThreshhold =
                                   std::numeric_limits<Number>::max());

   //! Copy constructor.
   PoissonGeneratorInvIfmAcNorm(const PoissonGeneratorInvIfmAcNorm& other) :
      _inversion(other._inversion),
      _inversionFromTheMode(other._inversionFromTheMode),
      _acceptanceComplementWinrand(other._acceptanceComplementWinrand),
      _normal(other._normal),
      _normalThreshhold(other._normalThreshhold) {}

   //! Assignment operator.
   PoissonGeneratorInvIfmAcNorm&
   operator=(const PoissonGeneratorInvIfmAcNorm& other) {
      if (this != &other) {
         _inversion = other._inversion;
         _inversionFromTheMode = other._inversionFromTheMode;
         _acceptanceComplementWinrand = other._acceptanceComplementWinrand;
         _normal = other._normal;
         _normalThreshhold = other._normalThreshhold;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorInvIfmAcNorm() {}

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _inversion.seed(seedValue);
   }

   //! Return a Poisson deviate with the specifed mean.
   result_type
   operator()(argument_type mean);
};

} // namespace numerical
}

#define __numerical_random_PoissonGeneratorInvIfmAcNorm_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorInvIfmAcNorm.ipp"
#undef __numerical_random_PoissonGeneratorInvIfmAcNorm_ipp__

#endif
