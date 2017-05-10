// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorIfmAcNorm.h
  \brief Poisson deviates using inversion, acceptance-complement, and normal approximation.
*/

#if !defined(__numerical_PoissonGeneratorIfmAcNorm_h__)
#define __numerical_PoissonGeneratorIfmAcNorm_h__

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionFromModeBuildUp.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorAcceptanceComplementWinrand.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorNormal.h"

namespace stlib
{
namespace numerical {

//! Poisson deviates using inversion from the mode, acceptance-complement, and normal approximation.
/*!
  This functor returns an integer value that is a random deviate drawn from a
  Poisson distribution with specified mean.  For small means the algorithm
  uses the inversion from the mode (build-up) method
  (see PoissonGeneratorInversionFromModeBuildUp);
  for medium means it uses the acceptance-complement method
  (see PoissonGeneratorAcceptanceComplementWinrand); for large means it uses
  normal approximation (see PoissonGeneratorNormal).

  \image html random/poisson/same/sameIfmAcNormSmallArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameIfmAcNormSmallArgument.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/same/sameIfmAcNormLargeArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameIfmAcNormLargeArgument.pdf "Execution times for the same means." width=0.5\textwidth


  \image html random/poisson/different/differentIfmAcNormSmallArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentIfmAcNormSmallArgument.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/different/differentIfmAcNormLargeArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentIfmAcNormLargeArgument.pdf "Execution times for different means." width=0.5\textwidth


  \image html random/poisson/distribution/distributionIfmAcNormSmallArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionIfmAcNormSmallArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionIfmAcNormLargeArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionIfmAcNormLargeArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         template<class> class Normal = NORMAL_GENERATOR_DEFAULT,
         typename _Result = double >
class PoissonGeneratorIfmAcNorm {
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

   //! The inversion from the mode method.
   PoissonGeneratorInversionFromModeBuildUp<DiscreteUniformGenerator>
   _inversionFromTheMode;
   //! The acceptance-complement method.
   PoissonGeneratorAcceptanceComplementWinrand<DiscreteUniformGenerator, Normal>
   _acceptanceComplementWinrand;
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
   PoissonGeneratorIfmAcNorm();

public:

   //! Construct using the normal generator and the threshhold.
   explicit
   PoissonGeneratorIfmAcNorm(NormalGenerator* normalGenerator,
                             Number normalThreshhold =
                                std::numeric_limits<Number>::max());

   //! Copy constructor.
   PoissonGeneratorIfmAcNorm(const PoissonGeneratorIfmAcNorm& other) :
      _inversionFromTheMode(other._inversionFromTheMode),
      _acceptanceComplementWinrand(other._acceptanceComplementWinrand),
      _normal(other._normal),
      _normalThreshhold(other._normalThreshhold) {}

   //! Assignment operator.
   PoissonGeneratorIfmAcNorm&
   operator=(const PoissonGeneratorIfmAcNorm& other) {
      if (this != &other) {
         _inversionFromTheMode = other._inversionFromTheMode;
         _acceptanceComplementWinrand = other._acceptanceComplementWinrand;
         _normal = other._normal;
         _normalThreshhold = other._normalThreshhold;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorIfmAcNorm() {}

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _acceptanceComplementWinrand.seed(seedValue);
   }

   //! Return a Poisson deviate with the specifed mean.
   result_type
   operator()(argument_type mean);
};


} // namespace numerical
}

#define __numerical_random_PoissonGeneratorIfmAcNorm_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorIfmAcNorm.ipp"
#undef __numerical_random_PoissonGeneratorIfmAcNorm_ipp__

#endif
