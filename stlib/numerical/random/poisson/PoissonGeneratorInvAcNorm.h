// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorInvAcNorm.h
  \brief Poisson deviates using inversion, acceptance-complement, and normal approximation.
*/

#if !defined(__numerical_PoissonGeneratorInvAcNorm_h__)
#define __numerical_PoissonGeneratorInvAcNorm_h__

#include "stlib/numerical/random/poisson/PoissonGeneratorInversionChopDown.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorAcceptanceComplementWinrand.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorNormal.h"

namespace stlib
{
namespace numerical {

//! Poisson deviates using inversion, acceptance-complement, and normal approximation.
/*!
  This functor returns an integer value that is a random deviate drawn from a
  Poisson distribution with specified mean.  For small means the algorithm
  uses the inversion (chop-down) method (see PoissonGeneratorInversionChopDown);
  for medium means it uses the acceptance-complement method
  (see PoissonGeneratorAcceptanceComplementWinrand); for large means it uses
  normal approximation (see PoissonGeneratorNormal).

  \image html random/poisson/same/sameInvAcNormSmallArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInvAcNormSmallArgument.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/same/sameInvAcNormLargeArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInvAcNormLargeArgument.pdf "Execution times for the same means." width=0.5\textwidth


  \image html random/poisson/different/differentInvAcNormSmallArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInvAcNormSmallArgument.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/different/differentInvAcNormLargeArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInvAcNormLargeArgument.pdf "Execution times for different means." width=0.5\textwidth


  \image html random/poisson/distribution/distributionInvAcNormSmallArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInvAcNormSmallArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionInvAcNormLargeArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInvAcNormLargeArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         template<class> class Normal = NORMAL_GENERATOR_DEFAULT,
         typename _Result = double >
class PoissonGeneratorInvAcNorm {
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
   //! The normal deviates for means greater than this.
   Number _normalThreshhold;

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   PoissonGeneratorInvAcNorm();

public:

   //! Construct using the normal generator and the threshhold.
   explicit
   PoissonGeneratorInvAcNorm(NormalGenerator* normalGenerator,
                             Number normalThreshhold =
                                std::numeric_limits<Number>::max());

   //! Copy constructor.
   PoissonGeneratorInvAcNorm(const PoissonGeneratorInvAcNorm& other) :
      _inversion(other._inversion),
      _acceptanceComplementWinrand(other._acceptanceComplementWinrand),
      _normal(other._normal),
      _normalThreshhold(other._normalThreshhold) {}

   //! Assignment operator.
   PoissonGeneratorInvAcNorm&
   operator=(const PoissonGeneratorInvAcNorm& other) {
      if (this != &other) {
         _inversion = other._inversion;
         _acceptanceComplementWinrand = other._acceptanceComplementWinrand;
         _normal = other._normal;
         _normalThreshhold = other._normalThreshhold;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorInvAcNorm() {}

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

#define __numerical_random_PoissonGeneratorInvAcNorm_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorInvAcNorm.ipp"
#undef __numerical_random_PoissonGeneratorInvAcNorm_ipp__

#endif
