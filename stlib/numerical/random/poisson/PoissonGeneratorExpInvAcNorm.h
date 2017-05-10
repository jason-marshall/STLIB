// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorExpInvAcNorm.h
  \brief Poisson deviates using exponential inter-arrival, inversion, acceptance-complement, and normal approximation.
*/

#if !defined(__numerical_PoissonGeneratorExpInvAcNorm_h__)
#define __numerical_PoissonGeneratorExpInvAcNorm_h__

#include "stlib/numerical/random/poisson/PoissonGeneratorExponentialInterArrival.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorInversionChopDown.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorAcceptanceComplementWinrand.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorNormal.h"

namespace stlib
{
namespace numerical {

//! Poisson deviates using exponential inter-arrival, inversion, acceptance-complement, and normal approximation.
/*!
  This functor returns an integer value that is a random deviate drawn from a
  Poisson distribution with specified mean.  For very small means the algorithm
  uses the exponential inter-arrival method (see
  PoissonGeneratorExponentialInterArrival); for small means it uses the
  chop-down version of inversion (see PoissonGeneratorInversionChopDown);
  for medium means it uses the acceptance-complement method
  (see PoissonGeneratorAcceptanceComplementWinrand); for large means it uses
  normal approximation (see PoissonGeneratorNormal).

  \image html random/poisson/same/sameExpInvAcNormSmallArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameExpInvAcNormSmallArgument.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/same/sameExpInvAcNormLargeArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameExpInvAcNormLargeArgument.pdf "Execution times for the same means." width=0.5\textwidth


  \image html random/poisson/different/differentExpInvAcNormSmallArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentExpInvAcNormSmallArgument.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/different/differentExpInvAcNormLargeArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentExpInvAcNormLargeArgument.pdf "Execution times for different means." width=0.5\textwidth


  \image html random/poisson/distribution/distributionExpInvAcNormSmallArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionExpInvAcNormSmallArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionExpInvAcNormLargeArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionExpInvAcNormLargeArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         template<class> class _Exponential = EXPONENTIAL_GENERATOR_DEFAULT,
         template<class> class Normal = NORMAL_GENERATOR_DEFAULT,
         typename _Result = double >
class PoissonGeneratorExpInvAcNorm {
public:

   //! The number type.
   typedef double Number;
   //! The argument type.
   typedef Number argument_type;
   //! The result type.
   typedef _Result result_type;
   //! The discrete uniform generator.
   typedef _Uniform DiscreteUniformGenerator;
   //! The exponential generator.
   typedef _Exponential<DiscreteUniformGenerator> ExponentialGenerator;
   //! The normal generator.
   typedef Normal<DiscreteUniformGenerator> NormalGenerator;

   //
   // Member data.
   //

private:

   //! The exponential inter-arrival method.
   PoissonGeneratorExponentialInterArrival
   <DiscreteUniformGenerator, _Exponential> _exponentialInterArrival;
   //! The inversion method.
   PoissonGeneratorInversionChopDown<DiscreteUniformGenerator> _inversion;
   //! The acceptance-complement method.
   PoissonGeneratorAcceptanceComplementWinrand
   <DiscreteUniformGenerator, Normal> _acceptanceComplementWinrand;
   //! The normal approximation method.
   PoissonGeneratorNormal<DiscreteUniformGenerator, Normal> _normal;
   //! The normal deviates for means greater than this.
   Number _normalThreshhold;

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   PoissonGeneratorExpInvAcNorm();

public:

   //! Construct using the exponential generator, the normal generator, and the threshhold.
   /*!
     Use the discrete from the exponential generator in the inversion method.
   */
   explicit
   PoissonGeneratorExpInvAcNorm(ExponentialGenerator* exponentialGenerator,
                                NormalGenerator* normalGenerator,
                                Number normalThreshhold =
                                   std::numeric_limits<Number>::max());

   //! Copy constructor.
   PoissonGeneratorExpInvAcNorm(const PoissonGeneratorExpInvAcNorm& other) :
      _exponentialInterArrival(other._exponentialInterArrival),
      _inversion(other._inversion),
      _acceptanceComplementWinrand(other._acceptanceComplementWinrand),
      _normal(other._normal),
      _normalThreshhold(other._normalThreshhold) {}

   //! Assignment operator.
   PoissonGeneratorExpInvAcNorm&
   operator=(const PoissonGeneratorExpInvAcNorm& other) {
      if (this != &other) {
         _exponentialInterArrival = other._exponentialInterArrival;
         _inversion = other._inversion;
         _acceptanceComplementWinrand = other._acceptanceComplementWinrand;
         _normal = other._normal;
         _normalThreshhold = other._normalThreshhold;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorExpInvAcNorm() {}

   //! Seed the uniform random number generators for each of the methods.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _exponentialInterArrival.seed(seedValue);
      // No need to seed the inversion method as it is borrowing the uniform
      // generator from the exponential method.
      _acceptanceComplementWinrand.seed(seedValue);
      _normal.seed(seedValue);
   }

   //! Return a Poisson deviate with the specifed mean.
   result_type
   operator()(argument_type mean);
};


} // namespace numerical
}

#define __numerical_random_PoissonGeneratorExpInvAcNorm_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorExpInvAcNorm.ipp"
#undef __numerical_random_PoissonGeneratorExpInvAcNorm_ipp__

#endif
