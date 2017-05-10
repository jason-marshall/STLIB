// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorExpAcNorm.h
  \brief Poisson deviates using exponential inter-arrival, acceptance-complement, and normal approximation.
*/

#if !defined(__numerical_PoissonGeneratorExpAcNorm_h__)
#define __numerical_PoissonGeneratorExpAcNorm_h__

#include "stlib/numerical/random/poisson/PoissonGeneratorExponentialInterArrival.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorAcceptanceComplementWinrand.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorNormal.h"

namespace stlib
{
namespace numerical {

//! Poisson deviates using exponential inter-arrival, acceptance-complement, and normal approximation.
/*!
  \param _Uniform The uniform random number generator.
  This generator can be initialized in the constructor or with seed().
  \param _Exponential The exponential generator.
  \param Normal The normal generator.
  \param _Result The result type.  By default it is double.

  This functor returns an integer value that is a random deviate drawn from a
  Poisson distribution with specified mean.  For small means the algorithm
  uses the exponential inter-arrival method (see
  PoissonGeneratorExponentialInterArrival);
  for medium means it uses the acceptance-complement method
  (see PoissonGeneratorAcceptanceComplementWinrand); for large means it uses
  normal approximation (see PoissonGeneratorNormal).

  \image html random/poisson/same/sameExpAcNormSmallArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameExpAcNormSmallArgument.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/same/sameExpAcNormLargeArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameExpAcNormLargeArgument.pdf "Execution times for the same means." width=0.5\textwidth


  \image html random/poisson/different/differentExpAcNormSmallArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentExpAcNormSmallArgument.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/different/differentExpAcNormLargeArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentExpAcNormLargeArgument.pdf "Execution times for different means." width=0.5\textwidth


  \image html random/poisson/distribution/distributionExpAcNormSmallArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionExpAcNormSmallArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionExpAcNormLargeArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionExpAcNormLargeArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         template<class> class _Exponential = EXPONENTIAL_GENERATOR_DEFAULT,
         template<class> class Normal = NORMAL_GENERATOR_DEFAULT,
         typename _Result = double >
class PoissonGeneratorExpAcNorm {
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
   PoissonGeneratorExpAcNorm();

public:

   //! Construct using the exponential generator, the normal generator, and the threshhold.
   explicit
   PoissonGeneratorExpAcNorm(ExponentialGenerator* exponentialGenerator,
                             NormalGenerator* normalGenerator,
                             Number normalThreshhold =
                                std::numeric_limits<Number>::max()) :
      _exponentialInterArrival(exponentialGenerator),
      _acceptanceComplementWinrand(normalGenerator),
      _normal(normalGenerator),
      _normalThreshhold(normalThreshhold) {}

   //! Copy constructor.
   PoissonGeneratorExpAcNorm(const PoissonGeneratorExpAcNorm& other) :
      _exponentialInterArrival(other._exponentialInterArrival),
      _acceptanceComplementWinrand(other._acceptanceComplementWinrand),
      _normal(other._normal),
      _normalThreshhold(other._normalThreshhold) {}

   //! Assignment operator.
   PoissonGeneratorExpAcNorm&
   operator=(const PoissonGeneratorExpAcNorm& other) {
      if (this != &other) {
         _exponentialInterArrival = other._exponentialInterArrival;
         _acceptanceComplementWinrand = other._acceptanceComplementWinrand;
         _normal = other._normal;
         _normalThreshhold = other._normalThreshhold;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorExpAcNorm() {}

   //! Seed the uniform random number generators for each of the methods.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _exponentialInterArrival.seed(seedValue);
      _acceptanceComplementWinrand.seed(seedValue);
      _normal.seed(seedValue);
   }

   //! Return a Poisson deviate with the specifed mean.
   result_type
   operator()(argument_type mean);
};


} // namespace numerical
}

#define __numerical_random_PoissonGeneratorExpAcNorm_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorExpAcNorm.ipp"
#undef __numerical_random_PoissonGeneratorExpAcNorm_ipp__

#endif
