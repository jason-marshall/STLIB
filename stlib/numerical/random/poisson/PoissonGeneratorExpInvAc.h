// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorExpInvAc.h
  \brief Poisson deviates using exponential inter-arrival, inversion, and acceptance-complement.
*/

#if !defined(__numerical_PoissonGeneratorExpInvAc_h__)
#define __numerical_PoissonGeneratorExpInvAc_h__

#include "stlib/numerical/random/poisson/PoissonGeneratorExponentialInterArrival.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorInversionChopDown.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorAcceptanceComplementWinrand.h"

namespace stlib
{
namespace numerical {

//! Poisson deviates using exponential inter-arrival, inversion, and acceptance-complement.
/*!
  \param T The number type.  By default it is double.
  \param Generator The uniform random number generator.
  This generator can be initialized in the constructor or with seed().

  This functor returns an integer value that is a random deviate drawn from a
  Poisson distribution with specified mean.  For very small means the algorithm
  uses the exponential inter-arrival method (see
  PoissonGeneratorExponentialInterArrival); for small means it uses the
  chop-down version of inversion (see PoissonGeneratorInversionChopDown);
  for the rest it uses the acceptance-complement method
  (see PoissonGeneratorAcceptanceComplementWinrand).

  \image html random/poisson/same/sameExpInvAcSmallArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameExpInvAcSmallArgument.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/same/sameExpInvAcLargeArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameExpInvAcLargeArgument.pdf "Execution times for the same means." width=0.5\textwidth


  \image html random/poisson/different/differentExpInvAcSmallArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentExpInvAcSmallArgument.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/different/differentExpInvAcLargeArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentExpInvAcLargeArgument.pdf "Execution times for different means." width=0.5\textwidth


  \image html random/poisson/distribution/distributionExpInvAcSmallArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionExpInvAcSmallArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionExpInvAcLargeArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionExpInvAcLargeArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         template<class> class _Exponential =
         EXPONENTIAL_GENERATOR_DEFAULT,
         template<class> class Normal = NORMAL_GENERATOR_DEFAULT,
         typename _Result = std::size_t >
class PoissonGeneratorExpInvAc {
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

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   PoissonGeneratorExpInvAc();

public:

   //! Construct using the exponential generator and the normal generator.
   /*!
     Use the discrete from the exponential generator in the inversion method.
   */
   explicit
   PoissonGeneratorExpInvAc(ExponentialGenerator* exponentialGenerator,
                            NormalGenerator* normalGenerator);

   //! Copy constructor.
   PoissonGeneratorExpInvAc(const PoissonGeneratorExpInvAc& other) :
      _exponentialInterArrival(other._exponentialInterArrival),
      _inversion(other._inversion),
      _acceptanceComplementWinrand(other._acceptanceComplementWinrand) {}

   //! Assignment operator.
   PoissonGeneratorExpInvAc&
   operator=(const PoissonGeneratorExpInvAc& other) {
      if (this != &other) {
         _exponentialInterArrival = other._exponentialInterArrival;
         _inversion = other._inversion;
         _acceptanceComplementWinrand = other._acceptanceComplementWinrand;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorExpInvAc() {}

   //! Seed the uniform random number generators for each of the methods.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _exponentialInterArrival.seed(seedValue);
      // No need to seed the inversion method as it is borrowing the uniform
      // generator from the exponential method.
      _acceptanceComplementWinrand.seed(seedValue);
   }

   //! Return a Poisson deviate with the specifed mean.
   result_type
   operator()(argument_type mean);
};


} // namespace numerical
}

#define __numerical_random_PoissonGeneratorExpInvAc_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorExpInvAc.ipp"
#undef __numerical_random_PoissonGeneratorExpInvAc_ipp__

#endif
