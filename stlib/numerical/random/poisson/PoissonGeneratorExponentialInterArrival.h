// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorExponentialInterArrival.h
  \brief Poisson deviate using exponential deviate inter-arrival times.
*/

#if !defined(__numerical_PoissonGeneratorExponentialInterArrival_h__)
#define __numerical_PoissonGeneratorExponentialInterArrival_h__

#include "stlib/numerical/random/exponential/Default.h"

#include <cstddef>

namespace stlib
{
namespace numerical {

//! Generator for Poisson deviates.
/*!
  This functor is adapted from the direct method of computing Poisson
  deviates presented in "Numerical Recipes".  The primary change I have
  made is an optimization for very small means (\f$\mathrm{mean} \ll 1\f$).
  This functor returns an integer value that is a random deviate drawn from a
  Poisson distribution with specified mean.  This is a practical method
  for small means.

  \image html random/poisson/same/sameExponentialInterArrival.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameExponentialInterArrival.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/different/differentExponentialInterArrival.jpg "Execution times for different means."
  \image latex random/poisson/different/differentExponentialInterArrival.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionExponentialInterArrival.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionExponentialInterArrival.pdf "Execution times for a distribution of means." width=0.5\textwidth

  One might compute Poisson deviates for small means.
  If the the mean is usually less than one, the
  defining NUMERICAL_POISSON_SMALL_MEAN may improve performance.
  In this case it can usually avoid computing an exponential
  for small means.

  If your application often calls this functor with exactly the same
  mean on successive calls, then defining the macro
  NUMERICAL_POISSON_CACHE_OLD_MEAN will improve the performance.  When it is
  defined, it caches values like the exponential of the mean and only
  recomputes them when the mean is different from the previous call.
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         template<class> class _Exponential = EXPONENTIAL_GENERATOR_DEFAULT,
         typename _Result = std::size_t >
class PoissonGeneratorExponentialInterArrival {
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

private:

   //
   // Member data.
   //

   //! The exponential generator.
   ExponentialGenerator* _exponentialGenerator;

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   PoissonGeneratorExponentialInterArrival();

public:

   //! Construct using the exponential generator.
   explicit
   PoissonGeneratorExponentialInterArrival
   (ExponentialGenerator* exponentialGenerator) :
      _exponentialGenerator(exponentialGenerator) {}

   //! Copy constructor.
   PoissonGeneratorExponentialInterArrival
   (const PoissonGeneratorExponentialInterArrival& other) :
      _exponentialGenerator(other._exponentialGenerator) {}

   //! Assignment operator.
   PoissonGeneratorExponentialInterArrival&
   operator=(const PoissonGeneratorExponentialInterArrival& other) {
      if (this != &other) {
         _exponentialGenerator = other._exponentialGenerator;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorExponentialInterArrival() {}

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _exponentialGenerator->seed(seedValue);
   }

   //! Return a Poisson deviate with the specifed mean.
   result_type
   operator()(argument_type mean);
};


} // namespace numerical
}

#define __numerical_random_PoissonGeneratorExponentialInterArrival_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorExponentialInterArrival.ipp"
#undef __numerical_random_PoissonGeneratorExponentialInterArrival_ipp__

#endif
