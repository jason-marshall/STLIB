// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorExponentialInterArrivalUnrolled.h
  \brief Poisson deviate using exponential deviate inter-arrival times.
*/

#if !defined(__numerical_PoissonGeneratorExponentialInterArrivalUnrolled_h__)
#define __numerical_PoissonGeneratorExponentialInterArrivalUnrolled_h__

#include "stlib/numerical/random/exponential/Default.h"

#include <cstddef>

namespace stlib
{
namespace numerical {

//! Generator for Poisson deviates.
/*!
  This is a test to see if unrolling the loop in this algorithm could help
  performance.  It doesn't.  For a mean of 0.00001, the PDF for a deviate
  of 2 is \f$e^{-0.00001} 0.00001^2 / 2! \approx 5e-11\f$.  Since
  \f$2^{-32} \approx 2e-10\f$, a deviate of 2 or higher is meaningless
  with 32 bit pseudo-random numbers.  In this algorithm, I unroll the loop
  to a depth of 2.  Still, this is slower than the standard algorithm
  implemented in PoissonGeneratorExponentialInterArrival .

  This is as expected.  Loop unrolling is usually not worthwhile when you
  have branches in the loop.
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         template<class> class _Exponential = EXPONENTIAL_GENERATOR_DEFAULT,
         typename _Result = std::size_t >
class PoissonGeneratorExponentialInterArrivalUnrolled {
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
   PoissonGeneratorExponentialInterArrivalUnrolled();

public:

   //! Construct using the normal generator.
   explicit
   PoissonGeneratorExponentialInterArrivalUnrolled
   (ExponentialGenerator* exponentialGenerator) :
      _exponentialGenerator(exponentialGenerator) {}

   //! Copy constructor.
   PoissonGeneratorExponentialInterArrivalUnrolled
   (const PoissonGeneratorExponentialInterArrivalUnrolled& other) :
      _exponentialGenerator(other._exponentialGenerator) {}

   //! Assignment operator.
   PoissonGeneratorExponentialInterArrivalUnrolled&
   operator=(const PoissonGeneratorExponentialInterArrivalUnrolled& other) {
      if (this != &other) {
         _exponentialGenerator = other._exponentialGenerator;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorExponentialInterArrivalUnrolled() {}

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

#define __numerical_random_PoissonGeneratorExponentialInterArrivalUnrolled_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorExponentialInterArrivalUnrolled.ipp"
#undef __numerical_random_PoissonGeneratorExponentialInterArrivalUnrolled_ipp__

#endif
