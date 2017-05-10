// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorDirectNr.h
  \brief Uniform random deviates.
*/

#if !defined(__numerical_PoissonGeneratorDirectNr_h__)
#define __numerical_PoissonGeneratorDirectNr_h__

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <cstddef>
#include <cmath>

namespace stlib
{
namespace numerical {

//! Generator for Poisson deviates.
/*!
  \param _Generator The uniform random number generator.
  This generator can be initialized in the constructor or with seed().
  \param _Result The result type.  By default it is std::size_t.

  This functor is adapted from the direct method of computing Poisson
  deviates presented in "Numerical Recipes".
  It returns an integer value that is a random deviate drawn from a
  Poisson distribution with specified mean.  This is a practical method
  for small means.

  \image html random/poisson/same/sameDirectNr.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameDirectNr.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/different/differentDirectNr.jpg "Execution times for different means."
  \image latex random/poisson/different/differentDirectNr.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionDirectNr.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionDirectNr.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         typename _Result = std::size_t >
class PoissonGeneratorDirectNr {
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
   Number _g, _oldm;

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   PoissonGeneratorDirectNr();

public:

   //! Construct using the uniform generator.
   explicit
   PoissonGeneratorDirectNr(DiscreteUniformGenerator* generator) :
      _discreteUniformGenerator(generator),
      _g(),
      _oldm(-1.0) {}

   //! Copy constructor.
   PoissonGeneratorDirectNr(const PoissonGeneratorDirectNr& other) :
      _discreteUniformGenerator(other._discreteUniformGenerator),
      _g(other._g),
      _oldm(other._oldm) {}

   //! Assignment operator.
   PoissonGeneratorDirectNr&
   operator=(const PoissonGeneratorDirectNr& other) {
      if (this != &other) {
         _discreteUniformGenerator = other._discreteUniformGenerator;
         _g = other._g;
         _oldm = other._oldm;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorDirectNr() {}

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

#define __numerical_random_PoissonGeneratorDirectNr_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorDirectNr.ipp"
#undef __numerical_random_PoissonGeneratorDirectNr_ipp__

#endif
