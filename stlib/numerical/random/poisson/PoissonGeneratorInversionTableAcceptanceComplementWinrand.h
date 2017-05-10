// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorInversionTableAcceptanceComplementWinrand.h
  \brief Poisson deviates using the WinRand implementation of table inversion/acceptance complement.
*/

#if !defined(__numerical_PoissonGeneratorInversionTableAcceptanceComplementWinrand_h__)
#define __numerical_PoissonGeneratorInversionTableAcceptanceComplementWinrand_h__

#include "stlib/numerical/random/normal/Default.h"

#include <algorithm>

#include <cmath>

namespace stlib
{
namespace numerical {

//! Poisson deviates using the WinRand implementation of table inversion/acceptance complement.
/*!
  This functor computes Poisson deviates using the
  <a href="http://www.stat.tugraz.at/stadl/random.html">WinRand</a>
  implementation of table inversion/acceptance complement.

  Modifications:
  - Changed <code>double</code> to <code>Number</code>
  - Use my own uniform random deviate generator and normal deviate generator.


  \image html random/poisson/same/sameInversionTableAcceptanceComplementWinrandSmallArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInversionTableAcceptanceComplementWinrandSmallArgument.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/same/sameInversionTableAcceptanceComplementWinrandLargeArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInversionTableAcceptanceComplementWinrandLargeArgument.pdf "Execution times for the same means." width=0.5\textwidth


  \image html random/poisson/different/differentInversionTableAcceptanceComplementWinrandSmallArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInversionTableAcceptanceComplementWinrandSmallArgument.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/different/differentInversionTableAcceptanceComplementWinrandLargeArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInversionTableAcceptanceComplementWinrandLargeArgument.pdf "Execution times for different means." width=0.5\textwidth


  \image html random/poisson/distribution/distributionInversionTableAcceptanceComplementWinrandSmallArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInversionTableAcceptanceComplementWinrandSmallArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionInversionTableAcceptanceComplementWinrandLargeArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInversionTableAcceptanceComplementWinrandLargeArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         template<class> class Normal = NORMAL_GENERATOR_DEFAULT,
         typename _Result = std::size_t >
class PoissonGeneratorInversionTableAcceptanceComplementWinrand {
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

private:

   //
   // Member data.
   //

   //! The normal generator.
   NormalGenerator* _normalGenerator;

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   PoissonGeneratorInversionTableAcceptanceComplementWinrand();

public:

   //! Construct using the normal generator.
   explicit
   PoissonGeneratorInversionTableAcceptanceComplementWinrand
   (NormalGenerator* normalGenerator) :
      _normalGenerator(normalGenerator) {}

   //! Copy constructor.
   PoissonGeneratorInversionTableAcceptanceComplementWinrand
   (const PoissonGeneratorInversionTableAcceptanceComplementWinrand& other) :
      _normalGenerator(other._normalGenerator) {}

   //! Assignment operator.
   PoissonGeneratorInversionTableAcceptanceComplementWinrand&
   operator=
   (const PoissonGeneratorInversionTableAcceptanceComplementWinrand& other) {
      if (this != &other) {
         _normalGenerator = other._normalGenerator;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorInversionTableAcceptanceComplementWinrand() {}

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _normalGenerator->seed(seedValue);
   }

   //! Return a Poisson deviate with the specifed mean.
   result_type
   operator()(argument_type mean);
};


} // namespace numerical
}

#define __numerical_random_PoissonGeneratorInversionTableAcceptanceComplementWinrand_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorInversionTableAcceptanceComplementWinrand.ipp"
#undef __numerical_random_PoissonGeneratorInversionTableAcceptanceComplementWinrand_ipp__

#endif
