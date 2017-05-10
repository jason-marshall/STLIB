// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorAcceptanceComplementWinrand.h
  \brief Poisson deviates using the WinRand implementation of acceptance complement.
*/

#if !defined(__numerical_PoissonGeneratorAcceptanceComplementWinrand_h__)
#define __numerical_PoissonGeneratorAcceptanceComplementWinrand_h__

#include "stlib/numerical/random/normal/Default.h"

#include <algorithm>

namespace stlib
{
namespace numerical {

//! Poisson deviates using the WinRand implementation of acceptance complement.
/*!
  \param Generator The uniform random number generator.
  This generator can be initialized in the constructor or with seed().
  \param _Result The result type. By default it is std::size_t.

  This functor computes Poisson deviates using the
  <a href="http://www.stat.tugraz.at/stadl/random.html">WinRand</a>
  implementation of acceptance complement.

  Modifications:
  - Changed <code>double</code> to <code>Number</code>
  - Use my own uniform random deviate generator and normal deviate generator.


  \image html random/poisson/same/sameAcceptanceComplementWinrand.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameAcceptanceComplementWinrand.pdf "Execution times for the same means." width=0.5\textwidth


  \image html random/poisson/different/differentAcceptanceComplementWinrand.jpg "Execution times for different means."
  \image latex random/poisson/different/differentAcceptanceComplementWinrand.pdf "Execution times for different means." width=0.5\textwidth


  \image html random/poisson/distribution/distributionAcceptanceComplementWinrand.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionAcceptanceComplementWinrand.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         template<class> class Normal = NORMAL_GENERATOR_DEFAULT,
         typename _Result = std::size_t >
class PoissonGeneratorAcceptanceComplementWinrand {
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
   PoissonGeneratorAcceptanceComplementWinrand();

public:

   //! Construct using the normal generator.
   explicit
   PoissonGeneratorAcceptanceComplementWinrand(NormalGenerator* normalGenerator) :
      _normalGenerator(normalGenerator) {}

   //! Copy constructor.
   PoissonGeneratorAcceptanceComplementWinrand
   (const PoissonGeneratorAcceptanceComplementWinrand& other) :
      _normalGenerator(other._normalGenerator) {}

   //! Assignment operator.
   PoissonGeneratorAcceptanceComplementWinrand&
   operator=(const PoissonGeneratorAcceptanceComplementWinrand& other) {
      if (this != &other) {
         _normalGenerator = other._normalGenerator;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorAcceptanceComplementWinrand() {}

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

#define __numerical_random_PoissonGeneratorAcceptanceComplementWinrand_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorAcceptanceComplementWinrand.ipp"
#undef __numerical_random_PoissonGeneratorAcceptanceComplementWinrand_ipp__

#endif
