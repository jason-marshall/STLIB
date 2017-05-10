// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorInversionRejectionPatchworkWinrand.h
  \brief Poisson deviates using the WinRand implementation of inversion/patchwork rejection.
*/

#if !defined(__numerical_PoissonGeneratorInversionRejectionPatchworkWinrand_h__)
//! Include guard.
#define __numerical_PoissonGeneratorInversionRejectionPatchworkWinrand_h__

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <algorithm>

#include <cmath>

namespace stlib
{
namespace numerical {

//! Poisson deviates using the WinRand implementation of inversion/patchwork rejection.
/*!
  This functor computes Poisson deviates using the
  <a href="http://www.stat.tugraz.at/stadl/random.html">WinRand</a>
  implementation of inversion/patchwork rejection.

  Modifications:
  - Changed <code>double</code> to <code>Number</code>
  - Changed <code>long int</code> to <code>int</code>
  - Use my own uniform random deviate generator.


  \image html random/poisson/same/sameInversionRejectionPatchworkWinrandSmallArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInversionRejectionPatchworkWinrandSmallArgument.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/same/sameInversionRejectionPatchworkWinrandLargeArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameInversionRejectionPatchworkWinrandLargeArgument.pdf "Execution times for the same means." width=0.5\textwidth


  \image html random/poisson/different/differentInversionRejectionPatchworkWinrandSmallArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInversionRejectionPatchworkWinrandSmallArgument.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/different/differentInversionRejectionPatchworkWinrandLargeArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentInversionRejectionPatchworkWinrandLargeArgument.pdf "Execution times for different means." width=0.5\textwidth


  \image html random/poisson/distribution/distributionInversionRejectionPatchworkWinrandSmallArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInversionRejectionPatchworkWinrandSmallArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionInversionRejectionPatchworkWinrandLargeArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionInversionRejectionPatchworkWinrandLargeArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         typename _Result = std::size_t >
class PoissonGeneratorInversionRejectionPatchworkWinrand {
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

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   PoissonGeneratorInversionRejectionPatchworkWinrand();

public:

   //! Construct using the uniform generator.
   explicit
   PoissonGeneratorInversionRejectionPatchworkWinrand
   (DiscreteUniformGenerator* generator) :
      _discreteUniformGenerator(generator) {}

   //! Copy constructor.
   PoissonGeneratorInversionRejectionPatchworkWinrand
   (const PoissonGeneratorInversionRejectionPatchworkWinrand& other) :
      _discreteUniformGenerator(other._discreteUniformGenerator) {}

   //! Assignment operator.
   PoissonGeneratorInversionRejectionPatchworkWinrand&
   operator=(const PoissonGeneratorInversionRejectionPatchworkWinrand& other) {
      if (this != &other) {
         _discreteUniformGenerator = other._discreteUniformGenerator;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorInversionRejectionPatchworkWinrand() {}

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _discreteUniformGenerator->seed(seedValue);
   }

   //! Return a Poisson deviate with the specifed mean.
   result_type
   operator()(argument_type mean);

private:

   Number
   f(int k, Number l_nu, Number c_pm) {
      return std::exp(k * l_nu - flogfak(k) - c_pm);
   }

   Number
   flogfak(int k) { /* log(k!) */
      const Number C0 = 9.18938533204672742e-01;
      const Number C1 = 8.33333333333333333e-02;
      const Number C3 = -2.77777777777777778e-03;
      const Number C5 = 7.93650793650793651e-04;
      const Number C7 = -5.95238095238095238e-04;

      static Number logfak[30] = {
         0.00000000000000000,   0.00000000000000000,   0.69314718055994531,
         1.79175946922805500,   3.17805383034794562,   4.78749174278204599,
         6.57925121201010100,   8.52516136106541430,  10.60460290274525023,
         12.80182748008146961,  15.10441257307551530,  17.50230784587388584,
         19.98721449566188615,  22.55216385312342289,  25.19122118273868150,
         27.89927138384089157,  30.67186010608067280,  33.50507345013688888,
         36.39544520803305358,  39.33988418719949404,  42.33561646075348503,
         45.38013889847690803,  48.47118135183522388,  51.60667556776437357,
         54.78472939811231919,  58.00360522298051994,  61.26170176100200198,
         64.55753862700633106,  67.88974313718153498,  71.25703896716800901
      };

      Number  r, rr;

      if (k >= 30) {
         r  = 1.0 / (Number) k;
         rr = r * r;
         return((k + 0.5)*std::log(k) - k + C0 +
                r*(C1 + rr*(C3 + rr*(C5 + rr*C7))));
      }
      else
         return(logfak[k]);
   }

};


} // namespace numerical
}

//! Include guard.
#define __numerical_random_PoissonGeneratorInversionRejectionPatchworkWinrand_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorInversionRejectionPatchworkWinrand.ipp"
#undef __numerical_random_PoissonGeneratorInversionRejectionPatchworkWinrand_ipp__

#endif
