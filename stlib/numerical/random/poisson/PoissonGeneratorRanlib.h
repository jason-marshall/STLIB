// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonGeneratorRanlib.h
  \brief Uniform random deviates.
*/

#if !defined(__numerical_PoissonGeneratorRanlib_h__)
#define __numerical_PoissonGeneratorRanlib_h__

#include "stlib/numerical/random/normal/Default.h"

#include <algorithm>

#include <cmath>

namespace stlib
{
namespace numerical {

//! Generator for Poisson deviates.
/*!
  This functor is adapted from the
  <a href="http://www.netlib.org/random/">Ranlib</a> library.
  Changes:
  - <code>float</code> -> <code>Number</code>
  - <code>long</code> -> <code>int</code>
  - Adde <code>std::</code> to the standard library functions.


  \image html random/poisson/same/sameRanlibSmallArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameRanlibSmallArgument.pdf "Execution times for the same means." width=0.5\textwidth

  \image html random/poisson/same/sameRanlibLargeArgument.jpg "Execution times for the same means."
  \image latex random/poisson/same/sameRanlibLargeArgument.pdf "Execution times for the same means." width=0.5\textwidth


  \image html random/poisson/different/differentRanlibSmallArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentRanlibSmallArgument.pdf "Execution times for different means." width=0.5\textwidth

  \image html random/poisson/different/differentRanlibLargeArgument.jpg "Execution times for different means."
  \image latex random/poisson/different/differentRanlibLargeArgument.pdf "Execution times for different means." width=0.5\textwidth


  \image html random/poisson/distribution/distributionRanlibSmallArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionRanlibSmallArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth

  \image html random/poisson/distribution/distributionRanlibLargeArgument.jpg "Execution times for a distribution of means."
  \image latex random/poisson/distribution/distributionRanlibLargeArgument.pdf "Execution times for a distribution of means." width=0.5\textwidth
*/
template < class _Uniform = DISCRETE_UNIFORM_GENERATOR_DEFAULT,
         template<class> class Normal = NORMAL_GENERATOR_DEFAULT,
         typename _Result = std::size_t >
class PoissonGeneratorRanlib {
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
   PoissonGeneratorRanlib();

public:

   //! Construct using the normal generator.
   explicit
   PoissonGeneratorRanlib(NormalGenerator* normalGenerator) :
      _normalGenerator(normalGenerator) {}

   //! Copy constructor.
   PoissonGeneratorRanlib(const PoissonGeneratorRanlib& other) :
      _normalGenerator(other._normalGenerator) {}

   //! Assignment operator.
   PoissonGeneratorRanlib&
   operator=(const PoissonGeneratorRanlib& other) {
      if (this != &other) {
         _normalGenerator = other._normalGenerator;
      }
      return *this;
   }

   //! Destructor.
   ~PoissonGeneratorRanlib() {}

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _normalGenerator->seed(seedValue);
   }

   //! Return a Poisson deviate with the specifed mean.
   result_type
   operator()(argument_type mean);

private:

   Number
   sexpo();

   //! Transfers sign of argument sign to argument num.
   Number
   fsign(const Number num, const Number sign) const {
      if ((sign > 0.0 && num < 0.0) || (sign<0.0 && num>0.0)) {
         return -num;
      }
      else {
         return num;
      }
   }

};


} // namespace numerical
}

#define __numerical_random_PoissonGeneratorRanlib_ipp__
#include "stlib/numerical/random/poisson/PoissonGeneratorRanlib.ipp"
#undef __numerical_random_PoissonGeneratorRanlib_ipp__

#endif
