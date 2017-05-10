// -*- C++ -*-

/*!
  \file numerical/random/normal/NormalGeneratorZigguratVoss.h
  \brief Normal random deviate with zero mean and unit variance.
*/

// Copyright from Jochen Voss.
/* gauss.c - gaussian random numbers, using the Ziggurat method
 *
 * Copyright (C) 2005  Jochen Voss.
 *
 * For details see the following article.
 *
 *     George Marsaglia, Wai Wan Tsang
 *     The Ziggurat Method for Generating Random Variables
 *     Journal of Statistical Software, vol. 5 (2000), no. 8
 *     http://www.jstatsoft.org/v05/i08/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#if !defined(__numerical_NormalGeneratorZigguratVoss_h__)
#define __numerical_NormalGeneratorZigguratVoss_h__

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <cmath>

namespace stlib
{
namespace numerical {

//! Normal random deviate with zero mean and unit variance.
/*!
  \param _Generator The discrete uniform generator.

  This generator can be initialized in the constructor or with seed().
*/
template < class _Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class NormalGeneratorZigguratVoss {
public:

   //! The discrete uniform generator.
   typedef _Generator DiscreteUniformGenerator;
   //! The number type.
   typedef double Number;
   //! The argument type.
   typedef void argument_type;
   //! The result type.
   typedef Number result_type;

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
   NormalGeneratorZigguratVoss();

public:

   //! Construct using the uniform generator.
   explicit
   NormalGeneratorZigguratVoss(DiscreteUniformGenerator* generator) :
      _discreteUniformGenerator(generator) {}

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   NormalGeneratorZigguratVoss(const NormalGeneratorZigguratVoss& other) :
      _discreteUniformGenerator(other._discreteUniformGenerator) {}

   //! Assignment operator.
   /*!
     \note The discrete,uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   NormalGeneratorZigguratVoss&
   operator=(const NormalGeneratorZigguratVoss& other) {
      if (this != &other) {
         _discreteUniformGenerator = other._discreteUniformGenerator;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~NormalGeneratorZigguratVoss() {}

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _discreteUniformGenerator->seed(seedValue);
   }

   //! Return a standard normal deviate.
   result_type
   operator()();

   //! Return a normal deviate with specified mean and variance.
   result_type
   operator()(const Number mean, const Number variance) {
      return std::sqrt(variance) * operator()() + mean;
   }

   //! Get the discrete uniform generator.
   DiscreteUniformGenerator*
   getDiscreteUniformGenerator() {
      return _discreteUniformGenerator;
   }
};


} // namespace numerical
}

#define __numerical_random_NormalGeneratorZigguratVoss_ipp__
#include "stlib/numerical/random/normal/NormalGeneratorZigguratVoss.ipp"
#undef __numerical_random_NormalGeneratorZigguratVoss_ipp__

#endif
