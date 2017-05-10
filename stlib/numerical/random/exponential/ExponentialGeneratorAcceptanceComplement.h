// -*- C++ -*-

/*!
  \file numerical/random/exponential/ExponentialGeneratorAcceptanceComplement.h
  \brief Exponential random deviate with specified mean.
*/

#if !defined(__numerical_ExponentialGeneratorAcceptanceComplement_h__)
#define __numerical_ExponentialGeneratorAcceptanceComplement_h__

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <limits>

#include <cmath>

namespace stlib
{
namespace numerical {

namespace ExponentialGeneratorAcceptanceComplementConstants {
//! The size of the table.
const int TableSize = 256;
//! Mask for the index.
const unsigned IndexMask = 255U;
}

//! Exponential random deviate with specified mean.
/*!
  \param _Generator The uniform random number generator.

  See the \ref numerical_random_exponential "exponential deviate page"
  for general information.

  Herman Rubin and Brad Johnson.
  "Efficient generation of exponential and normal deviates."
  Journal of Statistical Computation and Simulation, Vol. 76, No. 6,
  509-518, 2006.
*/
template < class _Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class ExponentialGeneratorAcceptanceComplement {
public:

   //! The discrete uniform generator.
   typedef _Generator DiscreteUniformGenerator;
   //! The number type.
   typedef double Number;
   //! The argument type.
   typedef void argument_type;
   //! The result type.
   typedef Number result_type;

private:

   //
   // Member data.
   //

   DiscreteUniformGenerator* _discreteUniformGenerator;
   Number _te, _t1,
          _we[ExponentialGeneratorAcceptanceComplementConstants::TableSize],
          _ae[ExponentialGeneratorAcceptanceComplementConstants::TableSize+1];

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   ExponentialGeneratorAcceptanceComplement();

public:

   //! Construct using the uniform generator.
   explicit
   ExponentialGeneratorAcceptanceComplement(DiscreteUniformGenerator* generator) :
      _discreteUniformGenerator(generator) {
      computeTables();
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   ExponentialGeneratorAcceptanceComplement
   (const ExponentialGeneratorAcceptanceComplement& other) :
      _discreteUniformGenerator(other._discreteUniformGenerator) {
      copy(other);
   }

   //! Assignment operator.
   /*!
     \note The discrete,uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   ExponentialGeneratorAcceptanceComplement&
   operator=(const ExponentialGeneratorAcceptanceComplement& other) {
      if (this != &other) {
         _discreteUniformGenerator = other._discreteUniformGenerator;
         copy(other);
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~ExponentialGeneratorAcceptanceComplement() {}

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _discreteUniformGenerator->seed(seedValue);
   }

   //! Return a standard exponential deviate.
   result_type
   operator()();

   //! Return an exponential deviate with specified mean.
   result_type
   operator()(const Number mean) {
      return mean * operator()();
   }

   //! Get a const pointer to the discrete uniform generator.
   const DiscreteUniformGenerator*
   getDiscreteUniformGenerator() const {
      return _discreteUniformGenerator;
   }

   //! Get a pointer to the discrete uniform generator.
   DiscreteUniformGenerator*
   getDiscreteUniformGenerator() {
      return _discreteUniformGenerator;
   }

private:

   void
   copy(const ExponentialGeneratorAcceptanceComplement& other);

   result_type
   computeAlternateGenerator();

   void
   computeTables();
};

} // namespace numerical
}

#define __numerical_random_exponential_ExponentialGeneratorAcceptanceComplement_ipp__
#include "stlib/numerical/random/exponential/ExponentialGeneratorAcceptanceComplement.ipp"
#undef __numerical_random_exponential_ExponentialGeneratorAcceptanceComplement_ipp__

#endif
