// -*- C++ -*-

/*!
  \file numerical/random/exponential/ExponentialGeneratorZiggurat.h
  \brief Exponential random deviate with specified mean.
*/

#if !defined(__numerical_ExponentialGeneratorZiggurat_h__)
#define __numerical_ExponentialGeneratorZiggurat_h__

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <cmath>

namespace stlib
{
namespace numerical {

//! Exponential random deviate with specified mean.
/*!
  \param _Generator The uniform random number generator.

  See the \ref numerical_random_exponential "exponential deviate page"
  for general information.

  This class implements the ziggurat method for exponential deviates
  presented in
  \ref numerical_random_marsaglia2000 "The ziggurat method for generating random variables."

  Go to http://www.jstatsoft.org/v05/i08/ to get the paper and code.
*/
template < class _Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class ExponentialGeneratorZiggurat {
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
   unsigned _jz, _iz, _ke[256];
   Number _we[256], _fe[256];

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   ExponentialGeneratorZiggurat();

public:

   //! Construct using the uniform generator.
   explicit
   ExponentialGeneratorZiggurat(DiscreteUniformGenerator* generator) :
      _discreteUniformGenerator(generator) {
      computeTables();
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   ExponentialGeneratorZiggurat(const ExponentialGeneratorZiggurat& other) :
      _discreteUniformGenerator(other._discreteUniformGenerator) {
      copy(other);
   }

   //! Assignment operator.
   /*!
     \note The discrete,uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   ExponentialGeneratorZiggurat&
   operator=(const ExponentialGeneratorZiggurat& other) {
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
   ~ExponentialGeneratorZiggurat() {}

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
   copy(const ExponentialGeneratorZiggurat& other);

   void
   computeTables();

   result_type
   fix();
};

} // namespace numerical
}

#define __numerical_random_exponential_ExponentialGeneratorZiggurat_ipp__
#include "stlib/numerical/random/exponential/ExponentialGeneratorZiggurat.ipp"
#undef __numerical_random_exponential_ExponentialGeneratorZiggurat_ipp__

#endif
