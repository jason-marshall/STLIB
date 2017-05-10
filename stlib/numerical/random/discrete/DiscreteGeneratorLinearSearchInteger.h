// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorLinearSearchInteger.h
  \brief discrete deviate.  Linear search.
*/

#if !defined(__numerical_DiscreteGeneratorLinearSearchInteger_h__)
#define __numerical_DiscreteGeneratorLinearSearchInteger_h__

#include "stlib/numerical/random/discrete/DgPmfInteger.h"

#include "stlib/numerical/random/uniform/Default.h"

#include <numeric>

namespace stlib
{
namespace numerical {

//! discrete deviate.  Linear search.
/*!
  \param _Pmf is the policy class that handles the probability mass function.
  By default it is DgPmfAndSum<> .
  The \c Number type is inherited from this class.
  Because the different policies have different template parameters, this
  is a concrete class, and not a template template.
  \param Generator is the discrete, uniform generator.

  This class determines the deviate with a linear search on the probabilities.
*/
template < class _Pmf = DgPmfInteger<>,
         class _Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorLinearSearchInteger :
   public _Pmf {
   //
   // Private types.
   //
private:

   //! The interface for the probability mass function and its sum.
   typedef _Pmf Base;

   //
   // Public types.
   //
public:

   //! The discrete uniform generator.
   typedef _Generator DiscreteUniformGenerator;
   //! The number type.
   typedef typename Base::Number Number;
   //! The argument type.
   typedef void argument_type;
   //! The result type.
   typedef std::size_t result_type;

   //
   // Member data.
   //
protected:

   //! The discrete uniform generator.
   DiscreteUniformGenerator* _discreteUniformGenerator;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGeneratorLinearSearchInteger();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorLinearSearchInteger
   (DiscreteUniformGenerator* generator) :
      // The PMF array is empty.
      Base(),
      // Store a pointer to the discrete uniform generator.
      _discreteUniformGenerator(generator) {}

   //! Construct from the uniform generator and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorLinearSearchInteger
   (DiscreteUniformGenerator* generator,
    ForwardIterator begin, ForwardIterator end) :
      Base(),
      // Store a pointer to the discrete uniform generator.
      _discreteUniformGenerator(generator) {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorLinearSearchInteger
   (const DiscreteGeneratorLinearSearchInteger& other) :
      Base(other),
      _discreteUniformGenerator(other._discreteUniformGenerator) {}

   //! Assignment operator.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorLinearSearchInteger&
   operator=(const DiscreteGeneratorLinearSearchInteger& other) {
      if (this != &other) {
         Base::operator=(other);
         _discreteUniformGenerator = other._discreteUniformGenerator;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorLinearSearchInteger() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Random number generation.
   //@{
public:

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _discreteUniformGenerator->seed(seedValue);
   }

   //! Return a discrete deviate.
   result_type
   operator()() {
      result_type index;
      do {
         index = Base::operator()((*_discreteUniformGenerator)());
      }
      while (operator[](index) == 0);
      return index;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! Get the probability mass function with the specified index.
   using Base::operator[];
   //! Get the number of possible deviates.
   using Base::size;
   //! Get the sum of the probability mass functions.
   using Base::sum;
   //! Return true if the sum of the PMF is positive.
   using Base::isValid;

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
public:

   using Base::operator==;

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
public:

   //! Initialize the probability mass function.
   using Base::initialize;

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
public:

   //! Print information about the data structure.
   using Base::print;

   //@}
};

} // namespace numerical
}

#endif
