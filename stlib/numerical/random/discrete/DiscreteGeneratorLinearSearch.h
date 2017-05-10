// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorLinearSearch.h
  \brief discrete deviate.  Linear search.
*/

#if !defined(__numerical_DiscreteGeneratorLinearSearch_h__)
#define __numerical_DiscreteGeneratorLinearSearch_h__

#include "stlib/numerical/random/discrete/DgPmfAndSum.h"
#include "stlib/numerical/random/discrete/linearSearch.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <boost/config.hpp>

#include <numeric>

namespace stlib
{
namespace numerical {

//! Discrete deviate.  Linear search.
/*!
  \param Generator is the discrete, uniform generator.

  This class determines the deviate with a linear search on the probabilities.
*/
template < class _Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorLinearSearch :
   public DgPmfAndSum<true> {
   //
   // Private types.
   //
private:

   //! The interface for the probability mass function and its sum.
   typedef DgPmfAndSum<true> Base;

   //
   // Public constants.
   //
public:

   //! The sum of the PMF is automatically updated.
   BOOST_STATIC_CONSTEXPR bool AutomaticUpdate = true;

   //
   // Public types.
   //
public:

   //! The discrete uniform generator.
   typedef _Generator DiscreteUniformGenerator;
   //! The continuous uniform generator.
   typedef ContinuousUniformGeneratorClosed<DiscreteUniformGenerator>
   ContinuousUniformGenerator;
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

   //! The continuous uniform generator.
   ContinuousUniformGenerator _continuousUniformGenerator;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGeneratorLinearSearch();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorLinearSearch(DiscreteUniformGenerator* generator) :
      // The PMF array is empty.
      Base(),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator) {}

   //! Construct from the uniform generator and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorLinearSearch(DiscreteUniformGenerator* generator,
                                 ForwardIterator begin,
                                 ForwardIterator end) :
      Base(),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator) {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorLinearSearch(const DiscreteGeneratorLinearSearch& other) :
      Base(other),
      _continuousUniformGenerator(other._continuousUniformGenerator) {}

   //! Assignment operator.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorLinearSearch&
   operator=(const DiscreteGeneratorLinearSearch& other) {
      if (this != &other) {
         Base::operator=(other);
         _continuousUniformGenerator = other._continuousUniformGenerator;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorLinearSearch() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Random number generation.
   //@{
public:

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _continuousUniformGenerator.seed(seedValue);
   }

   //! Return a discrete deviate.
   result_type
   operator()() {
      // Loop until we get a valid deviate.
      result_type index;
      do {
         index = linearSearchChopDownGuarded
                 (Base::begin(), Base::end(), _continuousUniformGenerator() * sum());
      }
      while (operator[](index) == 0);
      return index;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! Get the probability with the specified index.
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
   //! Set the probability mass function with the specified index.
   using Base::set;

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
