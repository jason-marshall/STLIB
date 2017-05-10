// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorLinearSearchDelayedUpdate.h
  \brief Discrete deviate.  Linear search.
*/

#if !defined(__numerical_DiscreteGeneratorLinearSearchDelayedUpdate_h__)
#define __numerical_DiscreteGeneratorLinearSearchDelayedUpdate_h__

#include "stlib/numerical/random/discrete/DgPmf.h"
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
class DiscreteGeneratorLinearSearchDelayedUpdate :
   public DgPmf<true> {
   //
   // Private types.
   //
private:

   //! The interface for the probability mass function and its sum.
   typedef DgPmf<true> Base;

   //
   // Public constants.
   //
public:

   //! The sum of the PMF is not automatically updated.
   BOOST_STATIC_CONSTEXPR bool AutomaticUpdate = false;

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

   //! Probability mass function.  (This is scaled and may not sum to unity.)
   using Base::_pmf;
   //! The sum of the PMF.
   Number _sum;
   //! The continuous uniform generator.
   ContinuousUniformGenerator _continuousUniformGenerator;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGeneratorLinearSearchDelayedUpdate();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorLinearSearchDelayedUpdate
   (DiscreteUniformGenerator* generator) :
      // The PMF array is empty.
      Base(),
      _sum(0),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator) {}

   //! Construct from the uniform generator and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorLinearSearchDelayedUpdate(DiscreteUniformGenerator* generator,
         ForwardIterator begin,
         ForwardIterator end) :
      Base(),
      _sum(0),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator) {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorLinearSearchDelayedUpdate
   (const DiscreteGeneratorLinearSearchDelayedUpdate& other) :
      Base(other),
      _sum(other._sum),
      _continuousUniformGenerator(other._continuousUniformGenerator) {}

   //! Assignment operator.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorLinearSearchDelayedUpdate&
   operator=(const DiscreteGeneratorLinearSearchDelayedUpdate& other) {
      if (this != &other) {
         Base::operator=(other);
         _sum = other._sum;
         _continuousUniformGenerator = other._continuousUniformGenerator;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorLinearSearchDelayedUpdate() {}

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
   Number
   sum() const {
      return _sum;
   }

   //! Return true if the sum of the PMF is positive.
   bool
   isValid() const {
      return _sum > 0;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
public:

   bool
   operator==(const DiscreteGeneratorLinearSearchDelayedUpdate& other) const {
      return Base::operator==(other) && _sum == other._sum;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
public:

   //! Set the probability mass function with the specified index.
   using Base::set;

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator begin, ForwardIterator end) {
      // Initialize the PMF.
      Base::initialize(begin, end);
      // Compute the sum.
      updateSum();
   }

   //! Compute the sum of the PMF.
   void
   updateSum() {
      _sum = std::accumulate(begin(), end(), 0.0);
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
public:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const {
      Base::print(out);
      out << "PMF sum = " << _sum << "\n";
   }

   //@}
};

} // namespace numerical
}

#endif
