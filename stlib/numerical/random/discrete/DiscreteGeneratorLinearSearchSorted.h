// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorLinearSearchSorted.h
  \brief discrete deviate.  Linear search.
*/

#if !defined(__numerical_DiscreteGeneratorLinearSearchSorted_h__)
#define __numerical_DiscreteGeneratorLinearSearchSorted_h__

#include "stlib/numerical/random/discrete/DgPmfAndSumOrderedPairPointer.h"
#include "stlib/numerical/random/discrete/linearSearch.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include "stlib/ads/counter/CounterWithReset.h"

#include <boost/config.hpp>

#include <algorithm>
#include <numeric>

namespace stlib
{
namespace numerical {

//! Discrete deviate.  Linear search.
/*!
  \param Generator is the discrete, uniform generator.

  This class determines the deviate with a linear search on the sorted
  probabilities.
*/
template < class _Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorLinearSearchSorted :
   public DgPmfAndSumOrderedPairPointer<true> {
   //
   // Private types.
   //
private:

   typedef DgPmfAndSumOrderedPairPointer<true> Base;

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
   //! The integer type for the rebuild counter.
   typedef ads::CounterWithReset<>::Integer Counter;

   //
   // Member data.
   //
protected:

   //! The continuous uniform generator.
   ContinuousUniformGenerator _continuousUniformGenerator;
   //! The number of times you can set a PMF element between rebuilds.
   ads::CounterWithReset<> _rebuildCounter;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGeneratorLinearSearchSorted();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorLinearSearchSorted(DiscreteUniformGenerator* generator) :
      // The PMF array is empty.
      Base(),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator),
      _rebuildCounter(Counter(0)) {}

   //! Construct from the uniform generator and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorLinearSearchSorted(DiscreteUniformGenerator* generator,
                                       ForwardIterator begin,
                                       ForwardIterator end) :
      // The PMF array is empty.
      Base(),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator),
      _rebuildCounter(Counter(0)) {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorLinearSearchSorted
   (const DiscreteGeneratorLinearSearchSorted& other) :
      Base(other),
      _continuousUniformGenerator(other._continuousUniformGenerator),
      _rebuildCounter(other._rebuildCounter) {}

   //! Assignment operator.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorLinearSearchSorted&
   operator=(const DiscreteGeneratorLinearSearchSorted& other) {
      if (this != &other) {
         Base::operator=(other);
         _continuousUniformGenerator = other._continuousUniformGenerator;
         _rebuildCounter = other._rebuildCounter;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorLinearSearchSorted() {}

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
      // Loop until we get a valid deviate (non-zero weighted probability).
      result_type index;
      do {
         index = linearSearchChopDownGuardedPair
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

   //! Get the number of steps between rebuilds.
   Counter
   getStepsBetweenRebuilds() const {
      return _rebuildCounter.getReset();
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
public:

   bool
   operator==(const DiscreteGeneratorLinearSearchSorted& other) {
      return Base::operator==(other) &&
             _continuousUniformGenerator == other._continuousUniformGenerator &&
             _rebuildCounter == other._rebuildCounter;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
public:

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator start, ForwardIterator finish) {
      // Initialize the PMF.
      Base::initialize(start, finish);
      // Set an appropriate number of times between rebuilds.
      _rebuildCounter.setReset(std::max(std::size_t(1000), size()));
      // Sort the PMF.
      rebuild();
   }

   //! Set the probability mass function with the specified index.
   void
   set(std::size_t index, Number value) {
      --_rebuildCounter;
      Base::set(index, value);
   }

   //! Set the number of steps between rebuilds.
   void
   setStepsBetweenRebuilds(const Counter n) {
      _rebuildCounter.setReset(n);
   }

private:

   //! Recompute the sum of the PMF if necessary. Sort if necessary.
   void
   update() {
      // Recompute the sum of the PMF if necessary.
      Base::update();

      if (_rebuildCounter() <= 0) {
         rebuild();
      }
   }

   //! Sort the PMF.
   void
   rebuild() {
      _rebuildCounter.reset();
      Base::ValueGreater compare;
      std::sort(Base::begin(), Base::end(), compare);
      Base::computePointers();
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
      out << "Steps between rebuilds = " << getStepsBetweenRebuilds() << '\n';
   }

   //@}
};

} // namespace numerical
}

#endif
