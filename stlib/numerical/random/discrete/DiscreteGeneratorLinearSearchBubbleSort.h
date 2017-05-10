// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorLinearSearchBubbleSort.h
  \brief discrete deviate.  Linear search.
*/

#if !defined(__numerical_DiscreteGeneratorLinearSearchBubbleSort_h__)
#define __numerical_DiscreteGeneratorLinearSearchBubbleSort_h__

#include "stlib/numerical/random/discrete/DgPmfAndSumOrderedPairPointer.h"
#include "stlib/numerical/random/discrete/linearSearch.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <boost/config.hpp>

#include <algorithm>
#include <numeric>

namespace stlib
{
namespace numerical {

//! Discrete deviate.  Linear search.
/*!
  \param Generator is the discrete, uniform generator.

  This class determines the deviate with a linear search on the bubble sorted
  probabilities.
*/
template < class _Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorLinearSearchBubbleSort :
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

   //
   // Member data.
   //
protected:

   //! The continuous uniform generator.
   ContinuousUniformGenerator _continuousUniformGenerator;
   //! The last deviate drawn.
   mutable result_type _deviate;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGeneratorLinearSearchBubbleSort();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorLinearSearchBubbleSort(DiscreteUniformGenerator* generator) :
      // The PMF array is empty.
      Base(),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator),
      _deviate(0) {}

   //! Construct from the uniform generator and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorLinearSearchBubbleSort(DiscreteUniformGenerator* generator,
                                           ForwardIterator begin,
                                           ForwardIterator end) :
      // The PMF array is empty.
      Base(),
      // Make a continuous uniform generator using the discrete uniform generator.
      _continuousUniformGenerator(generator),
      _deviate(0) {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorLinearSearchBubbleSort
   (const DiscreteGeneratorLinearSearchBubbleSort& other) :
      Base(other),
      _continuousUniformGenerator(other._continuousUniformGenerator),
      _deviate(other._deviate) {}

   //! Assignment operator.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorLinearSearchBubbleSort&
   operator=(const DiscreteGeneratorLinearSearchBubbleSort& other) {
      if (this != &other) {
         Base::operator=(other);
         _continuousUniformGenerator = other._continuousUniformGenerator;
         _deviate = other._deviate;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorLinearSearchBubbleSort() {}

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
      do {
         _deviate = linearSearchChopDownGuardedPair
                    (Base::begin(), Base::end(), _continuousUniformGenerator() * sum());
      }
      while (operator[](_deviate) == 0);
      return _deviate;
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
   //! Return the position of the specified event in the ordered array.
   using Base::position;
   //! Return true if the sum of the PMF is positive.
   using Base::isValid;

   //! Return the expected cost for generating a deviate.
   Number
   cost() const {
      Number c = 0;
      for (std::size_t i = 0; i != size(); ++i) {
         c += (i + 1) * _pmfPairs[i].first;
      }
      return c / sum();
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
public:

   bool
   operator==(const DiscreteGeneratorLinearSearchBubbleSort& other) {
      return Base::operator==(other) &&
             _continuousUniformGenerator == other._continuousUniformGenerator &&
             _deviate == other._deviate;
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
   initialize(ForwardIterator start, ForwardIterator finish) {
      // Initialize the PMF.
      Base::initialize(start, finish);
      // Valid value for the last deviate drawn.
      _deviate = 0;
      // Sort the PMF.
      sort();
   }

private:

   //! Recompute the sum of the PMF if necessary.
   void
   update() {
      // Recompute the sum of the PMF if necessary.
      Base::update();

      //
      // Bubble up the event for the last drawn deviate.
      //
#ifdef STLIB_DEBUG
      assert(_deviate < size());
#endif
      const std::size_t p = Base::position(_deviate);
      if (p != 0) {
         // Try a big jump.
         if (_pmfPairs[p/2].first < _pmfPairs[p].first) {
            swapPositions(p / 2, p);
         }
         // Try a little jump.
         else if (_pmfPairs[p-1].first < _pmfPairs[p].first) {
            swapPositions(p - 1, p);
         }
      }
   }

#if 0
   //! Swap the two elements in the PMF array.
   void
   swap(const std::size_t i, const std::size_t j) {
      // Swap the value/index pairs.
      std::swap(*_pointers[i], *_pointers[j]);
      // Swap the pointers
      std::swap(_pointers[i], _pointers[j]);
   }
#endif

   //! Swap the two elements in the PMF array.
   void
   swapPositions(const std::size_t i, const std::size_t j) {
      // Swap the pointers
      std::swap(_pointers[_pmfPairs[i].second], _pointers[_pmfPairs[j].second]);
      // Swap the value/index pairs.
      std::swap(_pmfPairs[i], _pmfPairs[j]);
   }

   //! Sort the PMF.
   void
   sort() {
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
   }

   //@}
};

} // namespace numerical
}

#endif
