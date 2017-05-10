// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorBinarySearchSorted.h
  \brief Discrete deviate.  Binary search.
*/

#if !defined(__numerical_DiscreteGeneratorBinarySearchSorted_h__)
#define __numerical_DiscreteGeneratorBinarySearchSorted_h__

#include "stlib/numerical/random/discrete/DgPmfOrderedPairPointer.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include "stlib/ads/counter/CounterWithReset.h"
#include "stlib/container/StaticArrayOfArrays.h"

#include <boost/config.hpp>

#include <algorithm>
#include <numeric>

namespace stlib
{
namespace numerical {

//! Discrete deviate.  Binary search.
/*!
  CONTINUE.
*/
template < class Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorBinarySearchSorted :
   public DgPmfOrderedPairPointer<false> {
   //
   // Private types.
   //
private:

   typedef DgPmfOrderedPairPointer<false> Base;

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
   typedef Generator DiscreteUniformGenerator;
   //! The continuous uniform generator.
   typedef ContinuousUniformGeneratorClosed<DiscreteUniformGenerator>
   ContinuousUniformGenerator;
   //! The number type.
   typedef typename Base::Number Number;
   //! A pair of a PMF value and index.
   typedef typename Base::PairValueIndex PairValueIndex;
   //! The argument type.
   typedef void argument_type;
   //! The result type.
   typedef std::size_t result_type;
   //! The integer type for the rebuild counter.
   typedef ads::CounterWithReset<>::Integer Counter;

   //
   // Member data.
   //
private:

   //! The continuous uniform generator.
   ContinuousUniformGenerator _continuousUniformGenerator;
   //! Cumulative distribution function.  (This is scaled and may not approach unity.)
   std::vector<Number> _cdf;
   //! The modified probability with the lowest index.
   std::size_t _firstModifiedProbability;
   //! Reaction influence array.
   const container::StaticArrayOfArrays<std::size_t>* _influence;
   //! The array of accumulated influencing probabilities. Store the indices so we can sort them.
   std::vector<PairValueIndex> _influencingProbabilities;
   //! The number of times you can set a PMF element between rebuilds.
   ads::CounterWithReset<> _rebuildCounter;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   DiscreteGeneratorBinarySearchSorted();


   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorBinarySearchSorted(DiscreteUniformGenerator* generator) :
      Base(),
      _continuousUniformGenerator(generator),
      _cdf(),
      _firstModifiedProbability(0),
      _influence(0),
      _influencingProbabilities(),
      _rebuildCounter(Counter(0)) {}

   //! Construct from the uniform generator, the influence array, and the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorBinarySearchSorted
   (DiscreteUniformGenerator* generator,
    const container::StaticArrayOfArrays<std::size_t>* influence,
    ForwardIterator begin, ForwardIterator end) :
      Base(),
      _continuousUniformGenerator(generator),
      _cdf(),
      _firstModifiedProbability(0),
      _influence(influence),
      _influencingProbabilities(),
      _rebuildCounter(Counter(0)) {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinarySearchSorted
   (const DiscreteGeneratorBinarySearchSorted& other) :
      Base(other),
      _continuousUniformGenerator(other._continuousUniformGenerator),
      _cdf(other._cdf),
      _firstModifiedProbability(other._firstModifiedProbability),
      _influence(other._influence),
      _influencingProbabilities(other._influencingProbabilities),
      _rebuildCounter(other._rebuildCounter) {}

   //! Assignment operator.
   /*!
     \note The discrete,uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinarySearchSorted&
   operator=(const DiscreteGeneratorBinarySearchSorted& other) {
      if (this != &other) {
         Base::operator=(other);
         _continuousUniformGenerator = other._continuousUniformGenerator;
         _cdf = other._cdf;
         _firstModifiedProbability = other._firstModifiedProbability;
         _influence = other._influence;
         _influencingProbabilities = other._influencingProbabilities;
         _rebuildCounter = other._rebuildCounter;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorBinarySearchSorted() {}

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
      // Note that this search will not find an event that has zero probability.
      // There is no need to loop until we get a valid deviate.
      std::size_t position =
         std::lower_bound(_cdf.begin(), _cdf.end(),
                          _continuousUniformGenerator() * _cdf.back()) -
         _cdf.begin();
      return _pmfPairs[position].second;
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
   /*!
     \pre The CDF must be computed before calling this function.
   */
   Number
   sum() const {
      return _cdf.back();
   }

   //! Return true if the sum of the PMF is positive.
   bool
   isValid() {
      // Recompute the PMF sum if necessary.
      update();
      return sum() > 0;
   }

   //! Get the number of steps between rebuilds.
   Counter
   getStepsBetweenRebuilds() const {
      return _rebuildCounter.getReset();
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
public:

   //! Set the probability mass function with the specified index.
   void
   set(std::size_t index, Number value) {
      --_rebuildCounter;
      Base::set(index, value);
      std::size_t position = Base::position(index);
      if (position <  _firstModifiedProbability) {
         _firstModifiedProbability = position;
      }
   }

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator begin, ForwardIterator end) {
      assert(_influence != 0);
      Base::initialize(begin, end);
#if 0
      Base::print(std::cout);
      std::cout << '\n';
#endif
      //! The array of influencing probabilities.
      _influencingProbabilities.resize(size());
      // Set an appropriate number of times between rebuilds.
      _rebuildCounter.setReset(std::max(std::size_t(1000), size()));
      // Allocate memory for the CDF.
      _cdf.resize(size());
      // Sort the PMF.
      rebuild();
#if 0
      Base::print(std::cout);
      std::cout << '\n';
#endif
   }

   //! Recompute the CDF.
   /*!
     This must be called after modifying the PMF and before drawing a deviate.
   */
   void
   updateSum() {
      // The CDF is correct up to _cdf[_firstModifiedProbability - 1].
      // Update the cumulative distribution function for the modified
      // probabilities.
      std::partial_sum(Base::pmfBegin() + _firstModifiedProbability,
                       Base::pmfEnd(), _cdf.begin() + _firstModifiedProbability);
      if (_firstModifiedProbability != 0) {
         const Number offset = _cdf[_firstModifiedProbability - 1];
         for (typename std::vector<Number>::iterator i =
                  _cdf.begin() + _firstModifiedProbability; i != _cdf.end(); ++i) {
            *i += offset;
         }
      }
      _firstModifiedProbability = _cdf.size();
   }

   //! Set the number of steps between rebuilds.
   void
   setStepsBetweenRebuilds(const Counter n) {
      _rebuildCounter.setReset(n);
   }

   //! Set the influence array.
   void
   setInfluence(const container::StaticArrayOfArrays<std::size_t>* influence) {
      _influence = influence;
   }

private:

   //! Recompute the sum of the PMF if necessary. Sort if necessary.
   void
   update() {
      if (_rebuildCounter() <= 0) {
         rebuild();
      }
   }

   //! Sort the PMF.
   void
   rebuild() {
      _rebuildCounter.reset();
      // Build the array of maximum influencing probabilities.
      for (std::size_t i = 0; i != size(); ++i) {
         _influencingProbabilities[i].first = 0;
         _influencingProbabilities[i].second = i;
      }
      std::size_t k;
      // For each probability.
      for (std::size_t i = 0; i != size(); ++i) {
         // For each probability that is influenced by it.
         for (std::size_t j = 0; j != _influence->size(i); ++j) {
            // If the i_th probabilty changes, the k_th will be affected.
            k = (*_influence)(i, j);
            _influencingProbabilities[k].first += operator[](i);
         }
      }

      // Sort in ascending order by influencing probabilities.
      std::sort(_influencingProbabilities.begin(),
                _influencingProbabilities.end(), Base::ValueLess());
      // Change the influencing probabilities to the PMF.
      for (Base::iterator i = _influencingProbabilities.begin();
            i != _influencingProbabilities.end(); ++i) {
         i->first = operator[](i->second);
      }
#if 0
      for (std::size_t i = 0; i != size(); ++i) {
         _influencingProbabilities[i].first =
            operator[](_influencingProbabilities[i].second);
      }
#endif
      // CONTINUE
#if 0
      // Copy to get the PMF in sorted order.
      std::copy(_influencingProbabilities.begin(),
                _influencingProbabilities.end(), _pmfPairs.begin());
#endif
      // Swap to get the PMF in sorted order.
      Base::_pmfPairs.swap(_influencingProbabilities);
      // Compute the pointers since we re-ordered.
      Base::computePointers();
      // Recompute the CDF.
      _firstModifiedProbability = 0;
      updateSum();
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
      out << "CDF =\n" << _cdf << '\n'
          << "First modified probability = " << _firstModifiedProbability << '\n'
          << "Influence = \n" << *_influence << '\n';
   }

   //@}
};

} // namespace numerical
}

#endif
