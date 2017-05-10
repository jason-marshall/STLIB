// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorBinned.h
  \brief discrete deviate.  Binned.
*/

#if !defined(__numerical_DiscreteGeneratorBinned_h__)
#define __numerical_DiscreteGeneratorBinned_h__

#include "stlib/numerical/random/discrete/DiscreteGeneratorBinsSplittingStacking.h"

#include "stlib/numerical/random/uniform/Default.h"

#include "stlib/ads/algorithm/sort.h"
#include "stlib/ext/vector.h"
#include "stlib/ext/array.h"

#include <numeric>

namespace stlib
{
namespace numerical {

//! discrete deviate.  Binned.
/*!
  \param T The number type.  By default it is double.

  CONTINUE.
*/
template < class Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorBinned {
public:

   //! The discrete uniform generator.
   typedef Generator DiscreteUniformGenerator;
   //! The number type.
   typedef double Number;
   //! The integer type for a counter.
   typedef std::ptrdiff_t Counter;
   //! The argument type.
   typedef void argument_type;
   //! The result type.
   typedef std::size_t result_type;

private:

   //! 2^8.
   enum {NumberOfBins = 256};

   typedef std::array<Number, NumberOfBins> PmfBinContainer;
   typedef std::array < std::size_t, NumberOfBins + 1 > IndexBinContainer;

   //
   // Member data.
   //

protected:

   //! The discrete uniform generator.
   DiscreteUniformGenerator* _discreteUniformGenerator;

   //! An upper bound on the height of the bins.
   Number _heightUpperBound;
   //! The binned probability mass function.
   PmfBinContainer _binnedPmf;
   //! The indices of the first deviate in the bin.
   IndexBinContainer _deviateIndices;

   //! The sum of the PMF.
   Number _pmfSum;
   //! The end of the PMF's that are split across multiple bins.
   std::size_t _splittingEnd;
   //! Probability mass function.  (This is scaled and may not sum to unity.)
   std::vector<Number> _pmf;
   //! The permutation of the probability mass function array.
   /*!
     This is useful when traversing the _pmf array.  We can efficiently go from
     the PMF value to its index.
   */
   std::vector<std::size_t> _permutation;
   //! The rank of the elements in _pmf array.
   /*!
     This is useful for manipulating the _pmf array by index.  \c _pmf[rank[i]]
     is the i_th element in the original PMF array.

     The rank array is the inverse of the permutation array mapping.  That is,
     \c _rank[_permutation[i]]==i and \c _permutation[_rank[i]]==i .
   */
   std::vector<std::size_t> _rank;
   //! The index of the first bin containing the PMF.
   std::vector<std::size_t> _binIndices;

   //! The number of times you can call setPmf() between repairs.
   Counter _stepsBetweenRepairs;
   //! The number of times you can call setPmf() before the next repair.
   Counter _stepsUntilNextRepair;
   //! The number of times you can call setPmf() between rebuilds.
   Counter _stepsBetweenRebuilds;
   //! The number of times you can call setPmf() before the next rebuild.
   Counter _stepsUntilNextRebuild;
   //! The target efficiency when rebuilding the data structure.
   Number _targetEfficiency;
   //! The minimum allowed efficiency.
   Number _minimumEfficiency;

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   DiscreteGeneratorBinned();

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorBinned(DiscreteUniformGenerator* generator) :
      _discreteUniformGenerator(generator),
      _heightUpperBound(-1),
      _binnedPmf(),
      _deviateIndices(),
      _pmfSum(-1),
      _splittingEnd(-1),
      _pmf(),
      _permutation(),
      _rank(),
      _binIndices(),
      // By default, take 1000 steps between repairs.
      _stepsBetweenRepairs(1000),
      _stepsUntilNextRepair(_stepsBetweenRepairs),
      // By default, take 1000 steps between rebuilds.
      _stepsBetweenRebuilds(1000),
      _stepsUntilNextRebuild(_stepsBetweenRebuilds),
      _targetEfficiency(0.75),
      _minimumEfficiency(0.25) {}

   //! Construct from the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorBinned(DiscreteUniformGenerator* generator,
                           ForwardIterator begin, ForwardIterator end) :
      _discreteUniformGenerator(generator),
      _heightUpperBound(-1),
      _binnedPmf(),
      _deviateIndices(),
      _pmfSum(-1),
      _splittingEnd(-1),
      _pmf(),
      _permutation(),
      _rank(),
      _binIndices(),
      // By default, take 1000 steps between repairs.
      _stepsBetweenRepairs(1000),
      _stepsUntilNextRepair(_stepsBetweenRepairs),
      // By default, take 1000 steps between rebuilds.
      _stepsBetweenRebuilds(1000),
      _stepsUntilNextRebuild(_stepsBetweenRebuilds),
      _targetEfficiency(0.75),
      _minimumEfficiency(0.25) {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinned(const DiscreteGeneratorBinned& other) :
      _discreteUniformGenerator(other._discreteUniformGenerator),
      _heightUpperBound(other._heightUpperBound),
      _binnedPmf(other._binnedPmf),
      _deviateIndices(other._deviateIndices),
      _pmfSum(other._pmfSum),
      _splittingEnd(other._splittingEnd),
      _pmf(other._pmf),
      _permutation(other._permutation),
      _rank(other._rank),
      _binIndices(other._binIndices),
      _stepsBetweenRepairs(other._stepsBetweenRepairs),
      _stepsUntilNextRepair(other._stepsUntilNextRepair),
      _stepsBetweenRebuilds(other._stepsBetweenRebuilds),
      _stepsUntilNextRebuild(other._stepsUntilNextRebuild),
      _targetEfficiency(other._targetEfficiency),
      _minimumEfficiency(other._minimumEfficiency) {}

   //! Assignment operator.
   /*!
     \note The discrete,uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinned&
   operator=(const DiscreteGeneratorBinned& other) {
      if (this != &other) {
         _discreteUniformGenerator = other._discreteUniformGenerator;
         _heightUpperBound = other._heightUpperBound;
         _binnedPmf = other._binnedPmf;
         _deviateIndices = other._deviateIndices;
         _pmfSum = other._pmfSum;
         _splittingEnd = other._splittingEnd;
         _pmf = other._pmf;
         _permutation = other._permutation;
         _rank = other._rank;
         _binIndices = other._binIndices;
         _stepsBetweenRepairs = other._stepsBetweenRepairs;
         _stepsUntilNextRepair = other._stepsUntilNextRepair;
         _stepsBetweenRebuilds = other._stepsBetweenRebuilds;
         _stepsUntilNextRebuild = other._stepsUntilNextRebuild;
         _targetEfficiency = other._targetEfficiency;
         _minimumEfficiency = other._minimumEfficiency;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorBinned() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Random number generation.
   //@{

   //! Seed the uniform random number generator.
   void
   seed(const typename DiscreteUniformGenerator::result_type seedValue) {
      _discreteUniformGenerator->seed(seedValue);
   }

   //! Return a discrete deviate.
   /*!
     This functor checks if the data structure need repairing or rebuilding.
   */
   result_type
   operator()();

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{

   //! Get the probability mass function with the specified index.
   Number
   getPmf(const std::size_t index) const {
      return _pmf[_rank[index]];
   }

   //! Get the number of possible deviates.
   std::size_t
   getSize() const {
      return _pmf.size();
   }

   //! Get the sum of the probability mass functions.
   Number
   getPmfSum() const {
      return _pmfSum;
   }

   //! Return true if the sum of the PMF is positive.
   bool
   isValid() const {
      return getPmfSum() > 0;
   }

   //! Get the number of steps between repairs.
   Counter
   getStepsBetweenRepairs() const {
      return _stepsBetweenRepairs;
   }

   //! Get the number of steps between rebuilds.
   Counter
   getStepsBetweenRebuilds() const {
      return _stepsBetweenRebuilds;
   }

   //! Get the target efficiency.
   /*!
     Rebuilding is only performed if the efficiency falls below this threshhold.
   */
   Number
   getTargetEfficiency() const {
      return _targetEfficiency;
   }

   //! Get the minimum allowed efficiency.
   Number
   getMinimumEfficiency() const {
      return _minimumEfficiency;
   }

   //! Compute the efficiency of the method.
   Number
   computeEfficiency() const {
      return _pmfSum / (_heightUpperBound * _binnedPmf.size());
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator begin, ForwardIterator end);

   //! Repair the data structure.
   /*!
     Recompute the PMF data.
   */
   void
   repair() {
      updatePmf();
      _stepsUntilNextRepair = _stepsBetweenRepairs;
   }

   //! Set the number of steps between repairs.
   void
   setStepsBetweenRepairs(const Counter n) {
      assert(n > 0);
      _stepsBetweenRepairs = n;
      _stepsUntilNextRepair = _stepsBetweenRepairs;
   }

   //! Rebuild the bins.
   void
   rebuild();

   //! Set the number of steps between rebuilds.
   void
   setStepsBetweenRebuilds(const Counter n) {
      assert(n > 0);
      _stepsBetweenRebuilds = n;
      _stepsUntilNextRebuild = _stepsBetweenRebuilds;
   }

   //! Set the target efficiency.
   /*!
     Rebuilding is only performed if the efficiency falls below this threshhold.
     Usually set it to a number between 0.5 and 1.
   */
   void
   setTargetEfficiency(const Number efficiency) const {
      _targetEfficiency = efficiency;
   }

   //! Set the minimum allowed efficiency.
   void
   setMinimumEfficiency(const Number efficiency) const {
      _minimumEfficiency = efficiency;
   }

   //! Set the probability mass function with the specified index.
   /*!
     This will update the data structure.
   */
   void
   setPmf(std::size_t index, Number value);

   //! Set the probability mass function with the specified index.
   /*!
     \note After calling this function, you must call updatePmf() before
     computing deviates.
   */
   void
   setPmfWithoutUpdating(std::size_t index, Number value) {
      // Update the PMF array.
      _pmf[_rank[index]] = value;

      --_stepsUntilNextRebuild;
   }

   //! Update the data structure following calls to setPmfWithoutUpdating() .
   void
   updatePmf();

   //! Update the data structure by recomputing the sum of the PMF's.
   void
   computePmfSum() {
      _pmfSum = std::accumulate(_pmf.begin(), _pmf.end(), 0.0);
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{

   //! Print information about the data structure.
   void
   print(std::ostream& out) const;

   //@}

private:

   //! Pack the PMF's into bins.
   /*!
     \pre The PMF must be sorted in descending order.
   */
   void
   packIntoBins();

   //! The number of bits used for indexing.
   static
   std::size_t
   IndexBits() {
      return 8;
   }

   //! 1 / (2^24 - 1)
   static
   Number
   MaxHeightInverse() {
      return 1.0 / (std::numeric_limits<unsigned>::max() / NumberOfBins - 1);
   }

   //! Get an index by masking with this.
   static
   unsigned
   IndexMask() {
      return 0x000000FF;
   }

   void
   fixBin(std::size_t binIndex);
};

} // namespace numerical
}

#define __numerical_random_DiscreteGeneratorBinned_ipp__
#include "stlib/numerical/random/discrete/DiscreteGeneratorBinned.ipp"
#undef __numerical_random_DiscreteGeneratorBinned_ipp__

#endif
