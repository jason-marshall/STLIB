// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorBinsSplittingStacking.h
  \brief discrete deviate.  BinsSplittingStacking.
*/

#if !defined(__numerical_DiscreteGeneratorBinsSplittingStacking_h__)
#define __numerical_DiscreteGeneratorBinsSplittingStacking_h__

#include "stlib/numerical/random/discrete/DgBinConstants.h"
#include "stlib/numerical/random/discrete/DgRepair.h"
#include "stlib/numerical/random/discrete/DgRebuild.h"

#include "stlib/numerical/random/uniform/Default.h"

#include "stlib/ads/algorithm/sort.h"
#include "stlib/ext/vector.h"

#include <boost/config.hpp>

#include <numeric>

namespace stlib
{
namespace numerical {

USING_STLIB_EXT_VECTOR_IO_OPERATORS;

//! Discrete deviate.  Bins with splitting and stacking.
/*!
  CONTINUE.
*/
template < bool IsDynamic = true,
         class Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorBinsSplittingStacking;



//! Discrete deviate.  Bins with splitting and stacking.
/*!
  CONTINUE.
*/
template<class Generator>
class DiscreteGeneratorBinsSplittingStacking<false, Generator> :
   public DgBinConstants {
   //
   // Public types.
   //
public:

   //! The discrete uniform generator.
   typedef Generator DiscreteUniformGenerator;
   //! The number type.
   typedef typename DgBinConstants::Number Number;
   //! The argument type.
   typedef void argument_type;
   //! The result type.
   typedef std::size_t result_type;

   //
   // More private types.
   //
private:

#if 0
   // CONTINUE: Using FixedArray's hurts performance.  I would not have
   // expected that.
   typedef ads::FixedArray<NumberOfBins, Number> PmfBinContainer;
   typedef ads::FixedArray < NumberOfBins + 1, std::size_t > IndexBinContainer;
#else
   typedef std::vector<Number> PmfBinContainer;
   typedef std::vector<std::size_t> IndexBinContainer;
#endif

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

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   DiscreteGeneratorBinsSplittingStacking();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorBinsSplittingStacking
   (DiscreteUniformGenerator* generator) :
      DgBinConstants(),
      _discreteUniformGenerator(generator),
      _heightUpperBound(-1),
#if 0
      _binnedPmf(),
      _deviateIndices(),
#else
      _binnedPmf(getNumberOfBins()),
      _deviateIndices(getNumberOfBins() + 1),
#endif
      _pmfSum(-1),
      _splittingEnd(-1),
      _pmf(),
      _permutation(),
      _rank(),
      _binIndices() {}

   //! Construct from the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorBinsSplittingStacking
   (DiscreteUniformGenerator* generator,
    ForwardIterator begin, ForwardIterator end) :
      DgBinConstants(),
      _discreteUniformGenerator(generator),
      _heightUpperBound(-1),
#if 0
      _binnedPmf(),
      _deviateIndices(),
#else
      _binnedPmf(getNumberOfBins()),
      _deviateIndices(getNumberOfBins() + 1),
#endif
      _pmfSum(-1),
      _splittingEnd(-1),
      _pmf(),
      _permutation(),
      _rank(),
      _binIndices() {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinsSplittingStacking
   (const DiscreteGeneratorBinsSplittingStacking& other) :
      DgBinConstants(other),
      _discreteUniformGenerator(other._discreteUniformGenerator),
      _heightUpperBound(other._heightUpperBound),
      _binnedPmf(other._binnedPmf),
      _deviateIndices(other._deviateIndices),
      _pmfSum(other._pmfSum),
      _splittingEnd(other._splittingEnd),
      _pmf(other._pmf),
      _permutation(other._permutation),
      _rank(other._rank),
      _binIndices(other._binIndices) {}

   //! Assignment operator.
   /*!
     \note The discrete,uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinsSplittingStacking&
   operator=(const DiscreteGeneratorBinsSplittingStacking& other) {
      if (this != &other) {
         DgBinConstants::operator=(other);
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
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorBinsSplittingStacking() {}

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
   operator()();

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! Get the probability mass function with the specified index.
   Number
   operator[](const std::size_t index) const {
      return _pmf[_rank[index]];
   }

   //! Get the number of possible deviates.
   std::size_t
   size() const {
      return _pmf.size();
   }

   //! Get the sum of the probability mass functions.
   Number
   sum() const {
      return _pmfSum;
   }

   //! Return true if the sum of the PMF is positive.
   bool
   isValid() const {
      return sum() > 0;
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
public:

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator begin, ForwardIterator end);

   //! Set the number of index bits.
   /*!
     This functionality is available only if the bin constants policy class
     is dynamic.
   */
   void
   setIndexBits(const std::size_t indexBits) {
      DgBinConstants::setIndexBits(indexBits);
      _binnedPmf.resize(getNumberOfBins());
      _deviateIndices.resize(getNumberOfBins() + 1);
   }

   //! Do nothing. This is only for compatibility with dynamic generators.
   void
   updatePmf() {
   }

protected:

   //! Rebuild the bins.
   void
   rebuild();

   //! Fix the PMF in the specified bin.
   void
   fixBin(std::size_t binIndex);

   //! Update the data structure by recomputing the sum of the PMF's.
   void
   computePmfSum() {
      _pmfSum = std::accumulate(_pmf.begin(), _pmf.end(), 0.0);
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
public:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const;

   //@}

   //--------------------------------------------------------------------------
   // Private member functions.
private:

   //! Pack the PMF's into bins.
   /*!
     \pre The PMF must be sorted in descending order.
   */
   void
   packIntoBins();

   //--------------------------------------------------------------------------
   // Inherited from the bin constants policy class.

   //! The number of bits used for indexing the bins.
   using DgBinConstants::getIndexBits;

   //! The number of bins. 2^IndexBits.
   using DgBinConstants::getNumberOfBins;

   //! The mask for extracting the index.
   using DgBinConstants::getIndexMask;

   //! The inverse of the maximum height.  1 / (2^(32-_indexBits) - 1).
   using DgBinConstants::getMaxHeightInverse;
};






//! Discrete deviate.  Bins with splitting and stacking.
/*!
  CONTINUE.
*/
template<class Generator>
class DiscreteGeneratorBinsSplittingStacking<true, Generator> :
   public DiscreteGeneratorBinsSplittingStacking<false, Generator>,
      DgRepairCounter<true>, DgRebuildCounter<true> {
   //
   // Public constants.
   //
public:

   //! The sum of the PMF is automatically updated.
   BOOST_STATIC_CONSTEXPR bool AutomaticUpdate = true;

   //
   // Private types.
   //
private:

   //! The bin constants policy class.
   typedef DiscreteGeneratorBinsSplittingStacking<false, Generator> Base;
   //! The interface for repairing the data structure.
   typedef DgRepairCounter<true> RepairBase;
   //! The interface for rebuilding the data structure.
   typedef DgRebuildCounter<true> RebuildBase;

   //
   // Public types.
   //
public:

   //! The discrete uniform generator.
   typedef typename Base::DiscreteUniformGenerator DiscreteUniformGenerator;
   //! The number type.
   typedef typename Base::Number Number;
   //! The argument type.
   typedef typename Base::argument_type argument_type;
   //! The result type.
   typedef typename Base::result_type result_type;

   //
   // Member data.
   //

private:

   //! An upper bound on the height of the bins.
   using Base::_heightUpperBound;
   //! The binned probability mass function.
   using Base::_binnedPmf;
   //! The indices of the first deviate in the bin.
   //using Base::_deviateIndices;

   //! The sum of the PMF.
   using Base::_pmfSum;
   //! The end of the PMF's that are split across multiple bins.
   using Base::_splittingEnd;
   //! Probability mass function.  (This is scaled and may not sum to unity.)
   using Base::_pmf;
   //! The permutation of the probability mass function array.
   //using Base::_permutation;
   //! The rank of the elements in _pmf array.
   using Base::_rank;
   //! The index of the first bin containing the PMF.
   using Base::_binIndices;

   //! The target efficiency when rebuilding the data structure.
   Number _targetEfficiency;
   //! The minimum allowed efficiency.
   Number _minimumEfficiency;

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   DiscreteGeneratorBinsSplittingStacking();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorBinsSplittingStacking
   (DiscreteUniformGenerator* generator) :
      Base(generator),
      RepairBase(),
      // By default, take 1000 steps between rebuilds.
      RebuildBase(1000L),
      _targetEfficiency(0.75),
      _minimumEfficiency(0.25) {}

   //! Construct from the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorBinsSplittingStacking
   (DiscreteUniformGenerator* generator,
    ForwardIterator begin, ForwardIterator end) :
      Base(generator),
      RepairBase(),
      // By default, take 1000 steps between rebuilds.
      RebuildBase(1000L),
      _targetEfficiency(0.75),
      _minimumEfficiency(0.25) {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinsSplittingStacking
   (const DiscreteGeneratorBinsSplittingStacking& other) :
      Base(other),
      RepairBase(other),
      RebuildBase(other),
      _targetEfficiency(other._targetEfficiency),
      _minimumEfficiency(other._minimumEfficiency) {}

   //! Assignment operator.
   /*!
     \note The discrete,uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinsSplittingStacking&
   operator=(const DiscreteGeneratorBinsSplittingStacking& other) {
      if (this != &other) {
         Base::operator=(other);
         RepairBase::operator=(other);
         RebuildBase::operator=(other);
         _targetEfficiency = other._targetEfficiency;
         _minimumEfficiency = other._minimumEfficiency;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorBinsSplittingStacking() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Random number generation.
   //@{
public:

   //! Seed the uniform random number generator.
   using Base::seed;

   //! Return a discrete deviate.
   using Base::operator();

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

   //! Compute the efficiency of the method.
   using Base::computeEfficiency;

   //! Get the number of steps between repairs.
   using RepairBase::getStepsBetweenRepairs;

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

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
public:

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator begin, ForwardIterator end);

   //! Set the number of index bits.
   using Base::setIndexBits;

   //! Set the probability mass function with the specified index.
   void
   set(std::size_t index, Number value);

   //! Set the probability mass functions.
   template<typename _RandomAccessIterator>
   void
   set(_RandomAccessIterator iterator) {
      for (std::size_t i = 0; i != _pmf.size(); ++i) {
         set(i, iterator[i]);
      }
   }

   //! Update the data structure following calls to setPmf().
   void
   updatePmf();

   //! Set the number of steps between repairs.
   using RepairBase::setStepsBetweenRepairs;

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

private:

   //! Repair the data structure.
   /*!
     Recompute the PMF data.
   */
   void
   repair();

   //! Fix the PMF in the specified bin.
   using Base::fixBin;

   //! Rebuild the bins.
   void
   rebuild();

   //! Pack the PMF's into bins.
   /*!
     \pre The PMF must be sorted in descending order.
   */
   void
   packIntoBins();

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
public:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const;

   //@}
};


} // namespace numerical
}

#define __numerical_random_DiscreteGeneratorBinsSplittingStacking_ipp__
#include "stlib/numerical/random/discrete/DiscreteGeneratorBinsSplittingStacking.ipp"
#undef __numerical_random_DiscreteGeneratorBinsSplittingStacking_ipp__

#endif
