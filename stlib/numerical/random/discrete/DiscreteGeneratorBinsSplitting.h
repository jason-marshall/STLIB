// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorBinsSplitting.h
  \brief discrete deviate.  BinsSplittingStacking.
*/

#if !defined(__numerical_DiscreteGeneratorBinsSplitting_h__)
#define __numerical_DiscreteGeneratorBinsSplitting_h__

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

// CONTINUE: Consider writing a class in which PMF's of zero would not have a
// bin.  This could be useful for the case that many reactions have zero
// propensity.

template < bool IsDynamic = true,
         class Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorBinsSplitting;

//! Discrete deviate.  Bins with splitting.
/*!
  \param Generator The discrete uniform generator.

  This generator stores the PMF in a bin array.  Note that the size of the
  PMF array must be no greater than the number of bins.
  Each probability is split across a number of bins.
  They are split so as to minimize the maximum bin height.  One can determine
  the splitting with a greedy algorithm:  Start by using a single bin
  for each probability.  Repeatedly expand the probability with greatest
  bin height into one more bin until all of the bins are being used.
  The correctness of this approach is easily proved by using recursion on
  the number of bins.

  Note that even probabilities with value zero occupy a bin.  This enables
  one to efficiently modify any probability.  One could design a data structure
  in which zero probabilities do not occupy bins.

  The rejection method is used to generate deviates.
  A portion of the bits are used for indexing into the array.
  The remaining bits are used for the rejection test.

  Consider the probabilities that are split across multiple bins.  The
  associated bin heights differ by at most a factor of two.  (Otherwise the
  probability for the shortest bins could be stored in one fewer bin and still
  have bin height less than the maximum.  Then the probability with maximum
  height could be expanded across an additional bin, thus reducing the maximum
  bin height.)  For the probabilities which are stored in a single
  bin, the bin heights are no greater than twice maximum height for those
  split across multiple bins.

  Let B be the number of bins, P be the number of
  probabilities, and S be the sum of the probabilities.
  If P = B, the maximum bin height is no greater than S.
  Otherwise, there are at most P - 1 bins that contain unsplit probabilities,
  and at least B - P + 1 bins that contain split probabilities.
  Otherwise, the maximum bin height is 4 S / (B - P + 1).
*/
template<class Generator>
class DiscreteGeneratorBinsSplitting<false, Generator> :
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
   // Member data.
   //

private:

   //! The discrete uniform generator.
   DiscreteUniformGenerator* _discreteUniformGenerator;

protected:

   //! An upper bound on the height of the bins.
   Number _heightUpperBound;
   //! The binned probability mass function.
   std::vector<Number> _binnedPmf;
   //! The indices of the deviate in the bin.
   std::vector<std::size_t> _deviateIndices;

   //! The sum of the PMF.
   Number _pmfSum;
   //! Probability mass function.  (This is scaled and may not sum to unity.)
   std::vector<Number> _pmf;
   //! The index of the first bin containing the PMF.
   std::vector<std::size_t> _binIndices;

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   DiscreteGeneratorBinsSplitting();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorBinsSplitting(DiscreteUniformGenerator* generator) :
      DgBinConstants(),
      _discreteUniformGenerator(generator),
      _heightUpperBound(-1),
      _binnedPmf(getNumberOfBins()),
      _deviateIndices(getNumberOfBins() + 1),
      _pmfSum(-1),
      _pmf(),
      _binIndices() {}

   //! Construct from the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorBinsSplitting(DiscreteUniformGenerator* generator,
                                  ForwardIterator begin,
                                  ForwardIterator end) :
      DgBinConstants(),
      _discreteUniformGenerator(generator),
      _heightUpperBound(-1),
      _binnedPmf(getNumberOfBins()),
      _deviateIndices(getNumberOfBins() + 1),
      _pmfSum(-1),
      _pmf(),
      _binIndices() {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinsSplitting
   (const DiscreteGeneratorBinsSplitting& other) :
      DgBinConstants(other),
      _discreteUniformGenerator(other._discreteUniformGenerator),
      _heightUpperBound(other._heightUpperBound),
      _binnedPmf(other._binnedPmf),
      _deviateIndices(other._deviateIndices),
      _pmfSum(other._pmfSum),
      _pmf(other._pmf),
      _binIndices(other._binIndices) {}

   //! Assignment operator.
   /*!
     \note The discrete,uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinsSplitting&
   operator=(const DiscreteGeneratorBinsSplitting& other) {
      if (this != &other) {
         DgBinConstants::operator=(other);
         _discreteUniformGenerator = other._discreteUniformGenerator;
         _heightUpperBound = other._heightUpperBound;
         _binnedPmf = other._binnedPmf;
         _deviateIndices = other._deviateIndices;
         _pmfSum = other._pmfSum;
         _pmf = other._pmf;
         _binIndices = other._binIndices;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorBinsSplitting() {}

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
      return _pmf[index];
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

     \note You must call initialize() after this function to (re)build the
     data structure.
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

   //! Update the data structure by recomputing the sum of the PMF's.
   void
   computePmfSum() {
      _pmfSum = std::accumulate(_pmf.begin(), _pmf.end(), 0.0);
   }

   //! Pack the PMF's into bins.
   /*!
     \pre The PMF must be sorted in descending order.
   */
   void
   packIntoBins();

   //! Fix the bin.
   void
   fixBin(std::size_t binIndex);

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
   // Inherited from the bin constants policy class.
private:

   //! The number of bits used for indexing the bins.
   using DgBinConstants::getIndexBits;

   //! The number of bins. 2^IndexBits.
   using DgBinConstants::getNumberOfBins;

   //! The mask for extracting the index.
   using DgBinConstants::getIndexMask;

   //! The inverse of the maximum height.  1 / (2^(32-_indexBits) - 1).
   using DgBinConstants::getMaxHeightInverse;
};



// CONTINUE
//#define MODIFY

//! discrete deviate.  Bins with splitting.
/*!
  \param Generator The discrete uniform generator.
*/
template<class Generator>
class DiscreteGeneratorBinsSplitting<true, Generator> :
   public DiscreteGeneratorBinsSplitting<false, Generator>,
      DgRepairCounter<true> {
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

   //! The static version of this data structure.
   typedef DiscreteGeneratorBinsSplitting<false, Generator> Base;
   //! The interface for repairing the data structure.
   typedef DgRepairCounter<true> RepairBase;

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
   //! The indices of the deviate in the bin.
   using Base::_deviateIndices;

   //! The sum of the PMF.
   using Base::_pmfSum;
   //! Probability mass function.  (This is scaled and may not sum to unity.)
   using Base::_pmf;
   //! The index of the first bin containing the PMF.
   using Base::_binIndices;

#ifdef MODIFY
   //! The index of the bin that will be modified when setting the PMF.
   std::vector<std::size_t> _binsToModify;
#endif
   //! The minimum allowed efficiency.
   Number _minimumEfficiency;
   //! The minimum allowed efficiency is the initial efficiency times this factor.
   Number _minimumEfficiencyFactor;

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   DiscreteGeneratorBinsSplitting();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorBinsSplitting(DiscreteUniformGenerator* generator) :
      Base(generator),
      RepairBase(),
#ifdef MODIFY
      _binsToModify(),
#endif
      _minimumEfficiency(-1),
      // CONTINUE
      _minimumEfficiencyFactor(0.75) {}

   //! Construct from the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorBinsSplitting(DiscreteUniformGenerator* generator,
                                  ForwardIterator begin,
                                  ForwardIterator end) :
      Base(generator),
      RepairBase(),
#ifdef MODIFY
      _binsToModify(),
#endif
      _minimumEfficiency(-1),
      _minimumEfficiencyFactor(0.75) {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinsSplitting
   (const DiscreteGeneratorBinsSplitting& other) :
      Base(other),
      RepairBase(other),
#ifdef MODIFY
      _binsToModify(other._binsToModify),
#endif
      _minimumEfficiency(other._minimumEfficiency),
      _minimumEfficiencyFactor(other._minimumEfficiencyFactor) {}

   //! Assignment operator.
   /*!
     \note The discrete,uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorBinsSplitting&
   operator=(const DiscreteGeneratorBinsSplitting& other) {
      if (this != &other) {
         Base::operator=(other);
         RepairBase::operator=(other);
#ifdef MODIFY
         _binsToModify = other._binsToModify;
#endif
         _minimumEfficiency = other._minimumEfficiency;
         _minimumEfficiencyFactor = other._minimumEfficiencyFactor;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorBinsSplitting() {}

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
         setPmf(i, iterator[i]);
      }
   }

   //! Update the data structure following calls to setPmf().
   void
   updatePmf();

   //! Set the number of steps between repairs.
   using RepairBase::setStepsBetweenRepairs;

   //! Set the minimum allowed efficiency factor.
   void
   setMinimumEfficiencyFactor(const Number efficiency) const {
      _minimumEfficiencyFactor = efficiency;
   }

private:

   //! Repair the data structure.
   /*!
     Recompute the PMF data.
   */
   void
   repair();

   //! Rebuild the bins.
   void
   rebuild();

   //! Update the minimum allowed efficiency.
   void
   updateMinimumAllowedEfficiency() {
      // Set the minimum allowed efficiency.
      _minimumEfficiency = computeEfficiency() * _minimumEfficiencyFactor;
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
};


} // namespace numerical
}

#define __numerical_random_DiscreteGeneratorBinsSplitting_ipp__
#include "stlib/numerical/random/discrete/DiscreteGeneratorBinsSplitting.ipp"
#undef __numerical_random_DiscreteGeneratorBinsSplitting_ipp__

#endif
