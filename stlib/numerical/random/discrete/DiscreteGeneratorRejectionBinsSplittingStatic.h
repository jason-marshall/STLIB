// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorRejectionBinsSplittingStatic.h
  \brief discrete deviate.  Rejection with bins and splitting.
*/

#if !defined(__numerical_DiscreteGeneratorRejectionBinsSplittingStatic_h__)
#define __numerical_DiscreteGeneratorRejectionBinsSplittingStatic_h__

#include "stlib/numerical/random/discrete/DgBinConstants.h"

#include "stlib/numerical/random/uniform/Default.h"

#include "stlib/ads/algorithm/sort.h"
#include "stlib/ads/functor/compare_handle.h"
#include "stlib/ext/vector.h"

#include <numeric>

namespace stlib
{
namespace numerical
{

USING_STLIB_EXT_VECTOR_IO_OPERATORS;

//! Discrete deviate.  Bins with splitting.
/*!
  Static specialization.

  \param ExactlyBalance A boolean value that indicates if the PMF should be
  exactly balanced when when distributing it over the bins.  Exact balancing
  minimized the maximum bin height.  You will usually get a little better
  performance by using exact balancing.  If memory usage is critical, you
  can go with approximate balancing.
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
  The maximum bin height is 4 S / (B - P + 1).
*/
template < bool ExactlyBalance = true,
         class Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorRejectionBinsSplittingStatic :
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
   // Private types.
   //
private:

   //
   // Member data.
   //

private:

   //! The discrete uniform generator.
   DiscreteUniformGenerator* _discreteUniformGenerator;

protected:

   // Bin data.

   //! An upper bound on the height of the bins.
   Number _heightUpperBound;
   //! The indices of the deviate in the bin.
   std::vector<std::size_t> _deviateIndices;

   // PMF data.

   //! The sum of the PMF.
   Number _sum;
   //! Probability mass function.  (This is scaled and may not sum to unity.)
   std::vector<Number> _pmf;
   //! The inverse of the number of bins for each probability.
   std::vector<Number> _inverseSizes;
   //! Sorted probability mass function.
   std::vector<typename std::vector<Number>::const_iterator> _sortedPmf;
   //! The indices of the bins containing the PMF.
   /*!
     We only use this if ExactlyBalance is true.
   */
   std::vector<std::vector<std::size_t> > _binIndices;

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   DiscreteGeneratorRejectionBinsSplittingStatic();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorRejectionBinsSplittingStatic
   (DiscreteUniformGenerator* generator) :
      DgBinConstants(),
      _discreteUniformGenerator(generator),
      // Bin data.
      _heightUpperBound(-1),
      _deviateIndices(getNumberOfBins()),
      // PMF data.
      _sum(-1),
      _pmf(),
      _inverseSizes(),
      _sortedPmf(),
      _binIndices() {
   }

   //! Construct from the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorRejectionBinsSplittingStatic
   (DiscreteUniformGenerator* generator,
    ForwardIterator begin, ForwardIterator end) :
      DgBinConstants(),
      _discreteUniformGenerator(generator),
      // Bin data.
      _heightUpperBound(-1),
      _deviateIndices(getNumberOfBins()),
      // PMF data.
      _sum(-1),
      _pmf(),
      _inverseSizes(),
      _sortedPmf(),
      _binIndices() {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorRejectionBinsSplittingStatic
   (const DiscreteGeneratorRejectionBinsSplittingStatic& other) :
      DgBinConstants(other),
      _discreteUniformGenerator(other._discreteUniformGenerator),
      // Bin data.
      _heightUpperBound(other._heightUpperBound),
      _deviateIndices(other._deviateIndices),
      // PMF data.
      _sum(other._sum),
      _pmf(other._pmf),
      _inverseSizes(other._inverseSizes),
      // We don't need to copy the data, but this makes the array the right size.
      _sortedPmf(other._sortedPmf),
      _binIndices(other._binIndices) {}

   //! Assignment operator.
   /*!
     \note The discrete,uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorRejectionBinsSplittingStatic&
   operator=(const DiscreteGeneratorRejectionBinsSplittingStatic& other) {
      if (this != &other) {
         DgBinConstants::operator=(other);
         _discreteUniformGenerator = other._discreteUniformGenerator;
         // Bin data.
         _heightUpperBound = other._heightUpperBound;
         _deviateIndices = other._deviateIndices;
         // PMF data.
         _sum = other._sum;
         _pmf = other._pmf;
         _inverseSizes = other._inverseSizes;
         // We don't need to copy the data, but this makes the array the right
         // size.
         _sortedPmf = other._sortedPmf;
         _binIndices = other._binIndices;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorRejectionBinsSplittingStatic() {}

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
      return _sum;
   }

   //! Return true if the sum of the PMF is positive.
   bool
   isValid() const {
      return _sum > 0;
   }

   //! Compute the efficiency of the method.
   Number
   computeEfficiency() const {
      return _sum / (_heightUpperBound * getNumberOfBins());
   }

protected:

   //! Compute the bin height for the given probability.
   /*!
     This is the same for each associated bin.
   */
   Number
   computeBinHeight(const std::size_t index) const {
      return _pmf[index] * _inverseSizes[index];
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
      _deviateIndices.resize(getNumberOfBins());
   }

protected:

   //! Rebuild the bins.
   void
   rebuild();

   //! Update the data structure by recomputing the sum of the PMF's.
   void
   computePmfSum() {
      _sum = std::accumulate(_pmf.begin(), _pmf.end(), 0.0);
   }

   //! Pack the PMF's into bins.
   /*!
     \pre _sum has been computed.
   */
   void
   packIntoBins() {
     _packIntoBins(std::integral_constant<bool, ExactlyBalance>());
   }

private:

   //! Compute an upper bound on the bin height.
   void
   computeUpperBound() {
      _heightUpperBound = 0;
      for (std::size_t i = 0; i != _pmf.size(); ++i) {
         _heightUpperBound = std::max(_heightUpperBound, computeBinHeight(i));
      }
   }

   //! Pack the PMF's into bins.  Fast, approximate balancing.
   void
   _packIntoBins(std::false_type /*ExactlyBalance*/);

   //! Pack the PMF's into bins.  Slower, exact balancing.
   void
   _packIntoBins(std::true_type /*ExactlyBalance*/);

   //! Make a sorted array of the PMF.
   void
   computeSortedPmf() {
      for (std::size_t i = 0; i != _sortedPmf.size(); ++i) {
         _sortedPmf[i] = _pmf.begin() + i;
      }
      std::sort(_sortedPmf.begin(), _sortedPmf.end(),
                ads::constructLessByHandle
                <typename std::vector<Number>::const_iterator>());
   }

   //! Compute the inverse of the number of bins for each probability.
   void
   computeInverseSizes();

   //! Balance to minimize the maximum bin height.
   void
   balance();

   //! Trade bins to reduce the maximum bin height.
   bool
   tradeBins();

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

} // namespace numerical
}

#define __numerical_random_DiscreteGeneratorRejectionBinsSplittingStatic_ipp__
#include "stlib/numerical/random/discrete/DiscreteGeneratorRejectionBinsSplittingStatic.ipp"
#undef __numerical_random_DiscreteGeneratorRejectionBinsSplittingStatic_ipp__

#endif
