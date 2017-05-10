// -*- C++ -*-

/*!
  \file numerical/random/discrete/DgPmfOrderedIndexRank.h
  \brief Ordered probability mass function for a discrete generator.
*/

#if !defined(__numerical_DgPmfOrderedIndexRank_h__)
#define __numerical_DgPmfOrderedIndexRank_h__

#include "stlib/numerical/random/discrete/linearSearch.h"

#include "stlib/ext/vector.h"

namespace stlib
{
namespace numerical {

//! Ordered probability mass function for a discrete generator.
/*!
  \param Pmf The policy class for the PMF.
  \param RebuildCounter The policy class for rebuilding (sorting) the
  probabilities.

  This is a base class for a ordered PMF.  It manages the permutation and
  ranks for the probabilities, but does not know how to sort them.
*/
template<class Pmf, class RebuildCounter>
class DgPmfOrderedIndexRank :
   public Pmf, RebuildCounter {
   //
   // Private types.
   //
private:

   //! The base type.
   typedef Pmf PmfBase;
   //! The interface for rebuilding the data structure.
   typedef RebuildCounter RebuildBase;

   //
   // Public types.
   //
public:

   //! The number type.
   typedef typename PmfBase::Number Number;
   //! The integer type for the repair counter.
   typedef typename RebuildBase::Counter Counter;

   //
   // Member data.
   //
private:

   //! The index of the element in the original probability mass function array.
   /*!
     This is useful when traversing the _pmf array.  We can efficiently go from
     the PMF value to its index.
   */
   std::vector<std::size_t> _index;
   //! The rank of the elements in _pmf array.
   /*!
     This is useful for manipulating the _pmf array by index.  \c _pmf[rank[i]]
     is the i_th element in the original PMF array.

     The rank array is the inverse of the index array mapping.  That is,
     \c _rank[_index[i]]==i and \c _index[_rank[i]]==i .
   */
   std::vector<std::size_t> _rank;

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
protected:

   //! Default constructor.
   DgPmfOrderedIndexRank() :
      PmfBase(),
      // By default, take 1000 steps between rebuilds.
      RebuildBase(Counter(1000)),
      // The arrays are empty.
      _index(),
      _rank() {}

   //! Copy constructor.
   DgPmfOrderedIndexRank(const DgPmfOrderedIndexRank& other) :
      PmfBase(other),
      RebuildBase(other),
      _index(other._index),
      _rank(other._rank) {}

   //! Assignment operator.
   DgPmfOrderedIndexRank&
   operator=(const DgPmfOrderedIndexRank& other) {
      if (this != &other) {
         PmfBase::operator=(other);
         RebuildBase::operator=(other);
         _index = other._index;
         _rank = other._rank;
      }
      return *this;
   }

   //! Destructor.
   ~DgPmfOrderedIndexRank() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Random number generation.
   //@{
protected:

   //! Return a discrete deviate.
   /*!
     Use the linear search of the base class.
   */
   std::size_t
   operator()(const Number r, const std::size_t offset) const {
      return _index[PmfBase::operator()(r, offset)];
   }

   //! Return a discrete deviate.
   /*!
     Use the linear search of the base class.
   */
   std::size_t
   operator()(const Number r) const {
      return _index[PmfBase::operator()(r)];
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! Get the probability mass function with the specified index.
   Number
   operator[](const std::size_t index) const {
      return PmfBase::operator[](_rank[index]);
   }

   //! Get the number of possible deviates.
   using PmfBase::size;

   //! Get the number of steps between rebuilds.
   using RebuildBase::getStepsBetweenRebuilds;

protected:

   //! Get the beginning of the probabilities in the PMF.
   using PmfBase::begin;

   //! Get the end of the probabilities in the PMF.
   using PmfBase::end;

   //! Get the index of the specified element.
   std::size_t
   getIndex(const std::size_t n) const {
      return _index[n];
   }

   //! Get the rank of the specified element.
   std::size_t
   getRank(const std::size_t n) const {
      return _rank[n];
   }

   //! Return true if the data structure should be rebuilt.
   using RebuildBase::shouldRebuild;

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
public:

   bool
   operator==(const DgPmfOrderedIndexRank& other) const {
      return PmfBase::operator==(other) && RebuildBase::operator==(other) &&
             _index == other._index && _rank == other._rank;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
public:

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator begin, ForwardIterator end) {
      // Initialize the PMF array.
      PmfBase::initialize(begin, end);

      _index.resize(size());
      _rank.resize(size());
      // Initialize the index array.
      for (std::size_t i = 0; i != _index.size(); ++i) {
         _index[i] = i;
      }
   }

   //! Set the number of steps between rebuilds.
   using RebuildBase::setStepsBetweenRebuilds;

protected:

   //! Set the probability mass function with the specified index.
   void
   set(std::size_t index, Number value) {
      PmfBase::set(_rank[index], value);
      RebuildBase::decrementRebuildCounter();
   }

   //! Set the probability mass functions.
   template<typename _RandomAccessIterator>
   void
   set(_RandomAccessIterator iterator) {
      for (std::size_t i = 0; i != size(); ++i) {
         PmfBase::set(_rank[i], iterator[i]);
      }
      RebuildBase::decrementRebuildCounter(size());
   }

   //! Reset the rebuild counter.
   using RebuildBase::resetRebuildCounter;

   //! Compute the ranks.
   void
   computeRanks() {
      for (std::size_t i = 0; i != _index.size(); ++i) {
         _rank[_index[i]] = i;
      }
   }

   //! Return the beginning of the indices.
   std::vector<std::size_t>::iterator
   getIndicesBeginning() {
      return _index.begin();
   }

   //! Return the end of the indices.
   std::vector<std::size_t>::iterator
   getIndicesEnd() {
      return _index.end();
   }

   //! Move the specified element up if it is greater than the preceding element.
   void
   bubbleUpDescending(const std::size_t deviate) {
      const std::size_t i = _rank[deviate];
      // If we should switch the order of this element and the one that
      // precedes it.
      if (i != 0 && PmfBase::operator[](i - 1) < PmfBase::operator[](i)) {
         // Update the ranks.
         ++_rank[_index[i-1]];
         --_rank[_index[i]];
         // Swap the values in the PMF and index arrays.
         const Number tmp = PmfBase::operator[](i - 1);
         PmfBase::set(i - 1, PmfBase::operator[](i));
         PmfBase::set(i, tmp);
         std::swap(_index[i-1], _index[i]);
      }
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
protected:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const {
      PmfBase::print(out);
      out << "Index = \n" << _index << '\n'
          << "Rank = \n" << _rank << '\n';
      RebuildBase::print(out);
   }

   //@}
};

} // namespace numerical
}

#endif
