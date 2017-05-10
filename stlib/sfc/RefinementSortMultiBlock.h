// -*- C++ -*-

#if !defined(__sfc_RefinementSortMultiBlock_h__)
#define __sfc_RefinementSortMultiBlock_h__

/*!
  \file
  \brief Build the codes for a AdaptiveCells with refinement and sorting.
*/

#include "stlib/sfc/BlockCode.h"

#include <cstring>

namespace stlib
{
namespace sfc
{

//! Build the codes for a AdaptiveCells with refinement and sorting.
/*!
  \deprecated This produces the same output as RefinementSort, but is less 
  efficient.

  Use the refinementSortMultiBlock() function instead of constructing an
  instance of this class. Here we implement this functionality with a
  class to simplify the internals.

  The input is a vector of code/index pairs. The codes are the result
  of using BlockCode::code(). That is, they are location codes at the
  highest level of refinement. The elements will be refined and sorted
  until no cells, except those at the highest level of refinement,
  contain more than the specified number of elements. We use the
  radix/counting sort. The output is a vector of code/index pairs
  suitable for building a AdaptiveCells data structure.
*/
template<typename _Traits>
class RefinementSortMultiBlock
{
  //
  // Types.
  //
public:

  //! The unsigned integer type is used for codes.
  typedef typename _Traits::Code Code;
  //! A code/index pair.
  typedef std::pair<Code, std::size_t> Pair;

  //
  // Constants.
  //
private:

  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Traits::Dimension;
  BOOST_STATIC_CONSTEXPR std::size_t NumOrthants = 1 << Dimension;
  BOOST_STATIC_CONSTEXPR std::size_t NumBlocks = 2;
  BOOST_STATIC_CONSTEXPR int RadixBits = NumBlocks * Dimension;
  BOOST_STATIC_CONSTEXPR Code Radix = 1 << RadixBits;
  BOOST_STATIC_CONSTEXPR Code Mask = Radix - 1;

  //
  // Member data.
  //
private:

  BlockCode<_Traits> const& _blockCode;
  std::vector<Pair>& _pairs;
  std::size_t const _maxElementsPerCell;
  std::vector<Pair> _buffer;
  std::array<Pair*, Radix> _insertIterators;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  //! No need for a copy constructor.
  RefinementSortMultiBlock(RefinementSortMultiBlock const&) = delete;

  //! No need for an assignment operator.
  RefinementSortMultiBlock&
  operator=(RefinementSortMultiBlock const&) = delete;

  //! Allocate member data.
  /*!
    \param blockCode The data structure for calculating codes.
    \param pairs The vector of integer/value pairs.
    \param maxElementsPerCell The maximum number of elements per cell (except
    for cells at the highest level of refinement. */
  RefinementSortMultiBlock(BlockCode<_Traits> const& blockCode, std::vector<Pair>* pairs,
                 std::size_t maxElementsPerCell);

  //! Refine and sort.
  void
  operator()();

private:

  void
  _sort(std::size_t begin, std::size_t end, std::size_t level);

  void
  _setLevelOrRecurse(std::size_t begin, std::size_t level,
                     std::size_t stride, std::size_t const* delimiters);

  void
  _setLevel(std::size_t begin, std::size_t end, std::size_t level);

  //@}
};


//! Build the codes for a AdaptiveCells with refinement and sorting.
/*!
  \deprecated This produces the same output as refinementSort(), but is less 
  efficient.
*/
template<typename _Traits>
inline
void
refinementSortMultiBlock
(BlockCode<_Traits> const& blockCode,
 std::vector<typename RefinementSortMultiBlock<_Traits>::Pair>* pairs,
 std::size_t const maxElementsPerCell)
{
  RefinementSortMultiBlock<_Traits>(blockCode, pairs, maxElementsPerCell)();
}


//! Build the codes for a AdaptiveCells, sort the objects.
/*!
  \param blockCode defines the geometry and calculates codes.
  \param objects will be sorted according to the codes.
  \param codeIndexPairs will be set to the sorted code/index pairs.
  \param maxElementsPerCell is the maximum allowed number of objects per cells.

  \deprecated This produces the same output as refinementSort(), but is less 
  efficient.
*/
template<typename _Traits, typename _Object>
void
refinementSortMultiBlock
(BlockCode<_Traits> const& blockCode,
 std::vector<_Object>* objects,
 std::vector<std::pair<typename _Traits::Code, std::size_t> >*
 codeIndexPairs,
 std::size_t maxElementsPerCell);


} // namespace sfc
} // namespace stlib

#define __sfc_RefinementSortMultiBlock_tcc__
#include "stlib/sfc/RefinementSortMultiBlock.tcc"
#undef __sfc_RefinementSortMultiBlock_tcc__

#endif
