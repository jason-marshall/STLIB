// -*- C++ -*-

#if !defined(__sfc_BuildCodesFromCounts_h__)
#define __sfc_BuildCodesFromCounts_h__

/*!
  \file
  \brief Build the codes for a AdaptiveCells from sorted code/count pairs.
*/

#include "stlib/sfc/BlockCode.h"

#include <numeric>

#include <cstring>

namespace stlib
{
namespace sfc
{


//! Build the codes for a AdaptiveCells from sorted code/count pairs.
/*!
  Use the buildCodesFromCounts() function instead of constructing an
  instance of this class. Here we implement this functionality with a
  class to simplify the internals.

  The input is a sorted vector of code/count pairs. The codes must all
  have the same level of refinement. This will be the maximum level of
  refinement for the output cells. For the output, cells are refined
  until no cells, except those at the highest level of refinement,
  contain more than the specified number of elements. The output is a
  vector of sorted codes.
*/
template<typename _Traits>
class BuildCodesFromCounts
{
  //
  // Types.
  //
public:

  //! The unsigned integer type is used for codes.
  typedef typename _Traits::Code Code;
  //! A code/count pair.
  typedef std::pair<Code, std::size_t> Pair;

  //
  // Constants.
  //
private:

  BOOST_STATIC_CONSTEXPR std::size_t NumChildren =
    BlockCode<_Traits>::NumChildren;

  //
  // Member data.
  //
private:

  BlockCode<_Traits> const& _blockCode;
  std::vector<Code> _codes;
  std::vector<std::size_t> _partialSum;
  std::size_t const _maxElementsPerCell;
  std::size_t const _maxLevel;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  //! No need for a copy constructor.
  BuildCodesFromCounts(BuildCodesFromCounts const&) = delete;

  //! No need for an assignment operator.
  BuildCodesFromCounts&
  operator=(BuildCodesFromCounts const&) = delete;

  //! Allocate member data.
  /*!
    \param blockCode The data structure for calculating codes.
    \param codes The vector of codes.
    \param maxElementsPerCell The maximum number of elements per cell (except
    for cells at the highest level of refinement. */
  BuildCodesFromCounts(BlockCode<_Traits> const& blockCode,
                       std::vector<Pair> const& pairs,
                       std::size_t maxElementsPerCell);

  //! Refine to determine the output codes.
  void
  operator()(std::vector<typename _Traits::Code>* outputCodes) const;

private:

  void
  _build(std::size_t begin, std::size_t end, Code code,
         std::vector<typename _Traits::Code>* outputCodes) const;

  //@}
};


//! Build the codes for a AdaptiveCells from sorted code/count pairs.
template<typename _Traits>
inline
void
buildCodesFromCounts
(BlockCode<_Traits> const& blockCode,
 std::vector<std::pair<typename _Traits::Code, std::size_t> > const&
 pairs,
 std::size_t const maxElementsPerCell,
 std::vector<typename _Traits::Code>* outputCodes)
{
  BuildCodesFromCounts<_Traits>(blockCode, pairs, maxElementsPerCell)
    (outputCodes);
}


} // namespace sfc
} // namespace stlib

#define __sfc_BuildCodesFromCounts_tcc__
#include "stlib/sfc/BuildCodesFromCounts.tcc"
#undef __sfc_BuildCodesFromCounts_tcc__

#endif
