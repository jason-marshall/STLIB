// -*- C++ -*-

#if !defined(__sfc_RefinementSortCodes_h__)
#define __sfc_RefinementSortCodes_h__

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

//! Sort codes using refinement.
/*!
  Use the refinementSortCodes() function instead of constructing an
  instance of this class. Here we implement this functionality with a
  class to simplify the internals.

  The input is a vector of codes. The codes are the result
  of using BlockCode::code(). That is, they are location codes at the
  highest level of refinement. We use refinement to partially sort the
  codes into virtual cells. Refinement is applied until no cell has more 
  than the maximum number of allowed objects or until the highest level of
  refinement is reached. The output is the partially sorted codes and number
  of levels of refinement that are required. Note that the code values
  are not modified.
*/
template<typename _Traits>
class RefinementSortCodes
{
  //
  // Types.
  //
public:

  //! The unsigned integer type is used for codes.
  typedef typename _Traits::Code Code;

  //
  // Constants.
  //
private:

  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Traits::Dimension;
  BOOST_STATIC_CONSTEXPR int RadixBits = Dimension;
  BOOST_STATIC_CONSTEXPR Code Radix = 1 << RadixBits;
  BOOST_STATIC_CONSTEXPR Code Mask = Radix - 1;

  //
  // Member data.
  //
private:

  BlockCode<_Traits> const& _blockCode;
  std::vector<Code>& _codes;
  std::size_t const _maxElementsPerCell;
  std::vector<Code> _buffer;
  std::array<Code*, Radix> _insertIterators;
  std::size_t _highestLevel;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  //! No need for a copy constructor.
  RefinementSortCodes(RefinementSortCodes const&) = delete;

  //! No need for an assignment operator.
  RefinementSortCodes&
  operator=(RefinementSortCodes const&) = delete;

  //! Allocate member data.
  /*!
    \param blockCode The data structure for calculating codes.
    \param codes The vector of codes.
    \param maxElementsPerCell The maximum number of elements per cell (except
    for cells at the highest level of refinement. */
  RefinementSortCodes(BlockCode<_Traits> const& blockCode,
                      std::vector<Code>* codes,
                      std::size_t maxElementsPerCell);

  //! Refine and sort.
  std::size_t
  operator()();

private:

  void
  _sort(std::size_t begin, std::size_t end, std::size_t level);

  //@}
};


//! Build the codes for a AdaptiveCells with refinement and sorting.
template<typename _Traits>
inline
std::size_t
refinementSortCodes(BlockCode<_Traits> const& blockCode,
                    std::vector<typename _Traits::Code>* codes,
                    std::size_t const maxElementsPerCell)
{
  return RefinementSortCodes<_Traits>(blockCode, codes, maxElementsPerCell)();
}


//! Convert a sequence of sorted object codes to a sequence of cell code/count pairs.
/*!
  \note The template parameter must be explicitly specified.
  \note Sequences of cell codes are terminated with the guard code, while
  sequences of object codes are not.
*/
template<typename _Traits>
void
objectCodesToCellCodeCountPairs
(std::vector<typename _Traits::Code> const& objectCodes,
 std::vector<std::pair<typename _Traits::Code, std::size_t> >*
 codeCountPairs);


} // namespace sfc
} // namespace stlib

#define __sfc_RefinementSortCodes_tcc__
#include "stlib/sfc/RefinementSortCodes.tcc"
#undef __sfc_RefinementSortCodes_tcc__

#endif
