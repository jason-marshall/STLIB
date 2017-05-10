// -*- C++ -*-

#if !defined(__sfc_BuildFromBlockCodeSizePairs_h__)
#define __sfc_BuildFromBlockCodeSizePairs_h__

/*!
  \file
  \brief Build the codes for a AdaptiveCells from code/size pairs.
*/

#include "stlib/sfc/BlockCode.h"
#include "stlib/sfc/Codes.h"

namespace stlib
{
namespace sfc
{

//! Build the codes for a AdaptiveCells from code/size pairs.
/*!
  Use the buildFromBlockCodeSizePairs() function instead of constructing an
  instance of this class. Here we implement this functionality with a
  class to simplify the internals.
*/
template<typename _Traits>
class BuildFromBlockCodeSizePairs
{
  //
  // Types.
  //
public:

  //! The unsigned integer type is used for codes.
  typedef typename _Traits::Code Code;
  //! A code/size pair.
  typedef std::pair<Code, std::size_t> Pair;

  //
  // Member data.
  //
private:

  BlockCode<_Traits> const& _blockCode;
  std::vector<Pair> const& _input;
  std::size_t const _maxElementsPerCell;
  std::vector<Pair>* const _output;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
  */
  //@{
public:

  //! No need for a copy constructor.
  BuildFromBlockCodeSizePairs(BuildFromBlockCodeSizePairs const&) = delete;

  //! No need for an assignment operator.
  BuildFromBlockCodeSizePairs&
  operator=(BuildFromBlockCodeSizePairs const&) = delete;

  //! Store const references to the input data.
  /*!
    \param blockCode The data structure for calculating codes.
    \param pairs The vector of code/size pairs.
    \param maxElementsPerCell The maximum number of elements per cell (except,
    possibly, for cells at the highest level of refinement. */
  BuildFromBlockCodeSizePairs(BlockCode<_Traits> const& blockCode,
                              std::vector<Pair> const& pairs,
                              std::size_t maxElementsPerCell,
                              std::vector<Pair>* outputCodes);

  //! Record the output codes.
  void
  operator()() const;

private:

 std::size_t
 _build(Code code, std::size_t i) const;

  //@}
};


//! Build the codes for a AdaptiveCells from code/size pairs.
template<typename _Traits>
inline
void
buildFromBlockCodeSizePairs
(BlockCode<_Traits> const& blockCode,
 std::vector<typename BuildFromBlockCodeSizePairs<_Traits>::Pair> const& input,
 std::size_t const maxElementsPerCell,
 std::vector<typename BuildFromBlockCodeSizePairs<_Traits>::Pair>* output)
{
  BuildFromBlockCodeSizePairs<_Traits>(blockCode, input, maxElementsPerCell,
                                       output)();
}


} // namespace sfc
} // namespace stlib

#define __sfc_BuildFromBlockCodeSizePairs_tcc__
#include "stlib/sfc/BuildFromBlockCodeSizePairs.tcc"
#undef __sfc_BuildFromBlockCodeSizePairs_tcc__

#endif
