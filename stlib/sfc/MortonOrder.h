// -*- C++ -*-

#if !defined(__sfc_MortonOrder_h__)
#define __sfc_MortonOrder_h__

/*!
  \file
  \brief Class for working with Morton codes.
*/

#include "stlib/sfc/DilateBits.h"

namespace stlib
{
namespace sfc
{

//! Class for generating Morton codes.
template<std::size_t _Dimension, typename _Code = std::uint64_t>
class MortonOrder
{
private:

  //! Functor for dilating bits.
  DilateBits<_Dimension> _dilate;

public:

  //! Return the Morton code for the specified discrete cell.
  /*!
   \param indices The indices of the cell.
   \param numLevels The number of levels of refinement.

   The indices are each in the range [0..2<sup>numLevels</sup>). The indices 
   are dilated and then combined with bitwise or operations.
  */
  _Code
  code(const std::array<_Code, _Dimension>& indices,
       std::size_t numLevels) const;
};

} // namespace sfc
} // namespace stlib

#define __sfc_MortonOrder_tcc__
#include "stlib/sfc/MortonOrder.tcc"
#undef __sfc_MortonOrder_tcc__

#endif
