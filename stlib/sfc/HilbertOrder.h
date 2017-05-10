// -*- C++ -*-

#if !defined(__sfc_HilbertOrder_h__)
#define __sfc_HilbertOrder_h__

/*!
  \file
  \brief Class for working with Hilbert codes.
*/

#include "stlib/ext/array.h"

#include <cstdint>

namespace stlib
{
namespace sfc
{

//! Class for generating Hilbert codes.
template<std::size_t _Dimension, typename _Code = std::uint64_t>
class HilbertOrder
{
public:

  //! Return the Hilbert code for the specified discrete cell.
  /*!
   \param indices The indices of the cell.
   \param numLevels The number of levels of refinement.
  */
  _Code
  code(std::array<_Code, _Dimension> const& indices,
       std::size_t numLevels) const;
};

} // namespace sfc
}

#define __sfc_HilbertOrder_tcc__
#include "stlib/sfc/HilbertOrder.tcc"
#undef __sfc_HilbertOrder_tcc__

#endif
