// -*- C++ -*-

#if !defined(__sfc_sortByHilbertCodes_h__)
#define __sfc_sortByHilbertCodes_h__

/*!
  \file
  \brief Sort objects by their spatial indexing codes.
*/

#include "stlib/sfc/sortByCodes.h"
#include "stlib/sfc/HilbertOrder.h"

namespace stlib
{
namespace sfc
{


//! Sort the objects by their Hilbert indices.
/*!
  \param objects The vector of objects.

  Note that the number type for the code must be specified explicitly. This 
  function is used to improve locality of reference. Thus, the spatial 
  correlation between successive objects does not need to be all that high.
  A 32-bit number type is probably sufficient for most applications.
*/
template<typename _Code, typename _Object>
void
sortByHilbertCodes(std::vector<_Object>* objects);


} // namespace sfc
} // namespace stlib

#define __sfc_sortByHilbertCodes_tcc__
#include "stlib/sfc/sortByHilbertCodes.tcc"
#undef __sfc_sortByHilbertCodes_tcc__

#endif
