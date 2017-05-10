// -*- C++ -*-

#if !defined(__sfc_sortByCodes_h__)
#define __sfc_sortByCodes_h__

/*!
  \file
  \brief Sort objects by their spatial indexing codes.
*/

#include "stlib/sfc/LocationCode.h"
#include "stlib/sfc/OrderedObjects.h"
#include "stlib/lorg/order.h"

#include <utility>

namespace stlib
{
namespace sfc
{


//! Sort the objects by their spatial index codes.
/*!
  \param order The data structure for calculating the codes.
  \param objects The vector of objects.
  \param codeIndexPairs This output vector records the codes and original 
  indices for the objects. This can be used to restore the original order.
*/
template<typename _Grid, typename _Object>
void
sortByCodes(_Grid const& order, std::vector<_Object>* objects,
            std::vector<std::pair<typename _Grid::Code,
            std::size_t> >* codeIndexPairs);


//! Sort the objects by their spatial index codes.
/*!
  \param order The data structure for calculating the codes.
  \param objects The vector of objects.
  \param codes This output vector records the codes.
*/
template<typename _Grid, typename _Object>
void
sortByCodes(_Grid const& order, std::vector<_Object>* objects,
            std::vector<typename _Grid::Code>* codes);


//! Sort the objects by their spatial index codes.
/*!
  \param order The data structure for calculating the codes.
  \param objects The vector of objects.
*/
template<typename _Grid, typename _Object>
void
sortByCodes(_Grid const& order, std::vector<_Object>* objects);


//! Sort the objects by their spatial index codes.
/*!
  \param objects The vector of objects.
  \param orderedObjects If specified, record the original order of the objects.

  Note that the number type for the code must be specified explicitly. This 
  function is used to improve locality of reference. Thus, the spatial 
  correlation between successive objects does not need to be all that high.
  A 32-bit number type is probably sufficient for most applications.
*/
template<typename _Code, typename _Object>
void
sortByCodes(std::vector<_Object>* objects,
            OrderedObjects* orderedObjects = nullptr);


} // namespace sfc
} // namespace stlib

#define __sfc_sortByCodes_tcc__
#include "stlib/sfc/sortByCodes.tcc"
#undef __sfc_sortByCodes_tcc__

#endif
