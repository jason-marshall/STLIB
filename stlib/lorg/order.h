// -*- C++ -*-

/*!
  \file lorg/order.h
  \brief Determine the order given the positions.
*/

#if !defined(__lorg_order_h__)
#define __lorg_order_h__

#include "stlib/lorg/codes.h"
#include "stlib/lorg/sort.h"

#include <algorithm>

namespace stlib
{
namespace lorg
{

//! Sort according to the code values.
template<typename _Integer>
void
sort(std::vector<std::pair<_Integer, std::size_t> >* pairs,
     int digits = sizeof(_Integer) * 8);
// Note that MSVC does not define std::numeric_limits<_Integer>::digits
// for unsigned char. Thus, I need to use the formula above.

//! Order according to the code values.
template<typename _Integer>
void
codeOrder(const std::vector<_Integer>& codes, std::vector<std::size_t>* ranked);

//! Order according to the code values.
template<typename _Integer>
void
codeOrder(const std::vector<_Integer>& codes, std::vector<std::size_t>* ranked,
          std::vector<std::size_t>* mapping);

//! Determine a random ordering for \c size objects.
void
randomOrder(std::size_t size, std::vector<std::size_t>* ranked);

//! Determine a random ordering for \c size objects.
void
randomOrder(std::size_t size, std::vector<std::size_t>* ranked,
            std::vector<std::size_t>* mapping);

//! Determine an order with good locality of reference.
template<typename _Integer, typename _Float, std::size_t _Dimension>
void
mortonOrder(const std::vector<std::array<_Float, _Dimension> >& positions,
            std::vector<std::size_t>* ranked);

//! Determine an order with good locality of reference.
template<typename _Integer, typename _Float, std::size_t _Dimension>
void
mortonOrder(const std::vector<std::array<_Float, _Dimension> >& positions,
            std::vector<std::size_t>* ranked,
            std::vector<std::size_t>* mapping);

//! Sort along the axis of greatest extent.
template<typename _Float, std::size_t _Dimension>
void
axisOrder(const std::vector<std::array<_Float, _Dimension> >& positions,
          std::vector<std::size_t>* ranked);

//! Sort along the axis of greatest extent.
template<typename _Float, std::size_t _Dimension>
void
axisOrder(const std::vector<std::array<_Float, _Dimension> >& positions,
          std::vector<std::size_t>* ranked,
          std::vector<std::size_t>* mapping);

} // namespace lorg
}

#define __lorg_order_tcc__
#include "stlib/lorg/order.tcc"
#undef __lorg_order_tcc__

#endif
