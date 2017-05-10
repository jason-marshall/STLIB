// -*- C++ -*-

/*!
  \file ads/algorithm/extremeElement.h
  \brief ExtremeElement functions.
*/

#if !defined(__ads_algorithm_extremeElement_h__)
#define __ads_algorithm_extremeElement_h__

#include <iterator>
#include <functional>

#include <cassert>

namespace stlib
{
//! All classes and functions in the ADS package are defined in the ads namespace.
namespace ads
{

//-----------------------------------------------------------------------------
/*! \defgroup algorithm_extremeElement Algorithm: extreme element functions */
// @{

//! Return the minimum element in a range of even length.
template<typename _RandomAccessIterator>
_RandomAccessIterator
findMinimumElementUnrolledEven(_RandomAccessIterator begin,
                               _RandomAccessIterator end);

//! Return the minimum element in a range of even length.
template<typename _RandomAccessIterator>
_RandomAccessIterator
findMinimumElementUnrolledOdd(_RandomAccessIterator begin,
                              _RandomAccessIterator end);

//! Return the minimum element in a range.
template<typename _RandomAccessIterator>
_RandomAccessIterator
findMinimumElementUnrolled(_RandomAccessIterator begin,
                           _RandomAccessIterator end);

//! Return the maximum element in a range of even length.
template<typename _RandomAccessIterator>
_RandomAccessIterator
findMaximumElementUnrolledEven(_RandomAccessIterator begin,
                               _RandomAccessIterator end);

//! Return the maximum element in a range of even length.
template<typename _RandomAccessIterator>
_RandomAccessIterator
findMaximumElementUnrolledOdd(_RandomAccessIterator begin,
                              _RandomAccessIterator end);

//! Return the maximum element in a range.
template<typename _RandomAccessIterator>
_RandomAccessIterator
findMaximumElementUnrolled(_RandomAccessIterator begin,
                           _RandomAccessIterator end);

//! Return the extreme element in a range of even length.
template<typename _RandomAccessIterator, typename _BinaryPredicate>
_RandomAccessIterator
findExtremeElementUnrolledEven(_RandomAccessIterator begin,
                               _RandomAccessIterator end,
                               _BinaryPredicate compare);

//! Return the extreme element in a range of even length.
template<typename _RandomAccessIterator, typename _BinaryPredicate>
_RandomAccessIterator
findExtremeElementUnrolledOdd(_RandomAccessIterator begin,
                              _RandomAccessIterator end,
                              _BinaryPredicate compare);

//! Return the extreme element in a range.
template<typename _RandomAccessIterator, typename _BinaryPredicate>
_RandomAccessIterator
findExtremeElementUnrolled(_RandomAccessIterator begin,
                           _RandomAccessIterator end, _BinaryPredicate compare);

// @}

} // namespace ads
} // namespace stlib

#define __ads_algorithm_extremeElement_ipp__
#include "stlib/ads/algorithm/extremeElement.ipp"
#undef __ads_algorithm_extremeElement_ipp__

#endif
