// -*- C++ -*-

/*!
  \file countGroups.h
  \brief Count the number of groups of consecutive equivalent elements.
*/

#if !defined(__ads_algorithm_countGroups_h__)
#define __ads_algorithm_countGroups_h__

#include <algorithm>
#include <functional>
#include <iterator>

namespace stlib
{
namespace ads
{

//-----------------------------------------------------------------------------
/*! \defgroup algorithm_countGroups Algorithm: CountGroups Elements */
// @{

//! Count the number of groups of consecutive equivalent elements.
/*! The elements must be equality comparable. */
template<typename _ForwardIterator>
std::size_t
countGroups(_ForwardIterator first, _ForwardIterator last);

//! Count the number of groups of consecutive equivalent elements.
/*! The elements must be equality comparable. */
template<typename _ForwardIterator, typename _BinaryPredicate>
std::size_t
countGroups(_ForwardIterator first, _ForwardIterator last,
            _BinaryPredicate equal);

// @}

} // namespace ads
} // namespace stlib

#define __ads_algorithm_countGroups_ipp__
#include "stlib/ads/algorithm/countGroups.ipp"
#undef __ads_algorithm_countGroups_ipp__

#endif
