// -*- C++ -*-

#if !defined(__sfc_Codes_h__)
#define __sfc_Codes_h__

/*!
  \file
  \brief Common functionality for codes.
*/


#include <algorithm>
#include <vector>

#include <cassert>

namespace stlib
{
namespace sfc
{


//! Check the validity of the sequence of object codes.
/*!
  \relates BlockCode
  \relates LocationCode

  \param order Data structure for manipulating codes.
  \param codes The sorted sequence of object codes.

  The individual codes must be valid and the sequence must be sorted.
  Note that object codes need not be distinct.
*/
template<template<typename> class _Order, typename _Traits>
void
checkValidityOfObjectCodes
(_Order<_Traits> const& order,
 std::vector<typename _Traits::Code> const& codes);


//! Check the validity of the sequence of cell codes.
/*!
  \relates BlockCode
  \relates LocationCode

  \param order Data structure for manipulating codes.
  \param codes The sorted sequence of codes.

  The individual codes must be valid. As a sequence, they must be strictly
  increasing and terminated with the guard code.
*/
template<template<typename> class _Order, typename _Traits>
void
checkValidityOfCellCodes
(_Order<_Traits> const& order,
 std::vector<typename _Traits::Code> const& codes);


//! Check the validity of the codes for the sequence of cell code/value pairs.
/*!
  \relates BlockCode
  \relates LocationCode

  \param order Data structure for manipulating codes.
  \param pairs The sorted sequence of code/value pairs.

  The individual codes must be valid. As a sequence, they must be strictly
  increasing and terminated with the guard code.
*/
template<template<typename> class _Order, typename _Traits, typename _T>
void
checkValidityOfCellCodes
(_Order<_Traits> const& order,
 std::vector<std::pair<typename _Traits::Code, _T> > const& pairs);


//! Check that the sorted codes are non-overlapping.
/*!
  \relates BlockCode
  \relates LocationCode

  \param blockCode Data structure for manipulating codes.
  \param codes The sorted sequence of codes.
*/
template<template<typename> class _Order, typename _Traits>
void
checkNonOverlapping
(_Order<_Traits> const& order,
 std::vector<typename _Traits::Code> const& codes);


//! Check that the sorted codes are non-overlapping.
/*!
  \relates BlockCode
  \relates LocationCode

  \param blockCode Data structure for manipulating codes.
  \param codes The sorted sequence of codes.
*/
template<template<typename> class _Order, typename _Traits, typename _T>
void
checkNonOverlapping
(_Order<_Traits> const& order,
 std::vector<std::pair<typename _Traits::Code, _T> > const& pairs);


} // namespace sfc
} // namespace stlib

#define __sfc_Codes_tcc__
#include "stlib/sfc/Codes.tcc"
#undef __sfc_Codes_tcc__

#endif
