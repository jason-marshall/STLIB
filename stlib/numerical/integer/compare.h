// -*- C++ -*-

/*!
  \file numerical/integer/compare.h
  \brief Compare integers.
*/

#if !defined(__numerical_integer_compare_h__)
#define __numerical_integer_compare_h__

#include <limits>
#include <type_traits>

namespace stlib
{
namespace numerical
{

//-----------------------------------------------------------------------------
//! \defgroup numerical_integer_compare Compare integers.
//@{

//! Return true if the argument is non-negative.
/*! The implementation is specialized for unsigned types to avoid compiler
  warnings about meaningless comparisons. */
template<typename _T>
bool
isNonNegative(_T x);

//\@}

} // namespace numerical
} // namespace stlib

#define __numerical_integer_compare_ipp__
#include "stlib/numerical/integer/compare.ipp"
#undef __numerical_integer_compare_ipp__

#endif
