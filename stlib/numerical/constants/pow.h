// -*- C++ -*-

/*!
  \file numerical/constants/pow.h
  \brief Compile-time const pow(x, n).
*/

#if !defined(__numerical_constants_pow_h__)
#define __numerical_constants_pow_h__

#include <boost/config.hpp>

namespace stlib
{
namespace numerical
{

//! Statically compute x^n for non-negative integer exponents.
/*! \note We do not check for the indeterminate form 0^0. Thus, evaluating
  pow(0, 0) will give the incorrect result, 1. */
template<typename _T>
inline
BOOST_CONSTEXPR
_T
pow(_T const x, std::size_t const n) BOOST_NOEXCEPT
{
  return n == 0 ? 1 : x * pow(x, n - 1);
}

} // namespace numerical
} // namespace stlib

#endif
