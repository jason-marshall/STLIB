// -*- C++ -*-

/*!
  \file numerical/constants/Exponentiation.h
  \brief Statically exponentiate for non-negative, integer exponents.
*/

#if !defined(__numerical_constants_Exponentiation_h__)
#define __numerical_constants_Exponentiation_h__

#include <boost/config.hpp>

namespace stlib
{
namespace numerical
{

//! Statically compute Base^Exponent for integer arguments.
/*!
  The general case class uses the recursion \f$B^E = B * B^{E-1}\f$.
  The limiting and special cases are defined below.
*/
template<typename _Integer, _Integer Base, _Integer Exponent>
struct Exponentiation {
  BOOST_STATIC_CONSTEXPR _Integer Result = Base *
    Exponentiation<_Integer, Base, Exponent - 1>::Result;
};

} // namespace numerical
}

#define __numerical_constants_Exponentiation_ipp__
#include "stlib/numerical/constants/Exponentiation.ipp"
#undef __numerical_constants_Exponentiation_ipp__

#endif
