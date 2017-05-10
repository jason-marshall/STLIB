// -*- C++ -*-

/*!
  \file numerical/constants/Logarithm.h
  \brief Statically evaluate the (ceiling of) the logorithm.
*/

#if !defined(__numerical_constants_Logarithm_h__)
#define __numerical_constants_Logarithm_h__

#include <boost/config.hpp>

namespace stlib
{
namespace numerical
{

//! Statically compute the (ceiling of the) logarithm with the given base.
/*!
  The general case class uses the recursion
  \f$\lceil log_B(A) \rceil = \lceil log_B((A+B-1)/B) \rceil + 1\f$.
  The limiting and special cases are defined below.
*/
template<typename _Integer, _Integer Base, _Integer Argument>
class Logarithm
{
public:
  BOOST_STATIC_CONSTEXPR _Integer Result =
    Logarithm < _Integer, Base, (Argument + Base - 1) / Base >::Result + 1;
};

} // namespace numerical
}

#define __numerical_constants_Logarithm_ipp__
#include "stlib/numerical/constants/Logarithm.ipp"
#undef __numerical_constants_Logarithm_ipp__

#endif
