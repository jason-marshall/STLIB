// -*- C++ -*-

/*!
  \file numerical/round/floor.h
  \brief The floor function.
*/

#if !defined(__numerical_round_floor_h__)
#define __numerical_round_floor_h__

#include <cmath>

namespace stlib
{
namespace numerical
{

template<typename _Integer, typename _Argument>
inline
_Integer
floorNonNegative(const _Argument x)
{
  return _Integer(x);
}

template<>
inline
float
floorNonNegative(const float x)
{
  return std::floor(x);
}

template<>
inline
double
floorNonNegative(const double x)
{
  return std::floor(x);
}

template<typename _Integer, typename _Argument>
inline
_Integer
floor(const _Argument x)
{
  const _Integer y = static_cast<_Integer>(x);
  if (x >= 0) {
    return y;
  }
  return (x == y ? y : y - 1);
}

template<>
inline
float
floor(const float x)
{
  return std::floor(x);
}

template<>
inline
double
floor(const double x)
{
  return std::floor(x);
}

} // namespace numerical
}

#endif
