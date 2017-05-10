// -*- C++ -*-

/*!
  \file numerical/round/ceil.h
  \brief The ceil function.
*/

#if !defined(__numerical_round_ceil_h__)
#define __numerical_round_ceil_h__

#include <cmath>

namespace stlib
{
namespace numerical
{

template<typename _Integer, typename _Argument>
inline
_Integer
ceilNonNegative(const _Argument x)
{
  const _Integer y = static_cast<_Integer>(x);
  return (y == x ? y : y + 1);
}

template<>
inline
float
ceilNonNegative(const float x)
{
  return std::ceil(x);
}

template<>
inline
double
ceilNonNegative(const double x)
{
  return std::ceil(x);
}

template<typename _Integer, typename _Argument>
inline
_Integer
ceil(const _Argument x)
{
  const _Integer y = static_cast<_Integer>(x);
  if (x <= 0) {
    return y;
  }
  return (y == x ? y : y + 1);
}

template<>
inline
float
ceil(const float x)
{
  return std::ceil(x);
}

template<>
inline
double
ceil(const double x)
{
  return std::ceil(x);
}

} // namespace numerical
}

#endif
