// -*- C++ -*-

/*!
  \file numerical/round/round.h
  \brief The round function.
*/

#if !defined(__numerical_round_round_h__)
#define __numerical_round_round_h__

#include <cmath>

namespace stlib
{
namespace numerical
{

template<typename _Integer, typename _Argument>
inline
_Integer
roundNonNegative(const _Argument x)
{
  return _Integer(x + 0.5);
}

template<>
inline
float
roundNonNegative(const float x)
{
  return std::floor(x + 0.5);
}

template<>
inline
double
roundNonNegative(const double x)
{
  return std::floor(x + 0.5);
}

template<typename _Integer, typename _Argument>
inline
_Integer
round(const _Argument x)
{
  if (x >= 0) {
    return _Integer(x + 0.5);
  }
  return _Integer(x - 0.5);
}

template<>
inline
float
round(const float x)
{
  return std::floor(x + 0.5);
}

template<>
inline
double
round(const double x)
{
  return std::floor(x + 0.5);
}

} // namespace numerical
}

#endif
