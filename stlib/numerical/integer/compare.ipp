// -*- C++ -*-

#ifndef __numerical_integer_compare_ipp__
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace numerical
{


template<typename _T>
inline
bool
_isNonNegative(const _T /*x*/, std::false_type /*isSigned*/)
{
  return true;
}


template<typename _T>
inline
bool
_isNonNegative(const _T x, std::true_type /*isSigned*/)
{
  return x >= 0;
}


template<typename _T>
inline
bool
isNonNegative(const _T x)
{
  return _isNonNegative(x, std::integral_constant<bool,
                        std::numeric_limits<_T>::is_signed>());
}


} // namespace numerical
}
