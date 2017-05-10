// -*- C++ -*-

#if !defined(__numerical_specialFunctions_LogarithmOfFactorialCached_ipp__)
#error This file is an implementation detail of LogarithmOfFactorialCached.
#endif

namespace stlib
{
namespace numerical
{


template<typename T>
inline
LogarithmOfFactorialCached<T>::
LogarithmOfFactorialCached(const std::size_t size) :
  _values(size)
{
  // First fill the table with log(n).
  if (size != 0) {
    _values[0] = 0;
  }
  for (std::size_t i = 1; i < size; ++i) {
    _values[i] = std::log(Number(i));
  }
  // Next use partial sums to get log(n!).
  std::partial_sum(_values.begin(), _values.end(), _values.begin());
}


} // namespace numerical
}
