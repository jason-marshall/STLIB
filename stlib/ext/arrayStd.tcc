// -*- C++ -*-

#if !defined(__stlib_ext_arrayStd_tcc__)
#error This file is an implementation detail of arrayStd.
#endif

namespace stlib
{
namespace ext
{

//
// Array-scalar arithmetic assignment operators.
//

template<typename T, std::size_t N>
inline
std::array<T, N>&
operator+=(std::array<T, N>& x, T const& value)
{
  for (typename std::array<T, N>::iterator i = x.begin(); i != x.end(); ++i) {
    *i += value;
  }
  return x;
}

template<typename T, std::size_t N>
inline
std::array<T, N>&
operator-=(std::array<T, N>& x, T const& value)
{
  for (typename std::array<T, N>::iterator i = x.begin(); i != x.end(); ++i) {
    *i -= value;
  }
  return x;
}

template<typename T, std::size_t N>
inline
std::array<T, N>&
operator*=(std::array<T, N>& x, T const& value)
{
  for (typename std::array<T, N>::iterator i = x.begin(); i != x.end(); ++i) {
    *i *= value;
  }
  return x;
}

template<typename T, std::size_t N>
inline
std::array<T, N>&
operator/=(std::array<T, N>& x, T const& value)
{
  for (typename std::array<T, N>::iterator i = x.begin(); i != x.end(); ++i) {
    *i /= value;
  }
  return x;
}

template<typename T, std::size_t N>
inline
std::array<T, N>&
operator%=(std::array<T, N>& x, T const& value)
{
  for (typename std::array<T, N>::iterator i = x.begin(); i != x.end(); ++i) {
    *i %= value;
  }
  return x;
}

template<typename T, std::size_t N>
inline
std::array<T, N>&
operator<<=(std::array<T, N>& x, int const offset)
{
  for (typename std::array<T, N>::iterator i = x.begin(); i != x.end(); ++i) {
    *i <<= offset;
  }
  return x;
}

template<typename T, std::size_t N>
inline
std::array<T, N>&
operator>>=(std::array<T, N>& x, int const offset)
{
  for (typename std::array<T, N>::iterator i = x.begin(); i != x.end(); ++i) {
    *i >>= offset;
  }
  return x;
}


} // namespace ext
} // namespace stlib
