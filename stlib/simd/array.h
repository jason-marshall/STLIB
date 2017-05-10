// -*- C++ -*-

#ifndef stlib_simd_array_h
#define stlib_simd_array_h

#include "stlib/simd/align.h"
#include "stlib/simd/functions.h"

#include <array>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <tuple>

namespace stlib
{
namespace simd
{


/// This class allows a specialization for zero-length arrays.
template<typename _T, std::size_t N>
struct _arrayTraits
{
  /// Define the C array type.
  typedef _T Array[N];

  /// Index a C array.
  static constexpr _T&
  index(Array const& t, std::size_t n) noexcept
  {
    return const_cast<_T&>(t[n]);
  }

  /// Return the address of the first element of an array.
  static constexpr _T*
  data(Array const& t) noexcept
  {
    return const_cast<_T*>(t);
  }
};


/// This is the specialization for zero-length arrays.
template<typename _T>
struct _arrayTraits<_T, 0>
{
  /// A dummy type for a zero-length C array.
  struct Array {};

  /// Indexing a zero-length array will result in a segmentation fault.
  static constexpr _T&
  index(Array const&, std::size_t) noexcept
  {
    return *static_cast<_T*>(nullptr);
  }

  /// Return a null pointer for the beginning of a zero-length array.
  static constexpr _T*
  data(Array const&) noexcept
  {
    return nullptr;
  }
};


/// Aligned array.
/**
  This is the same as std::array, except that the array is aligned. 
  Specifically it is aligned according to Alignment. The size of this data
  structure is also a multiple of stlib::simd::Alignment. Therefore, regardless of the 
  size (number of elements), you can iterate over this array using 
  SIMD vectors. When you load or store the last SIMD vector, you may be
  using storage that is the padding beyond the last array element.
*/
template<typename _T, std::size_t N>
struct array {
  typedef _T value_type;
  typedef value_type& reference;
  typedef value_type const& const_reference;
  typedef value_type* iterator;
  typedef value_type const* const_iterator;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  // Support zero-sized arrays.
  typedef _arrayTraits<_T, N> _Traits;
  ALIGN_SIMD typename _Traits::Array _data;

  // No explicit construct/copy/destroy for aggregate type.

  void
  fill(value_type const& x)
  {
    std::fill_n(begin(), size(), x);
  }

  void
  swap(array& other)
    noexcept(noexcept(std::swap(std::declval<_T&>(), std::declval<_T&>())))
  {
    std::swap_ranges(begin(), end(), other.begin());
  }

  // Iterators.
  iterator
  begin() noexcept
  {
    return iterator(data());
  }

  const_iterator
  begin() const noexcept
  {
    return const_iterator(data());
  }

  iterator
  end() noexcept
  {
    return iterator(data() + N);
  }

  const_iterator
  end() const noexcept
  {
    return const_iterator(data() + N);
  }

  reverse_iterator
  rbegin() noexcept
  {
    return reverse_iterator(end());
  }

  const_reverse_iterator
  rbegin() const noexcept
  {
    return const_reverse_iterator(end());
  }

  reverse_iterator
  rend() noexcept
  {
    return reverse_iterator(begin());
  }

  const_reverse_iterator
  rend() const noexcept
  {
    return const_reverse_iterator(begin());
  }

  const_iterator
  cbegin() const noexcept
  {
    return const_iterator(data());
  }

  const_iterator
  cend() const noexcept
  {
    return const_iterator(data() + N);
  }

  const_reverse_iterator
  crbegin() const noexcept
  {
    return const_reverse_iterator(end());
  }

  const_reverse_iterator
  crend() const noexcept
  {
    return const_reverse_iterator(begin());
  }

  // Capacity.
  size_type
  size() const noexcept
  {
    return N;
  }

  size_type
  max_size() const noexcept
  {
    return N;
  }

  bool
  empty() const noexcept
  {
    return size() == 0;
  }

  // Element access.
  reference
  operator[](size_type n) noexcept
  {
    return _Traits::index(_data, n);
  }

  const_reference
  operator[](size_type n) const noexcept
  {
    return _Traits::index(_data, n);
  }

  reference
  at(size_type n)
  {
    if (n >= N) {
      throw std::out_of_range("simd::array::at: index out of range.");
    }
    return operator[](n);
  }

  const_reference
  at(size_type n) const
  {
    if (n >= N) {
      throw std::out_of_range("simd::array::at: index out of range.");
    }
    return operator[](n);
  }

  reference
  front() noexcept
  {
    return *begin();
  }

  const_reference
  front() const noexcept
  {
    return *begin();
  }

  reference
  back() noexcept
  {
    return N ? *(end() - 1) : *end();
  }

  const_reference
  back() const noexcept
  {
    return N ? *(end() - 1) : *end();
  }

  _T*
  data() noexcept
  {
    return _Traits::data(_data);
  }

  const _T*
  data() const noexcept
  {
    return _Traits::data(_data);
  }
};

// Array comparisons.
template<typename _T, std::size_t N>
inline
bool
operator==(array<_T, N> const& x, array<_T, N> const& y)
{
  return std::equal(x.begin(), x.end(), y.begin());
}

template<typename _T, std::size_t N>
inline
bool
operator!=(array<_T, N> const& x, array<_T, N> const& y)
{
  return !(x == y);
}

template<typename _T, std::size_t N>
inline
bool
operator<(array<_T, N> const& x, array<_T, N> const& y)
{
  return std::lexicographical_compare(x.begin(), x.end(),
                                      y.begin(), y.end());
}

template<typename _T, std::size_t N>
inline
bool
operator>(array<_T, N> const& x, array<_T, N> const& y)
{
  return y < x;
}

template<typename _T, std::size_t N>
inline
bool
operator<=(array<_T, N> const& x, array<_T, N> const& y)
{
  return !(x > y);
}

template<typename _T, std::size_t N>
inline
bool
operator>=(array<_T, N> const& x, array<_T, N> const& y)
{
  return !(x < y);
}

// Specialized algorithms.
template<typename _T, std::size_t N>
inline
void
swap(array<_T, N>& x, array<_T, N>& y)
  noexcept(noexcept(x.swap(y)))
{
  x.swap(y);
}

template<int _Int, typename _T, std::size_t N>
inline
constexpr
_T&
get(array<_T, N>& arr) noexcept
{
  static_assert(_Int < N, "index is out of bounds");
  return arr[_Int];
}

template<int _Int, typename _T, std::size_t N>
inline
constexpr
_T&&
get(array<_T, N>&& arr) noexcept
{
  static_assert(_Int < N, "index is out of bounds");
  return std::move(arr[_Int]);
}

template<int _Int, typename _T, std::size_t N>
inline
constexpr
const _T&
get(array<_T, N> const& arr) noexcept
{
  static_assert(_Int < N, "index is out of bounds");
  return arr[_Int];
}

/// Convert from one type to another.
template<typename _T1, std::size_t N, typename _T2>
inline
void
convert(array<_T1, N> const& input, array<_T2, N>* output) {
  for (std::size_t i = 0; i != N; ++i) {
    (*output)[i] = _T2(input[i]);
  }
}

inline
void
convert(array<double, 2> const& input, array<float, 2>* output) {
#ifdef __SSE2__
  store(&(*output)[0], _mm_cvtpd_ps(_mm_load_pd(&input[0])));
#else
  for (std::size_t i = 0; i != 2; ++i) {
    (*output)[i] = float(input[i]);
  }
#endif
}

inline
void
convert(array<double, 3> const& input, array<float, 3>* output) {
#if defined(__AVX__)
  static_assert(sizeof(array<float, 3>) == sizeof(array<float, 4>),
                "Invalid size.");
  store(&(*output)[0], _mm256_cvtpd_ps(_mm256_load_pd(&input[0])));
#elif defined(__SSE2__)
  static_assert(sizeof(array<float, 3>) == sizeof(array<float, 4>),
                "Invalid size.");
  __m128 const lo = _mm_cvtpd_ps(_mm_load_pd(&input[0]));
  __m128 const hi = _mm_cvtpd_ps(_mm_load_pd(&input[2]));
  store(&(*output)[0], _mm_movelh_ps(lo, hi));
#else
  for (std::size_t i = 0; i != 3; ++i) {
    (*output)[i] = float(input[i]);
  }
#endif
}

inline
void
convert(array<double, 4> const& input, array<float, 4>* output) {
#if defined(__AVX__)
  store(&(*output)[0], _mm256_cvtpd_ps(_mm256_load_pd(&input[0])));
#elif defined(__SSE2__)
  __m128 const lo = _mm_cvtpd_ps(_mm_load_pd(&input[0]));
  __m128 const hi = _mm_cvtpd_ps(_mm_load_pd(&input[2]));
  store(&(*output)[0], _mm_movelh_ps(lo, hi));
#else
  for (std::size_t i = 0; i != 4; ++i) {
    (*output)[i] = float(input[i]);
  }
#endif
}


} // namespace simd
} // namespace stlib


// CONTINUE: This is commented out because some implementations define
// tuple_size and tuple_element as struct's while others define them as
// classes.
#if 0
namespace std
{

// Tuple interface to class template array.

/// @c value records the size of the array.
template<typename _T, std::size_t N>
struct tuple_size<stlib::simd::array<_T, N> > :
    public std::integral_constant<std::size_t, N>
{
};

/// Define the type of the specified element.
template<int _Int, typename _T, std::size_t N>
struct tuple_element<_Int, stlib::simd::array<_T, N> > {
  static_assert(_Int < N, "index is out of bounds");
  typedef _T type;
};

} // namespace std
#endif

#endif // stlib_simd_array_h
