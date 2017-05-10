// -*- C++ -*-

/*!
  \file cuda/array.h
  \brief Implementation of std::array for cuda.
*/

#if !defined(__cuda_array_h__)
#define __cuda_array_h__

//#include <iterator>
#include <cstddef>

namespace std
{
namespace tr1
{

template<typename T, std::size_t N>
struct array {
  typedef T value_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef value_type* iterator;
  typedef const value_type* const_iterator;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
#if 0
  // CONTINUE: I can enable these if I write a CUDA replacement.
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
#endif

  // Don't use the aligned attribute, which would cause 16 byte alignment.
  //value_type _data[N ? N : 1] __attribute__((__aligned__));
  value_type _data[N ? N : 1];

  // No explicit construct/copy/destroy for aggregate type.

  void
  __device__
  __host__
  assign(const value_type& x)
  {
    for (std::size_t i = 0; i != size(); ++i) {
      _data[i] = x;
    }
  }

  void
  __device__
  __host__
  swap(array& other)
  {
    value_type tmp;
    for (std::size_t i = 0; i != size(); ++i) {
      tmp = _data[i];
      _data[i] = other._data[i];
      other._data[i] = tmp;
    }
  }

  // Iterators.
  iterator
  __device__
  __host__
  begin()
  {
    return iterator(&_data[0]);
  }

  const_iterator
  __device__
  __host__
  begin() const
  {
    return const_iterator(&_data[0]);
  }

  iterator
  __device__
  __host__
  end()
  {
    return iterator(&_data[N]);
  }

  const_iterator
  __device__
  __host__
  end() const
  {
    return const_iterator(&_data[N]);
  }

#if 0
  reverse_iterator
  __device__
  __host__
  rbegin()
  {
    return reverse_iterator(end());
  }

  const_reverse_iterator
  __device__
  __host__
  rbegin() const
  {
    return const_reverse_iterator(end());
  }

  reverse_iterator
  __device__
  __host__
  rend()
  {
    return reverse_iterator(begin());
  }

  const_reverse_iterator
  __device__
  __host__
  rend() const
  {
    return const_reverse_iterator(begin());
  }
#endif

  // Capacity.
  size_type
  __device__
  __host__
  size() const
  {
    return N;
  }

  size_type
  __device__
  __host__
  max_size() const
  {
    return N;
  }

  bool
  __device__
  __host__
  empty() const
  {
    return size() == 0;
  }

  // Element access.
  reference
  __device__
  __host__
  operator[](const size_type n)
  {
    return _data[n];
  }

  const_reference
  __device__
  __host__
  operator[](const size_type n) const
  {
    return _data[n];
  }

  reference
  __device__
  __host__
  at(const size_type n)
  {
    return _data[n];
  }

  const_reference
  __device__
  __host__
  at(const size_type n) const
  {
    return _data[n];
  }

  reference
  __device__
  __host__
  front()
  {
    return *begin();
  }

  const_reference
  __device__
  __host__
  front() const
  {
    return *begin();
  }

  reference
  __device__
  __host__
  back()
  {
    return N ? *(end() - 1) : *end();
  }

  const_reference
  __device__
  __host__
  back() const
  {
    return N ? *(end() - 1) : *end();
  }

  T*
  __device__
  __host__
  data()
  {
    return &_data[0];
  }

  const T*
  __device__
  __host__
  data() const
  {
    return &_data[0];
  }
};

} // namespace tr1
} // namespace std

#endif
