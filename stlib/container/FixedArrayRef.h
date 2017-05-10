// -*- C++ -*-

/*!
  \file stlib/container/FixedArrayRef.h
  \brief A fixed-size array that references memory.
*/

#if !defined(__container_FixedArrayRef_h__)
#define __container_FixedArrayRef_h__

#include <algorithm>

namespace stlib
{
namespace container
{

//! A fixed-size array that references memory.
/*!
  This class implements the same interface as std::array. It adds
  the function setData() to reference an externally allocated array.
  This class does not take ownership of the data. It must be deallocated
  only after the instance of this class is destroyed.
*/
template<typename _T, std::size_t N>
class FixedArrayRef
{
  // Types.
public:
  typedef _T value_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef value_type* iterator;
  typedef const value_type* const_iterator;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  // Member data.
private:
  value_type* _data;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default destructor as this class does not manage the array
    memory.
  */
  //@{
public:

  //! Default constructor. Invalid data pointer.
  FixedArrayRef() :
    _data(0)
  {
  }

  //! Construct from a pointer to the array data.
  FixedArrayRef(iterator data) :
    _data(data)
  {
  }

  //! Copy constructor. Reference the same array data.
  FixedArrayRef(const FixedArrayRef& other) :
    _data(other._data)
  {
  }

  //! Assignment operator. Copy the elements.
  FixedArrayRef&
  operator=(const FixedArrayRef& other)
  {
    if (&other != this) {
#ifdef STLIB_DEBUG
      assert(_data);
#endif
      std::copy(other._data, other._data + N, _data);
    }
    return *this;
  }

  //! Reference an externally allocated array.
  /*! This function must be called before accessing elements of the array. */
  void
  setData(iterator data)
  {
    _data = data;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name array interface.
  //@{

  void
  assign(const_reference x)
  {
    std::fill_n(begin(), size(), x);
  }

  void
  swap(FixedArrayRef& other)
  {
    std::swap(_data, other._data);
  }

  // Iterators.
  iterator
  begin()
  {
    return _data;
  }

  const_iterator
  begin() const
  {
    return _data;
  }

  iterator
  end()
  {
    return _data + N;
  }

  const_iterator
  end() const
  {
    return _data + N;
  }

  reverse_iterator
  rbegin()
  {
    return reverse_iterator(end());
  }

  const_reverse_iterator
  rbegin() const
  {
    return const_reverse_iterator(end());
  }

  reverse_iterator
  rend()
  {
    return reverse_iterator(begin());
  }

  const_reverse_iterator
  rend() const
  {
    return const_reverse_iterator(begin());
  }

  // Capacity.
  size_type
  size() const
  {
    return N;
  }

  size_type
  max_size() const
  {
    return N;
  }

  bool
  empty() const
  {
    return size() == 0;
  }

  // Element access.
  reference
  operator[](size_type n)
  {
    return _data[n];
  }

  const_reference
  operator[](size_type n) const
  {
    return _data[n];
  }

  reference
  at(size_type n)
  {
    _check(n);
    return _data[n];
  }

  const_reference
  at(size_type n) const
  {
    _check(n);
    return _data[n];
  }

  reference
  front()
  {
    return *begin();
  }

  const_reference
  front() const
  {
    return *begin();
  }

  reference
  back()
  {
    return N ? *(end() - 1) : *end();
  }

  const_reference
  back() const
  {
    return N ? *(end() - 1) : *end();
  }

  value_type*
  data()
  {
    return _data;
  }

  const value_type*
  data() const
  {
    return _data;
  }

  //@}

private:

#ifdef STLIB_DEBUG
  void
  _check(const size_type n) const
  {
    assert(n < N);
  }
#else
  void
  _check(const size_type /*n*/) const
  {
  }
#endif

};

//! Equality
template<typename _T, std::size_t N>
inline bool
operator==(const FixedArrayRef<_T, N>& x, const FixedArrayRef<_T, N>& y)
{
  return std::equal(x.begin(), x.end(), y.begin());
}

//! Inequality
template<typename _T, std::size_t N>
inline bool
operator!=(const FixedArrayRef<_T, N>& x, const FixedArrayRef<_T, N>& y)
{
  return !(x == y);
}

//! Less than lexicographical comparison.
template<typename _T, std::size_t N>
inline bool
operator<(const FixedArrayRef<_T, N>& x, const FixedArrayRef<_T, N>& y)
{
  return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}

//! Greater than.
template<typename _T, std::size_t N>
inline bool
operator>(const FixedArrayRef<_T, N>& x, const FixedArrayRef<_T, N>& y)
{
  return y < x;
}

//! Less than or equal to.
template<typename _T, std::size_t N>
inline bool
operator<=(const FixedArrayRef<_T, N>& x, const FixedArrayRef<_T, N>& y)
{
  return !(x > y);
}

//! Greater than or equal to.
template<typename _T, std::size_t N>
inline bool
operator>=(const FixedArrayRef<_T, N>& x, const FixedArrayRef<_T, N>& y)
{
  return !(x < y);
}

// Specialized algorithms [6.2.2.2].
template<typename _T, std::size_t N>
inline void
swap(FixedArrayRef<_T, N>& x, FixedArrayRef<_T, N>& y)
{
  x.swap(y);
}

//! Tuple interface to class template array [6.2.2.5].
template<typename _T> struct tuple_size;
template<int _Int, typename _T> struct tuple_element;

template<typename _T, std::size_t N>
struct tuple_size<FixedArrayRef<_T, N> > {
  BOOST_STATIC_CONSTEXPR int value = N;
};

template<typename _T, std::size_t N>
BOOST_CONSTEXPR int tuple_size<FixedArrayRef<_T, N> >::value;

template<int _Int, typename _T, std::size_t N>
struct tuple_element<_Int, FixedArrayRef<_T, N> > {
  typedef _T type;
};

template<int _Int, typename _T, std::size_t N>
inline _T&
get(FixedArrayRef<_T, N>& x)
{
  return x[_Int];
}

template<int _Int, typename _T, std::size_t N>
inline const _T&
get(const FixedArrayRef<_T, N>& x)
{
  return x[_Int];
}

//! Write the space-separated elements.
/*!
  Format:
  \verbatim
  x[0] x[1] x[2] ... \endverbatim
*/
template<typename _T, size_t N>
inline
std::ostream&
operator<<(std::ostream& out, const FixedArrayRef<_T, N>& x)
{
  std::copy(x.begin(), x.end(), std::ostream_iterator<_T>(out, " "));
  return out;
}

//! Read the elements.
template<typename _T, size_t N>
inline
std::istream&
operator>>(std::istream& in, FixedArrayRef<_T, N>& x)
{
  for (size_t n = 0; n != x.size(); ++n) {
    in >> x[n];
  }
  return in;
}

//! Write the elements in binary format.
template<typename _T, size_t N>
inline
void
write(std::ostream& out, const FixedArrayRef<_T, N>& x)
{
  out.write(reinterpret_cast<const char*>(&x), sizeof(FixedArrayRef<_T, N>));
}

//! Read the elements in binary format.
template<typename _T, size_t N>
inline
void
read(std::istream& in, FixedArrayRef<_T, N>* x)
{
  in.read(reinterpret_cast<char*>(x), sizeof(FixedArrayRef<_T, N>));
}

} // namespace container
}

#endif
