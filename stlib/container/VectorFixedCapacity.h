// -*- C++ -*-

/*!
  \file VectorFixedCapacity.h
  \brief A class for a resizable vector with fixed capacity.
*/

#if !defined(__container_VectorFixedCapacity_h__)
#define __container_VectorFixedCapacity_h__

#include <boost/config.hpp>

#include <array>

#include <cassert>

namespace stlib
{
namespace container
{


/// A resizable vector with a fixed capacity.
/**
   The boost::container::static_vector class provides a vector with a
   fixed capacity. That would be a better choice for general
   purposes. The reason that I wrote this class is to provide such a
   vector with the most efficient push_back() and pop_back()
   operations. They are efficient because, unless STLIB_DEBUG is
   defined, the validity of the operations is not checked. That is,
   executing push_back() on a full vector or pop_back() on an empty
   vector both result in undefined behavior. If you were to use
   boost::container::static_vector, such calls would result in an
   exception being thrown.
*/
template<typename _T, std::size_t _Capacity>
class VectorFixedCapacity
{
public:

  typedef _T value_type;
  typedef value_type* pointer;
  typedef value_type const* const_pointer;
  typedef value_type& reference;
  typedef value_type const& const_reference;
  typedef value_type* iterator;
  typedef value_type const* const_iterator;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;


private:

  /// The number of elements in the vector.
  size_type _size;
  /// The array for the elements.
  std::array<_T, _Capacity> _data;

  template<typename _Integer>
  void
  _initializeIntegerOrIterator(_Integer size, _Integer value, std::true_type)
  {
    _size = size;
    fill(value);
  }

  template<typename _InputIterator>
  void
  _initializeIntegerOrIterator(_InputIterator begin, _InputIterator end,
                               std::false_type)
  {
    _size = 0;
    while (begin != end) {
      push_back(*begin++);
    }
  }

public:

  //
  // Constructors, etc.
  //

  /// The default constructor results in an empty vector.
  VectorFixedCapacity() :
    _size(0)
  {
  }

  /// Creates a vector with default constructed elements.
  /**
     \param n The number of elements to initially create.

     This constructor fills the vector with \a n default constructed elements.
  */
  explicit
  VectorFixedCapacity(size_type n) :
    _size(n)
  {
    assert(n <= _Capacity);
    std::fill_n(data(), n, value_type());
  }

  /// Creates a vector with copies of an exemplar element.
  /**
     \param n The number of elements to initially create.
     \param value An element to copy.

     This constructor fills the vector with \a n copies of \a value.
  */
  explicit
  VectorFixedCapacity(size_type n, value_type const& value) :
    _size(n)
  {
    assert(n <= _Capacity);
    std::fill_n(data(), n, value);
  }

  VectorFixedCapacity(VectorFixedCapacity const& other) :
    _size(other._size),
    _data(other._data)
  {
  }

  // There's no need to define a move constructor because it would not differ
  // from the copy constructor.

  /// Construct from an initializer list.
  VectorFixedCapacity(std::initializer_list<value_type> il) :
    _size(il.size())
  {
    std::copy(il.begin(), il.end(), data());
  }

  template<typename _InputIterator>
  VectorFixedCapacity(_InputIterator begin, _InputIterator end)
  {
    _initializeIntegerOrIterator
      (begin, end, typename std::is_integral<_InputIterator>::type{});
  }

  // The synthesized destructor is fine.

  VectorFixedCapacity&
  operator=(VectorFixedCapacity const& other)
  {
    if (&other != this) {
      _size = other._size;
      std::copy(other.begin(), other.end(), begin());
    }
    return *this;
  }

  // There is no need for a move assignment operator because it would not 
  // differ from the copy assignment operator.

  VectorFixedCapacity&
  operator=(std::initializer_list<value_type> il)
  {
    _size = il.size();
    std::copy(il.begin(), il.end(), data());
    return *this;
  }

  // CONTINUE: Assign.

  //
  // Modifiers.
  //

  void
  fill(value_type const& x)
  {
    std::fill_n(begin(), size(), x);
  }

  // Note that in the following the outer noexcept is a specifier while the 
  // inner one is an operator that returns either true or false.
  void
  swap(VectorFixedCapacity& other)
    noexcept(noexcept(std::swap(std::declval<_T&>(), std::declval<_T&>())))
  {
    std::swap(_size, other._size);
    _data.swap(other._data);
  }

  void
  resize(size_type newSize)
  {
    resize(newSize, value_type());
  }

  void
  resize(size_type newSize, value_type const& value)
  {
    if (newSize > size()) {
      assert(newSize <= _Capacity);
      std::fill_n(end(), newSize - size(), value);
    }
    _size = newSize;
  }

  void
  push_back(value_type const& x) noexcept
  {
#ifdef STLIB_DEBUG
    assert(size() != _Capacity);
#endif
    _data[_size++] = x;
  }

  void
  pop_back() noexcept
  {
#ifdef STLIB_DEBUG
    assert(! empty());
#endif
    --_size;
  }

  // CONTINUE insert() and erase().

  //
  // Accessors.
  //

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
    return iterator(data() + size());
  }

  const_iterator
  end() const noexcept
  {
    return const_iterator(data() + size());
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
    return const_iterator(data() + size());
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


  size_type 
  size() const noexcept
  {
    return _size;
  }

  constexpr size_type 
  max_size() const noexcept
  {
    return _Capacity;
  }

  bool 
  empty() const noexcept
  {
    return size() == 0;
  }


  reference
  operator[](size_type n) noexcept
  {
    return _data[n];
  }

  constexpr const_reference
  operator[](size_type n) const noexcept
  {
    return _data[n];
  }

  reference
  at(size_type n)
  {
    if (n >= size()) {
      throw std::out_of_range("VectorFixedCapacity::at: index out of range.");
    }
    return _data[n];
  }

  constexpr const_reference
  at(size_type n) const
  {
    // Result of conditional expression must be an lvalue so use
    // boolean ? lvalue : (throw-expr, lvalue)
    return n < size() ? _data[n] :
      (std::out_of_range("VectorFixedCapacity::at: index out of range."),
       _data[0]);
  }

  reference 
  front() noexcept
  {
    return *begin();
  }

  constexpr const_reference 
  front() const noexcept
  {
    return _data[0];
  }

  reference 
  back() noexcept
  {
    return _Capacity ? *(end() - 1) : *end();
  }

  const_reference 
  back() const noexcept
  { 
    return _Capacity ? _data[size() - 1] : _data[0];
  }

  pointer
  data() noexcept
  {
    return _data.data();
  }

  const_pointer
  data() const noexcept
  {
    return _data.data();
  }
};


template<typename _T, std::size_t _Capacity>
inline bool
operator==(VectorFixedCapacity<_T, _Capacity> const& a,
           VectorFixedCapacity<_T, _Capacity> const& b)
{
  return std::equal(a.begin(), a.end(), b.begin());
}

template<typename _T, std::size_t _Capacity>
inline bool
operator!=(VectorFixedCapacity<_T, _Capacity> const& a,
           VectorFixedCapacity<_T, _Capacity> const& b)
{
  return !(a == b);
}

template<typename _T, std::size_t _Capacity>
inline bool
operator<(VectorFixedCapacity<_T, _Capacity> const& a,
          VectorFixedCapacity<_T, _Capacity> const& b)
{ 
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); 
}

template<typename _T, std::size_t _Capacity>
inline bool
operator>(VectorFixedCapacity<_T, _Capacity> const& a,
          VectorFixedCapacity<_T, _Capacity> const& b)
{
  return b < a;
}

template<typename _T, std::size_t _Capacity>
inline bool
operator<=(VectorFixedCapacity<_T, _Capacity> const& a,
           VectorFixedCapacity<_T, _Capacity> const& b)
{
  return !(a > b);
}

template<typename _T, std::size_t _Capacity>
inline bool
operator>=(VectorFixedCapacity<_T, _Capacity> const& a,
           VectorFixedCapacity<_T, _Capacity> const& b)
{
  return !(a < b);
}


template<typename _T, std::size_t _Capacity>
inline void
swap(VectorFixedCapacity<_T, _Capacity>& a,
     VectorFixedCapacity<_T, _Capacity>& b)
  noexcept(noexcept(a.swap(b)))
{
  a.swap(b);
}

template<std::size_t _Int, typename _T, std::size_t _Capacity>
constexpr _T&
get(VectorFixedCapacity<_T, _Capacity>& a) noexcept
{
  static_assert(_Int < _Capacity, "index is out of bounds");
  return a[_Int];
}

template<std::size_t _Int, typename _T, std::size_t _Capacity>
constexpr _T&&
get(VectorFixedCapacity<_T, _Capacity>&& a) noexcept
{
  static_assert(_Int < _Capacity, "index is out of bounds");
  return std::move(get<_Int>(a));
}

template<std::size_t _Int, typename _T, std::size_t _Capacity>
constexpr const _T&
get(VectorFixedCapacity<_T, _Capacity> const& a) noexcept
{
  static_assert(_Int < _Capacity, "index is out of bounds");
  return a[_Int];
}


} // namespace container
} // namespace stlib

#endif
