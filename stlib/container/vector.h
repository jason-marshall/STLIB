// -*- C++ -*-

/*!
  \file stlib/container/vector.h
  \brief %vector that allocates its memory and has contiguous storage.
*/

#if !defined(__container_vector_h__)
#define __container_vector_h__

#include <algorithm>
#include <iterator>
#include <limits>

#include <cassert>

namespace stlib
{
namespace container
{

//! Replacement for std::vector.
/*! Using \c std::vector<__m128> is problematic because \c std::vector calls the
  value type's destructor for some operations. \c __m128 is neither a built-in
  type, nor a class; it does not have a default constructor nor a destructor.
  This drop-in replacement for \c std::vector addresses this problem.

  \note This class does not support input iterators. They complicate the
  implementation. */
template<typename _T>
class vector
{

  //
  // Types.
  //
public:

  typedef _T value_type;
  typedef value_type* pointer;
  typedef const value_type* const_pointer;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef pointer iterator;
  typedef const_pointer const_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

  //
  // Member data.
  //
private:

  pointer _start;
  pointer _finish;
  pointer _endOfStorage;

public:

  //! Create an empty vector.
  vector() :
    _start(0),
    _finish(0),
    _endOfStorage(0)
  {
  }

  //! Create an vector with specified size. The elements are not initialized.
  explicit
  vector(const size_type n)
  {
    _allocateFull(n);
  }

  // Enumerate these possibilities so as not to conflict with the range
  // constructor.

  //! Create an vector with specified size. Fill with the specified value.
  explicit
  vector(const unsigned long n, const value_type& value)
  {
    _allocateFull(n);
    std::fill(begin(), end(), value);
  }

  //! Create an vector with specified size. Fill with the specified value.
  explicit
  vector(const unsigned n, const value_type& value)
  {
    _allocateFull(n);
    std::fill(begin(), end(), value);
  }

  //! Create an vector with specified size. Fill with the specified value.
  explicit
  vector(const long n, const value_type& value)
  {
    _allocateFull(n);
    std::fill(begin(), end(), value);
  }

  //! Create an vector with specified size. Fill with the specified value.
  explicit
  vector(const int n, const value_type& value)
  {
    _allocateFull(n);
    std::fill(begin(), end(), value);
  }

  //! Range constructor.
  /*! Note that input iterators are not supported. */
  template<typename _ForwardIterator>
  vector(_ForwardIterator first, _ForwardIterator last)
  {
    _allocateFull(std::distance(first, last));
    std::copy(first, last, begin());
  }

  //! Copy constructor.
  vector(const vector& x)
  {
    _allocateFull(x.size());
    std::copy(x.begin(), x.end(), begin());
  }

  //! Destructor.
  ~vector()
  {
    _destroy();
  }

  //! Assignment operator.
  vector&
  operator=(const vector& x)
  {
    if (&x != this) {
      if (x.size() > capacity()) {
        _destroy();
        _allocateFull(x.size());
      }
      else {
        _finish = _start + x.size();
      }
      std::copy(x.begin(), x.end(), begin());
    }
    return *this;
  }

  //! Resize and assign the value to all elements.
  void
  assign(const size_type n, const value_type& value)
  {
    resize(n);
    std::fill(begin(), end(), value);
  }

  //! Resize and fill the elements with the sequence.
  /*! Note that input iterators are not allowed. */
  template<typename _ForwardIterator>
  void
  assign(_ForwardIterator first, _ForwardIterator last)
  {
    resize(std::distance(first, last));
    std::copy(first, last, begin());
  }

  //! Return an iterator to the first element.
  iterator
  begin()
  {
    return _start;
  }

  //! Return a const iterator to the first element.
  const_iterator
  begin() const
  {
    return _start;
  }

  //! Return an iterator to one past the last element.
  iterator
  end()
  {
    return _finish;
  }

  //! Return a const iterator to one past the last element.
  const_iterator
  end() const
  {
    return _finish;
  }

  //! Return a reverse iterator to the last element.
  reverse_iterator
  rbegin()
  {
    return reverse_iterator(end());
  }

  //! Return a reverse const iterator to the last element.
  const_reverse_iterator
  rbegin() const
  {
    return const_reverse_iterator(end());
  }

  //! Return a reverse iterator to one before the first element.
  reverse_iterator
  rend()
  {
    return reverse_iterator(begin());
  }

  //! Return a reverse const iterator to one before the first element.
  const_reverse_iterator
  rend() const
  {
    return const_reverse_iterator(begin());
  }

  //! Return the number of elements.
  size_type
  size() const
  {
    return size_type(_finish - _start);
  }

  //! Return the largest possible size.
  size_type
  max_size() const
  {
    return std::numeric_limits<size_type>::max();
  }

  //! Resize the vector. The elements have undefined values.
  void
  resize(const size_type n)
  {
    if (n <= size()) {
      _finish = _start + n;
    }
    else {
      _destroy();
      _allocateFull(n);
    }
  }

  //! Resize the vector. Fill with the specified value.
  void
  resize(const size_type n, const value_type& x)
  {
    resize(n);
    std::fill(begin(), end(), x);
  }

  //! Return the maximum size before needing to allocate more memory.
  size_type
  capacity() const
  {
    return size_type(_endOfStorage - _start);
  }

  //! Return true if the vector is empty.
  bool
  empty() const
  {
    return begin() == end();
  }

  //! Pre-allocate memory.
  void
  reserve(size_type n)
  {
    if (n > capacity()) {
      _increaseCapacity(n);
    }
  }

  //! Return a reference to the nth element.
  reference
  operator[](const size_type n)
  {
    return *(_start + n);
  }

  //! Return a const reference to the nth element.
  const_reference
  operator[](const size_type n) const
  {
    return *(_start + n);
  }

  //! Return a reference to the nth element. Check the range.
  reference
  at(const size_type n)
  {
    assert(n < size());
    return (*this)[n];
  }

  //! Return a const reference to the nth element. Check the range.
  /**
   *  @brief  Provides access to the data contained in the %vector.
   *  @param n The index of the element for which data should be
   *  accessed.
   *  @return  Read-only (constant) reference to data.
   *  @throw  std::out_of_range  If @a n is an invalid index.
   *
   *  This function provides for safer data access.  The parameter
   *  is first checked that it is in the range of the vector.  The
   *  function throws out_of_range if the check fails.
   */
  const_reference
  at(const size_type n) const
  {
    assert(n < size());
    return (*this)[n];
  }

  //! Return a reference to the first element.
  reference
  front()
  {
    return *begin();
  }

  //! Return a const reference to the first element.
  const_reference
  front() const
  {
    return *begin();
  }

  //! Return a reference to the last element.
  reference
  back()
  {
    return *(end() - 1);
  }

  //! Return a const reference to the last element.
  const_reference
  back() const
  {
    return *(end() - 1);
  }

  //! Return a pointer to the data.
  pointer
  data()
  {
    return _start;
  }

  //! Return a const pointer to the data.
  const_pointer
  data() const
  {
    return _start;
  }

  //! Add an element to the end of the vector.
  void
  push_back(const value_type& x)
  {
    if (_finish == _endOfStorage) {
      _increaseCapacity();
    }
    *_finish++ = x;
  }

  //! Remove the last element.
  void
  pop_back()
  {
    --_finish;
  }

  //! Insert the value before the specified iterator.
  iterator
  insert(iterator position, const value_type& x)
  {
    // Increase the capacity if necessary.
    if (_finish == _endOfStorage) {
      const std::ptrdiff_t n = position - _start;
      _increaseCapacity();
      position = _start + n;
    }
    // Make space for the new element.
    iterator i = _finish;
    while (i != position) {
      *i = *(i - 1);
      --i;
    }
    ++_finish;
    // Assign and return an iterator to the element.
    *position = x;
    return position;
  }

  // CONTINUE: Implement for four integer types.
#if 0
  //! Insert n copies of the value before the specified iterator.
  void
  insert(iterator position, size_type n, const value_type& x)
  {

  }
#endif

  // CONTINUE: Implement.
#if 0
  //! Insert the range before the given position.
  template<typename _ForwardIterator>
  void
  insert(iterator position, _ForwardIterator first, _ForwardIterator last)
  {
  }
#endif

  //! Remove the specified element.
  iterator
  erase(iterator position)
  {
    for (iterator i = position; i + 1 != _finish; ++i) {
      *i = *(i + 1);
    }
    --_finish;
    // Return an iterator to the element after the one that has been erased.
    return position;
  }

  // CONTINUE: Implement.
#if 0
  //! Remove a range of elements.
  iterator
  erase(iterator first, iterator last);
#endif

  //! Swap data with another vector.
  void
  swap(vector& x)
  {
    std::swap(_start, x._start);
    std::swap(_finish, x._finish);
    std::swap(_endOfStorage, x._endOfStorage);
  }

  //! Erase all elements.
  void
  clear()
  {
    _finish = _start;
  }

protected:

  //! De-allocate memory.
  void
  _destroy()
  {
    _destroy(_start);
    _start = 0;
  }

  //! De-allocate the specified memory.
  void
  _destroy(const pointer p)
  {
    if (p) {
      delete[] p;
    }
  }

  //! Allocate memory. Do not set _finish.
  void
  _allocate(const size_type n)
  {
    if (n) {
      _start = new value_type[n];
    }
    else {
      _start = 0;
    }
    _endOfStorage = _start + n;
  }

  //! Allocate memory, set the size to the capacity.
  void
  _allocateFull(const size_type n)
  {
    _allocate(n);
    _finish = _endOfStorage;
  }

  //! Increase the capacity.
  void
  _increaseCapacity()
  {
    _increaseCapacity(std::max(size_type(1), 2 * size()));
  }

  //! Increase the capacity to the given size.
  void
  _increaseCapacity(const size_type n)
  {
    pointer oldStart = _start;
    size_type sz = size();
    _allocate(n);
    _finish = _start + sz;
    std::copy(oldStart, oldStart + sz, begin());
    _destroy(oldStart);
  }
};


//! Equality comparison.
template<typename _T>
inline
bool
operator==(const vector<_T>& x, const vector<_T>& y)
{
  return x.size() == y.size()
         && std::equal(x.begin(), x.end(), y.begin());
}

//! Less than, lexicographical comparison.
template<typename _T>
inline
bool
operator<(const vector<_T>& x, const vector<_T>& y)
{
  return std::lexicographical_compare(x.begin(), x.end(),
                                      y.begin(), y.end());
}

//! Based on operator==.
template<typename _T>
inline
bool
operator!=(const vector<_T>& x, const vector<_T>& y)
{
  return !(x == y);
}

//! Based on operator<.
template<typename _T>
inline
bool
operator>(const vector<_T>& x, const vector<_T>& y)
{
  return y < x;
}

//! Based on operator<.
template<typename _T>
inline
bool
operator<=(const vector<_T>& x, const vector<_T>& y)
{
  return !(y < x);
}

//! Based on operator<.
template<typename _T>
inline
bool
operator>=(const vector<_T>& x, const vector<_T>& y)
{
  return !(x < y);
}

//! Swap the vectors.
template<typename _T>
inline
void
swap(vector<_T>& x, vector<_T>& y)
{
  x.swap(y);
}

} // namespace container
}

#define __container_vector_ipp__
#include "stlib/container/vector.ipp"
#undef __container_vector_ipp__

#endif
