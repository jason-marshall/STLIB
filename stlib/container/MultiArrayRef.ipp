// -*- C++ -*-

#if !defined(__container_MultiArrayRef_ipp__)
#error This file is an implementation detail of the class MultiArrayRef.
#endif

namespace stlib
{
namespace container
{

//--------------------------------------------------------------------------
// Constructors etc.

// Copy constructor.
template<typename _T, std::size_t _Dimension>
inline
MultiArrayRef<_T, _Dimension>::
MultiArrayRef(const MultiArrayRef& other) :
  VirtualBase(other),
  Base(other),
  ViewBase(other)
{
}

// Construct from a pointer to the memory, the array extents, and optionally
// the storage order.
template<typename _T, std::size_t _Dimension>
inline
MultiArrayRef<_T, _Dimension>::
MultiArrayRef(pointer data, const SizeList& extents,
              const Storage& storage) :
  VirtualBase(data, extents, ext::filled_array<IndexList>(0), storage,
              computeStrides(extents, storage)),
  Base(data, extents, storage),
  ViewBase(data, extents, bases(), storage, strides())
{
}

// Construct from a pointer to the memory, the array extents, the index bases,
// and optionally the storage order.
template<typename _T, std::size_t _Dimension>
inline
MultiArrayRef<_T, _Dimension>::
MultiArrayRef(pointer data, const SizeList& extents,
              const IndexList& bases, const Storage& storage) :
  VirtualBase(data, extents, bases, storage, computeStrides(extents, storage)),
  Base(data, extents, bases, storage),
  ViewBase(data, extents, bases, storage, strides())
{
}

// Assignment operator for other array views.
template<typename _T, std::size_t _Dimension>
template<typename _T2>
inline
MultiArrayRef<_T, _Dimension>&
MultiArrayRef<_T, _Dimension>::
operator=(const MultiArrayConstView<_T2, _Dimension>& other)
{
  typedef MultiIndexRangeIterator<Dimension> Iterator;

#ifdef STLIB_DEBUG
  // The arrays must have the same index range.
  assert(extents() == other.extents() && bases() == other.bases());
#endif

  // Copy the elements.
  if (storage() == other.storage()) {
    // If the storage orders are the same, we can use the built-in iterators.
    std::copy(other.begin(), other.end(), begin());
  }
  else {
    // If the storage orders differ, iterate over the index range and do
    // array indexing.
    const Range range(extents(), bases());
    const Iterator end = Iterator::end(range);
    for (Iterator i = Iterator::begin(range); i != end; ++i) {
      (*this)(*i) = other(*i);
    }
  }
  return *this;
}

// Assignment operator for arrays with contiguous memory.
template<typename _T, std::size_t _Dimension>
template<typename _T2>
inline
MultiArrayRef<_T, _Dimension>&
MultiArrayRef<_T, _Dimension>::
operator=(const MultiArrayConstRef<_T2, _Dimension>& other)
{
  typedef MultiIndexRangeIterator<Dimension> Iterator;

#ifdef STLIB_DEBUG
  // The arrays must have the same index range.
  assert(extents() == other.extents() && bases() == other.bases());
#endif

  // Copy the elements.
  if (storage() == other.storage()) {
    // If the storage orders are the same, we can use the built-in iterators.
    std::copy(other.begin(), other.end(), begin());
  }
  else {
    // If the storage orders differ, iterate over the index range and do
    // array indexing.
    const Range range(extents(), bases());
    const Iterator end = Iterator::end(range);
    for (Iterator i = Iterator::begin(range); i != end; ++i) {
      (*this)(*i) = other(*i);
    }
  }
  return *this;
}

// Assignment operator.
template<typename _T, std::size_t _Dimension>
inline
MultiArrayRef<_T, _Dimension>&
MultiArrayRef<_T, _Dimension>::
operator=(const MultiArrayRef& other)
{
  typedef MultiIndexRangeIterator<Dimension> Iterator;

  if (this != &other) {
#ifdef STLIB_DEBUG
    // The arrays must have the same index range.
    assert(extents() == other.extents() && bases() == other.bases());
#endif

    // Copy the elements.
    if (storage() == other.storage()) {
      // If the storage orders are the same, we can use the built-in iterators.
      std::copy(other.begin(), other.end(), begin());
    }
    else {
      // If the storage orders differ, iterate over the index range and do
      // array indexing.
      const Range range(extents(), bases());
      const Iterator end = Iterator::end(range);
      for (Iterator i = Iterator::begin(range); i != end; ++i) {
        (*this)(*i) = other(*i);
      }
    }
  }
  return *this;
}

//----------------------------------------------------------------------------
// Assignment operators with scalar operand.
//----------------------------------------------------------------------------

// Array-scalar addition.
template<typename _T, std::size_t _Dimension>
inline
MultiArrayRef<_T, _Dimension>&
operator+=(MultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename MultiArrayRef<_T, _Dimension>::iterator
  iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i += value;
  }
  return x;
}

// Array-scalar subtraction.
template<typename _T, std::size_t _Dimension>
inline
MultiArrayRef<_T, _Dimension>&
operator-=(MultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename MultiArrayRef<_T, _Dimension>::iterator
  iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i -= value;
  }
  return x;
}

// Array-scalar multiplication.
template<typename _T, std::size_t _Dimension>
inline
MultiArrayRef<_T, _Dimension>&
operator*=(MultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename MultiArrayRef<_T, _Dimension>::iterator
  iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i *= value;
  }
  return x;
}

// Array-scalar division.
template<typename _T, std::size_t _Dimension>
inline
MultiArrayRef<_T, _Dimension>&
operator/=(MultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename MultiArrayRef<_T, _Dimension>::iterator
  iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i /= value;
  }
  return x;
}

// Array-scalar modulus.
template<typename _T, std::size_t _Dimension>
inline
MultiArrayRef<_T, _Dimension>&
operator%=(MultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename MultiArrayRef<_T, _Dimension>::iterator
  iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i %= value;
  }
  return x;
}

// Left shift.
template<typename _T, std::size_t _Dimension>
inline
MultiArrayRef<_T, _Dimension>&
operator<<=(MultiArrayRef<_T, _Dimension>& x, const int offset)
{
  typedef typename MultiArrayRef<_T, _Dimension>::iterator
  iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i <<= offset;
  }
  return x;
}

// Right shift.
template<typename _T, std::size_t _Dimension>
inline
MultiArrayRef<_T, _Dimension>&
operator>>=(MultiArrayRef<_T, _Dimension>& x, const int offset)
{
  typedef typename MultiArrayRef<_T, _Dimension>::iterator
  iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i >>= offset;
  }
  return x;
}

//----------------------------------------------------------------------------
// File I/O

// Read the %array extents, index bases, storage, and elements.
template<typename _T, std::size_t _Dimension>
inline
std::istream&
operator>>(std::istream& in, MultiArrayRef<_T, _Dimension>& x)
{
  typename MultiArrayRef<_T, _Dimension>::SizeList extents;
  in >> extents;
  typename MultiArrayRef<_T, _Dimension>::IndexList bases;
  in >> bases;
  typename MultiArrayRef<_T, _Dimension>::Storage storage;
  in >> storage;
  x.rebuild(extents, bases, storage);
  for (typename MultiArrayRef<_T, _Dimension>::iterator i = x.begin();
       i != x.end(); ++i) {
    in >> *i;
  }
  return in;
}

} // namespace container
}
