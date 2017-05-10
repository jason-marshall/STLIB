// -*- C++ -*-

#if !defined(__container_MultiArrayView_ipp__)
#error This file is an implementation detail of the class MultiArrayView.
#endif

namespace stlib
{
namespace container
{

//--------------------------------------------------------------------------
// Constructors etc.

// Assignment operator for other array views.
template<typename _T, std::size_t _Dimension>
template<typename _T2>
inline
MultiArrayView<_T, _Dimension>&
MultiArrayView<_T, _Dimension>::
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
MultiArrayView<_T, _Dimension>&
MultiArrayView<_T, _Dimension>::
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
MultiArrayView<_T, _Dimension>&
MultiArrayView<_T, _Dimension>::
operator=(const MultiArrayView& other)
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

//--------------------------------------------------------------------------
// Manipulators.

#if 0
// For the overlapping elements, set the first array to the unary function of
// the second.
template < typename _T1, std::size_t _Dimension, typename _T2,
           typename _UnaryFunction >
inline
void
applyUnaryToOverlap(MultiArrayConstView<_T1, _Dimension>* a,
                    const MultiArrayConstView<_T2, _Dimension>& b,
                    _UnaryFunction f)
{
  typedef MultiIndexRange<_Dimension> Range;
  typedef MultiIndexRangeIterator<_Dimension> Iterator;

  const Range range = intersect(Range(a->extents(), a->bases()),
                                Range(b.extents(), b.bases()));
  const Iterator end = Iterator::end(range);
  for (Iterator i = Iterator::begin(range); i != end; ++i) {
    (*a)(*i) = f(b(*i));
  }
}
#endif

//----------------------------------------------------------------------------
// Assignment operators with scalar operand.

// Array-scalar addition.
template<typename _T, std::size_t _Dimension>
inline
MultiArrayView<_T, _Dimension>&
operator+=(MultiArrayView<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename MultiArrayView<_T, _Dimension>::iterator
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
MultiArrayView<_T, _Dimension>&
operator-=(MultiArrayView<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename MultiArrayView<_T, _Dimension>::iterator
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
MultiArrayView<_T, _Dimension>&
operator*=(MultiArrayView<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename MultiArrayView<_T, _Dimension>::iterator
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
MultiArrayView<_T, _Dimension>&
operator/=(MultiArrayView<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename MultiArrayView<_T, _Dimension>::iterator
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
MultiArrayView<_T, _Dimension>&
operator%=(MultiArrayView<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename MultiArrayView<_T, _Dimension>::iterator
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
MultiArrayView<_T, _Dimension>&
operator<<=(MultiArrayView<_T, _Dimension>& x,
            const int offset)
{
  typedef typename MultiArrayView<_T, _Dimension>::iterator
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
MultiArrayView<_T, _Dimension>&
operator>>=(MultiArrayView<_T, _Dimension>& x,
            const int offset)
{
  typedef typename MultiArrayView<_T, _Dimension>::iterator
  iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i >>= offset;
  }
  return x;
}

} // namespace container
}
