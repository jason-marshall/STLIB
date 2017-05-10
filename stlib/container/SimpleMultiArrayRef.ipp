// -*- C++ -*-

#if !defined(__container_SimpleMultiArrayRef_ipp__)
#error This file is an implementation detail of the class SimpleMultiArrayRef.
#endif

namespace stlib
{
namespace container
{

//--------------------------------------------------------------------------
// Constructors etc.

// Construct from a pointer to the memory, the array extents, and optionally
// the storage order.
template<typename _T, std::size_t _Dimension>
inline
SimpleMultiArrayRef<_T, _Dimension>::
SimpleMultiArrayRef(pointer data, const typename Base::IndexList& extents) :
  Base(data, extents),
  _data(data)
{
}

template<typename _T, std::size_t _Dimension>
inline
void
SimpleMultiArrayRef<_T, _Dimension>::
rebuild(pointer data, const typename Base::IndexList& extents)
{
  Base::rebuild(data, extents);
  _data = data;
}

// Assignment operator for arrays with contiguous memory.
template<typename _T, std::size_t _Dimension>
template<typename _T2>
inline
SimpleMultiArrayRef<_T, _Dimension>&
SimpleMultiArrayRef<_T, _Dimension>::
operator=(const SimpleMultiArrayConstRef<_T2, _Dimension>& other)
{
#ifdef STLIB_DEBUG
  // The arrays must have the same index range.
  assert(Base::extents() == other.extents());
#endif
  // Copy the elements.
  std::copy(other.begin(), other.end(), begin());
  return *this;
}

// Assignment operator.
template<typename _T, std::size_t _Dimension>
inline
SimpleMultiArrayRef<_T, _Dimension>&
SimpleMultiArrayRef<_T, _Dimension>::
operator=(const SimpleMultiArrayRef& other)
{
  if (this != &other) {
#ifdef STLIB_DEBUG
    // The arrays must have the same index range.
    assert(Base::extents() == other.extents());
#endif
    // Copy the elements.
    std::copy(other.begin(), other.end(), begin());
  }
  return *this;
}

//----------------------------------------------------------------------------
// Assignment operators with scalar operand.
//----------------------------------------------------------------------------

// Array-scalar addition.
template<typename _T, std::size_t _Dimension>
inline
SimpleMultiArrayRef<_T, _Dimension>&
operator+=(SimpleMultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename SimpleMultiArrayRef<_T, _Dimension>::iterator
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
SimpleMultiArrayRef<_T, _Dimension>&
operator-=(SimpleMultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename SimpleMultiArrayRef<_T, _Dimension>::iterator
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
SimpleMultiArrayRef<_T, _Dimension>&
operator*=(SimpleMultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename SimpleMultiArrayRef<_T, _Dimension>::iterator
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
SimpleMultiArrayRef<_T, _Dimension>&
operator/=(SimpleMultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename SimpleMultiArrayRef<_T, _Dimension>::iterator
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
SimpleMultiArrayRef<_T, _Dimension>&
operator%=(SimpleMultiArrayRef<_T, _Dimension>& x,
           typename boost::call_traits<_T>::param_type value)
{
  typedef typename SimpleMultiArrayRef<_T, _Dimension>::iterator
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
SimpleMultiArrayRef<_T, _Dimension>&
operator<<=(SimpleMultiArrayRef<_T, _Dimension>& x, const int offset)
{
  typedef typename SimpleMultiArrayRef<_T, _Dimension>::iterator
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
SimpleMultiArrayRef<_T, _Dimension>&
operator>>=(SimpleMultiArrayRef<_T, _Dimension>& x, const int offset)
{
  typedef typename SimpleMultiArrayRef<_T, _Dimension>::iterator
  iterator;
  const iterator end = x.end();
  for (iterator i = x.begin(); i != end; ++i) {
    *i >>= offset;
  }
  return x;
}

//----------------------------------------------------------------------------
// File I/O

// Read the %array extents and elements.
template<typename _T, std::size_t _Dimension>
inline
std::istream&
operator>>(std::istream& in, SimpleMultiArrayRef<_T, _Dimension>& x)
{
  typename SimpleMultiArrayRef<_T, _Dimension>::IndexList extents;
  in >> extents;
  assert(extents == x.extents());
  for (typename SimpleMultiArrayRef<_T, _Dimension>::iterator i = x.begin();
       i != x.end(); ++i) {
    in >> *i;
  }
  return in;
}

} // namespace container
}
