// -*- C++ -*-

#if !defined(__container_ArrayRef_ipp__)
#error This file is an implementation detail of the class ArrayRef.
#endif

namespace stlib
{
namespace container
{

//--------------------------------------------------------------------------
// Constructors etc.

// Copy constructor.
template<typename _T>
inline
ArrayRef<_T>::
ArrayRef(const ArrayRef& other) :
  VirtualBase(other),
  Base(other),
  ViewBase(other)
{
}

// Construct from a pointer to the memory and the size.
template<typename _T>
inline
ArrayRef<_T>::
ArrayRef(pointer data, const size_type size) :
  VirtualBase(data, size, 0, 1),
  Base(data, size),
  ViewBase(data, size, base(), stride())
{
}

// Construct from a pointer to the memory, the size, and the index base.
template<typename _T>
inline
ArrayRef<_T>::
ArrayRef(pointer data, const size_type size, const Index base) :
  VirtualBase(data, size, base, 1),
  Base(data, size, base),
  ViewBase(data, size, base, stride())
{
}

// Assignment operator for other array views.
template<typename _T>
template<typename _T2>
inline
ArrayRef<_T>&
ArrayRef<_T>::
operator=(const ArrayConstView<_T2>& other)
{
#ifdef STLIB_DEBUG
  // The arrays must have the same index range.
  assert(size() == other.size() && base() == other.base());
#endif
  // Copy the elements.
  std::copy(other.begin(), other.end(), begin());
  return *this;
}

// Assignment operator for arrays with contiguous memory.
template<typename _T>
template<typename _T2>
inline
ArrayRef<_T>&
ArrayRef<_T>::
operator=(const ArrayConstRef<_T2>& other)
{
#ifdef STLIB_DEBUG
  // The arrays must have the same index range.
  assert(size() == other.size() && base() == other.base());
#endif
  // Copy the elements.
  std::copy(other.begin(), other.end(), begin());
  return *this;
}

// Assignment operator.
template<typename _T>
inline
ArrayRef<_T>&
ArrayRef<_T>::
operator=(const ArrayRef& other)
{
  if (this != &other) {
#ifdef STLIB_DEBUG
    // The arrays must have the same index range.
    assert(size() == other.size() && base() == other.base());
#endif
    // Copy the elements.
    std::copy(other.begin(), other.end(), begin());
  }
  return *this;
}

} // namespace container
}
