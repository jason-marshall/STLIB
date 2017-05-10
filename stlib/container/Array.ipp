// -*- C++ -*-

#if !defined(__container_Array_ipp__)
#error This file is an implementation detail of the class Array.
#endif

namespace stlib
{
namespace container
{

//--------------------------------------------------------------------------
// Constructors etc.

// Default constructor.
template<typename _T>
inline
Array<_T>::
Array() :
  // Null pointer to memory and zero extents.
  VirtualBase(0, 0, 0, 1),
  Base(0, size())
{
}

// Copy constructor for different types.
template<typename _T>
template<typename _T2>
inline
Array<_T>::
Array(const ArrayConstRef<_T2>& other) :
  VirtualBase(0, other.size(), other.base(),
              other.stride()),
  Base(0, size(), base())
{
  // Allocate the memory.
  setData(new value_type[other.size()]);
  // Copy the elements.
  std::copy(other.begin(), other.end(), begin());
}

// Copy constructor.
template<typename _T>
inline
Array<_T>::
Array(const Array& other) :
  VirtualBase(0, other.size(), other.base(),
              other.stride()),
  Base(0, size(), base())
{
  // Allocate the memory.
  setData(new value_type[other.size()]);
  // Copy the elements.
  std::copy(other.begin(), other.end(), begin());
}

// Construct from the size.
template<typename _T>
inline
Array<_T>::
Array(const size_type size) :
  VirtualBase(0, size, 0, 1),
  Base(0, size)
{
  // Allocate the memory.
  setData(new value_type[size]);
}


// Construct from the size and the index base
template<typename _T>
inline
Array<_T>::
Array(const size_type size, const Index base) :
  VirtualBase(0, size, base, 1),
  Base(0, size, base)
{
  // Allocate the memory.
  setData(new value_type[size]);
}

// Assignment operator for other array views.
template<typename _T>
template<typename _T2>
inline
Array<_T>&
Array<_T>::
operator=(const ArrayConstView<_T2>& other)
{
  Base::operator=(other);
  return *this;
}

// Assignment operator for arrays with contiguous memory.
template<typename _T>
template<typename _T2>
inline
Array<_T>&
Array<_T>::
operator=(const ArrayConstRef<_T2>& other)
{
  Base::operator=(other);
  return *this;
}

// Assignment operator.
template<typename _T>
inline
Array<_T>&
Array<_T>::
operator=(const Array& other)
{
  if (this != &other) {
    Base::operator=(other);
  }
  return *this;
}

// Rebuild the data structure. Re-allocate memory if the size changes.
template<typename _T>
inline
void
Array<_T>::
rebuild(const size_type newSize, const Index base)
{
  if (newSize == size()) {
    Base::rebuild(data(), size(), base);
  }
  else {
    Base::rebuild(new value_type[newSize], newSize, base);
  }
}

} // namespace container
}
