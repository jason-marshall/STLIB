// -*- C++ -*-

#if !defined(__container_SimpleMultiArray_ipp__)
#error This file is an implementation detail of the class SimpleMultiArray.
#endif

namespace stlib
{
namespace container
{

//--------------------------------------------------------------------------
// Constructors etc.

// Default constructor.
template<typename _T, std::size_t _Dimension>
inline
SimpleMultiArray<_T, _Dimension>::
SimpleMultiArray() :
  // Null pointer to memory and zero extents.
  Base(0, ext::filled_array<typename Base::IndexList>(0))
{
}

// Copy constructor for different types.
template<typename _T, std::size_t _Dimension>
template<typename _T2>
inline
SimpleMultiArray<_T, _Dimension>::
SimpleMultiArray(const SimpleMultiArrayConstRef<_T2, _Dimension>& other) :
  Base(0, other.extents())
{
  // Allocate the memory.
  Base::setData(new typename Base::value_type[other.size()]);
  // Copy the elements.
  std::copy(other.begin(), other.end(), Base::begin());
}

// Copy constructor.
template<typename _T, std::size_t _Dimension>
inline
SimpleMultiArray<_T, _Dimension>::
SimpleMultiArray(const SimpleMultiArray& other) :
  Base(0, other.extents())
{
  // Allocate the memory.
  Base::setData(new typename Base::value_type[other.size()]);
  // Copy the elements.
  std::copy(other.begin(), other.end(), Base::begin());
}

// Construct from the array extents, and an initial value.
template<typename _T, std::size_t _Dimension>
inline
SimpleMultiArray<_T, _Dimension>::
SimpleMultiArray(const typename Base::IndexList& extents) :
  Base(0, extents)
{
  // Allocate the memory.
  Base::setData(new typename Base::value_type[ext::product(extents)]);
}


// Construct from the array extents, and an initial value.
template<typename _T, std::size_t _Dimension>
inline
SimpleMultiArray<_T, _Dimension>::
SimpleMultiArray(const typename Base::IndexList& extents,
                 const typename Base::value_type& value) :
  Base(0, extents)
{
  // Allocate the memory.
  Base::setData(new typename Base::value_type[ext::product(extents)]);
  // Initialize the data.
  std::fill(Base::begin(), Base::end(), value);
}


// Assignment operator for arrays with contiguous memory.
template<typename _T, std::size_t _Dimension>
template<typename _T2>
inline
SimpleMultiArray<_T, _Dimension>&
SimpleMultiArray<_T, _Dimension>::
operator=(const SimpleMultiArrayConstRef<_T2, _Dimension>& other)
{
  Base::operator=(other);
  return *this;
}

// Assignment operator.
template<typename _T, std::size_t _Dimension>
inline
SimpleMultiArray<_T, _Dimension>&
SimpleMultiArray<_T, _Dimension>::
operator=(const SimpleMultiArray& other)
{
  if (this != &other) {
    Base::operator=(other);
  }
  return *this;
}

// Rebuild the data structure. Re-allocate memory if the size changes.
template<typename _T, std::size_t _Dimension>
inline
void
SimpleMultiArray<_T, _Dimension>::
rebuild(const typename Base::IndexList& extents)
{
  const typename Base::size_type newSize = ext::product(extents);
  if (newSize == Base::size()) {
    Base::rebuild(Base::data(), extents);
  }
  else {
    destroy();
    Base::rebuild(new typename Base::value_type[newSize], extents);
  }
}

//----------------------------------------------------------------------------
// File I/O

// Read the %array extents, index bases, storage, and elements.
template<typename _T, std::size_t _Dimension>
inline
std::istream&
operator>>(std::istream& in, SimpleMultiArray<_T, _Dimension>& x)
{
  typename SimpleMultiArray<_T, _Dimension>::IndexList extents;
  in >> extents;
  x.rebuild(extents);
  for (typename SimpleMultiArray<_T, _Dimension>::iterator i = x.begin();
       i != x.end(); ++i) {
    in >> *i;
  }
  return in;
}

} // namespace container
}
