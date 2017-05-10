// -*- C++ -*-

#if !defined(__container_MultiArray_ipp__)
#error This file is an implementation detail of the class MultiArray.
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
MultiArray<_T, _Dimension>::
MultiArray() :
  // Null pointer to memory and zero extents.
  VirtualBase(0, ext::filled_array<SizeList>(0),
              ext::filled_array<IndexList>(0), ColumnMajor(),
              ext::filled_array<IndexList>(1)),
  Base(0, extents())
{
}

// Copy constructor for different types.
template<typename _T, std::size_t _Dimension>
template<typename _T2>
inline
MultiArray<_T, _Dimension>::
MultiArray(const MultiArrayConstRef<_T2, _Dimension>& other) :
  VirtualBase(0, other.extents(), other.bases(),
              other.storage(), other.strides()),
  Base(0, extents(), bases(), storage())
{
  // Allocate the memory.
  setData(new value_type[other.size()]);
  // Copy the elements.
  std::copy(other.begin(), other.end(), begin());
}

// Copy constructor.
template<typename _T, std::size_t _Dimension>
inline
MultiArray<_T, _Dimension>::
MultiArray(const MultiArray& other) :
  VirtualBase(0, other.extents(), other.bases(),
              other.storage(), other.strides()),
  Base(0, extents(), bases(), storage())
{
  // Allocate the memory.
  setData(new value_type[other.size()]);
  // Copy the elements.
  std::copy(other.begin(), other.end(), begin());
}

// Construct from the array extents, and optionally an initial value.
template<typename _T, std::size_t _Dimension>
inline
MultiArray<_T, _Dimension>::
MultiArray(const SizeList& extents, const value_type& value) :
  VirtualBase(0, extents, ext::filled_array<IndexList>(0),
              Storage(ColumnMajor()), computeStrides(extents)),
  Base(0, extents)
{
  // Allocate the memory.
  setData(new value_type[ext::product(extents)]);
  // Initialize the data.
  std::fill(begin(), end(), value);
}


// Construct from the %array extents, the storage order, and optionally an
// initial value.
template<typename _T, std::size_t _Dimension>
inline
MultiArray<_T, _Dimension>::
MultiArray(const SizeList& extents, const Storage& storage,
           const value_type& value) :
  VirtualBase(0, extents, ext::filled_array<IndexList>(0),
              storage, computeStrides(extents, storage)),
  Base(0, extents, storage)
{
  // Allocate the memory.
  setData(new value_type[ext::product(extents)]);
  // Initialize the data.
  std::fill(begin(), end(), value);
}


// Construct from the %array extents, the index bases, and optionally an
// initial value.
template<typename _T, std::size_t _Dimension>
inline
MultiArray<_T, _Dimension>::
MultiArray(const SizeList& extents, const IndexList& bases,
           const value_type& value) :
  VirtualBase(0, extents, bases, Storage(ColumnMajor()),
              computeStrides(extents)),
  Base(0, extents, bases, Storage(ColumnMajor()))
{
  // Allocate the memory.
  setData(new value_type[ext::product(extents)]);
  // Initialize the data.
  std::fill(begin(), end(), value);
}


// Construct from the %array extents, the index bases, the storage order,
// and optionally an initial value.
template<typename _T, std::size_t _Dimension>
inline
MultiArray<_T, _Dimension>::
MultiArray(const SizeList& extents, const IndexList& bases,
           const Storage& storage, const value_type& value) :
  VirtualBase(0, extents, bases, storage, computeStrides(extents, storage)),
  Base(0, extents, bases, storage)
{
  // Allocate the memory.
  setData(new value_type[ext::product(extents)]);
  // Initialize the data.
  std::fill(begin(), end(), value);
}


// Assignment operator for other array views.
template<typename _T, std::size_t _Dimension>
template<typename _T2>
inline
MultiArray<_T, _Dimension>&
MultiArray<_T, _Dimension>::
operator=(const MultiArrayConstView<_T2, _Dimension>& other)
{
  Base::operator=(other);
  return *this;
}

// Assignment operator for arrays with contiguous memory.
template<typename _T, std::size_t _Dimension>
template<typename _T2>
inline
MultiArray<_T, _Dimension>&
MultiArray<_T, _Dimension>::
operator=(const MultiArrayConstRef<_T2, _Dimension>& other)
{
  Base::operator=(other);
  return *this;
}

// Assignment operator.
template<typename _T, std::size_t _Dimension>
inline
MultiArray<_T, _Dimension>&
MultiArray<_T, _Dimension>::
operator=(const MultiArray& other)
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
MultiArray<_T, _Dimension>::
rebuild(const SizeList& extents, const IndexList& bases,
        const Storage& storage)
{
  const size_type newSize = ext::product(extents);
  if (newSize == size()) {
    Base::rebuild(data(), extents, bases, storage);
  }
  else {
    destroy();
    Base::rebuild(new value_type[newSize], extents, bases, storage);
  }
}

//----------------------------------------------------------------------------
// File I/O

// Read the %array extents, index bases, storage, and elements.
template<typename _T, std::size_t _Dimension>
inline
std::istream&
operator>>(std::istream& in, MultiArray<_T, _Dimension>& x)
{
  typename MultiArray<_T, _Dimension>::SizeList extents;
  in >> extents;
  typename MultiArray<_T, _Dimension>::IndexList bases;
  in >> bases;
  typename MultiArray<_T, _Dimension>::Storage storage;
  in >> storage;
  x.rebuild(extents, bases, storage);
  for (typename MultiArray<_T, _Dimension>::iterator i = x.begin();
       i != x.end(); ++i) {
    in >> *i;
  }
  return in;
}

// Read the %array extents, index bases, storage, and elements.
template<typename _T, std::size_t _Dimension>
inline
void
read(MultiArray<_T, _Dimension>* x, std::istream& in)
{
  typename MultiArray<_T, _Dimension>::SizeList extents;
  ext::read(&extents, in);
  typename MultiArray<_T, _Dimension>::IndexList bases;
  ext::read(&bases, in);
  typename MultiArray<_T, _Dimension>::Storage storage;
  ext::read(&storage, in);
  x->rebuild(extents, bases, storage);
  readElements(x, in);
}


} // namespace container
}
