// -*- C++ -*-

#if !defined(__container_MultiArrayBase_ipp__)
#error This file is an implementation detail of the class MultiArrayBase.
#endif

namespace stlib
{
namespace container
{

//--------------------------------------------------------------------------
// Constructors etc.

// Construct from the array extents, the index bases, the storage order, and
// the strides.
template<std::size_t _Dimension>
inline
MultiArrayBase<_Dimension>::
MultiArrayBase(const SizeList& extents, const IndexList& bases,
               const Storage& storage, const IndexList& strides) :
  _extents(extents),
  _bases(bases),
  _storage(storage),
  _strides(strides),
  _offset(ext::dot(_strides, _bases)),
  _size(ext::product(_extents))
{
}

// Rebuild the data structure.
template<std::size_t _Dimension>
inline
void
MultiArrayBase<_Dimension>::
rebuild(const SizeList& extents, const IndexList& bases,
        const Storage& storage, const IndexList& strides)
{
  _extents = extents;
  _bases = bases;
  _storage = storage;
  _strides = strides;
  _offset = ext::dot(_strides, _bases);
  _size = ext::product(_extents);
}

} // namespace container
}
