// -*- C++ -*-

#if !defined(__sfc_Siblings_tcc__)
#error This file is an implementation detail of Siblings.
#endif

namespace stlib
{
namespace sfc
{


//--------------------------------------------------------------------------
// Constructors etc.


template<std::size_t _Dimension>
inline
Siblings<_Dimension>::
Siblings() :
  _size(0),
  _data()
{
}


template<std::size_t _Dimension>
inline
void
Siblings<_Dimension>::
push_back(const_reference x)
{
#ifdef STLIB_DEBUG
  assert(_size != NumChildren);
#endif
  _data[_size++] = x;
}


template<std::size_t _Dimension>
inline
void
Siblings<_Dimension>::
pop_back()
{
#ifdef STLIB_DEBUG
  assert(! empty());
#endif
  --_size;
}


template<std::size_t _Dimension>
inline
typename Siblings<_Dimension>::const_reference
Siblings<_Dimension>::
operator[](const size_type n) const
{
#ifdef STLIB_DEBUG
  assert(n < _size);
#endif
  return _data[n];
}


template<std::size_t _Dimension>
inline
typename Siblings<_Dimension>::reference
Siblings<_Dimension>::
operator[](const size_type n)
{
#ifdef STLIB_DEBUG
  assert(n < _size);
#endif
  return _data[n];
}


} // namespace sfc
}
