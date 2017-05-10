// -*- C++ -*-

#if !defined(__geom_CyclicIndex_ipp__)
#error This file is an implementation detail of the class CyclicIndex.
#endif

namespace stlib
{
namespace geom
{

//
// Constructors and destructor.
//

template<typename _Index>
inline
CyclicIndex<_Index>&
CyclicIndex<_Index>::operator=(const CyclicIndex& other)
{
  // Avoid assignment to self
  if (&other != this) {
    _index = other._index;
    _n = other._n;
  }
  // Return *this so assignments can chain
  return * this;
}

//
// Manipulators
//

template<typename _Index>
inline
void
CyclicIndex<_Index>::set(Index i)
{
  if (i < 0) {
    i += _n * (- i / _n + 1);
  }
  _index = i % _n;
}

//
// Increment and decrement operators
//

template<typename _Index>
inline
CyclicIndex<_Index>&
operator++(CyclicIndex<_Index>& ci)
{
  ci._index = (ci._index + 1) % ci._n;
  return ci;
}

template<typename _Index>
inline
CyclicIndex<_Index>&
operator--(CyclicIndex<_Index>& ci)
{
  ci._index = (ci._index + ci._n - 1) % ci._n;
  return ci;
}

} // namespace geom
}
