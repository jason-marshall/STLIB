// -*- C++ -*-

#if !defined(__geom_CyclicIndex_h__)
#define __geom_CyclicIndex_h__

#include <cassert>

namespace stlib
{
namespace geom
{

template<typename _Index>
class CyclicIndex;

//
// Increment and Decrement Operators.
//

//! Increment the index.
/*! \relates CyclicIndex */
template<typename _Index>
CyclicIndex<_Index>&
operator++(CyclicIndex<_Index>& ci);

//! Decrement the index.
/*! \relates CyclicIndex */
template<typename _Index>
CyclicIndex<_Index>&
operator--(CyclicIndex<_Index>& ci);


//! A class for a cyclic index.
template<typename _Index>
class CyclicIndex
{
public:

  //! The integer index type.
  typedef _Index Index;

  //
  // Data
  //

private:

  Index _index, _n;

  //
  // Friends
  //

  friend CyclicIndex& operator++<>(CyclicIndex& ci);
  friend CyclicIndex& operator--<>(CyclicIndex& ci);

  //! Default constructor not implemented.
  CyclicIndex();

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

  //! Constructor.  Initialize index to zero.
  CyclicIndex(const Index n) :
    _index(0),
    _n(n)
  {
#ifdef STLIB_DEBUG
    assert(_n > 0);
#endif
  }

  //! Copy constructor.
  CyclicIndex(const CyclicIndex& other) :
    _index(other._index),
    _n(other._n)
  {
#ifdef STLIB_DEBUG
    assert(_n > 0);
#endif
  }

  //! Assignment operator.
  CyclicIndex&
  operator=(const CyclicIndex& other);

  //! Trivial destructor.
  ~CyclicIndex() {}

  // @}
  //--------------------------------------------------------------------------
  //! \name Accesors.
  // @{

  //! Return the index.
  Index
  operator()() const
  {
    return _index;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{

  //! Set the index to i mod N.
  void
  set(Index i);
};

} // namespace geom
}

#define __geom_CyclicIndex_ipp__
#include "stlib/geom/polytope/CyclicIndex.ipp"
#undef __geom_CyclicIndex_ipp__

#endif
