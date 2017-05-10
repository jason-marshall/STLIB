// -*- C++ -*-

#if !defined(__container_ViewIterator_ipp__)
#error This file is an implementation detail of the class ViewIterator.
#endif

namespace stlib
{
namespace container
{

//--------------------------------------------------------------------------
// Constructors etc.

// Return an iterator to the beginning of the index range.
template<typename _Array, bool _IsConst>
inline
ViewIterator<_Array, _IsConst>
ViewIterator<_Array, _IsConst>::
begin(ArrayReference array)
{
  ViewIterator x;
  x._iterator = array.data();
  x._array = &array;
  return x;
}

// Return an iterator to the end of the index range.
template<typename _Array, bool _IsConst>
inline
ViewIterator<_Array, _IsConst>
ViewIterator<_Array, _IsConst>::
end(ArrayReference array)
{
  ViewIterator x;
  x._iterator = array.data() + array.size() * array.stride();
  x._array = &array;
  return x;
}

// Copy constructor from non-const.
template<typename _Array, bool _IsConst>
template<bool _IsConst2>
inline
ViewIterator<_Array, _IsConst>::
ViewIterator(const ViewIterator<Array, _IsConst2>& other) :
  _iterator(other.base()),
  _array(other.array())
{
}

// Assignment operator from non-const.

template<typename _Array, bool _IsConst>
template<bool _IsConst2>
inline
ViewIterator<_Array, _IsConst>&
ViewIterator<_Array, _IsConst>::
operator=(const ViewIterator<Array, _IsConst2>& other)
{
  _iterator = other.base();
  _array = other.array();
  return *this;
}

//--------------------------------------------------------------------------
// Forward iterator requirements.

// Pre-increment.
template<typename _Array, bool _IsConst>
inline
ViewIterator<_Array, _IsConst>&
ViewIterator<_Array, _IsConst>::
operator++()
{
#ifdef STLIB_DEBUG
  assert(isValid());
#endif

  // Increment the iterator.
  _iterator += _array->stride();

#ifdef STLIB_DEBUG
  assert(isValid() || isEnd());
#endif

  return *this;
}

// Post-increment.
template<typename _Array, bool _IsConst>
inline
ViewIterator<_Array, _IsConst>
ViewIterator<_Array, _IsConst>::
operator++(int)
{
  ViewIterator tmp(*this);
  ++*this;
  return tmp;
}

//--------------------------------------------------------------------------
// Bidirectional iterator requirements.

// Pre-decrement.
template<typename _Array, bool _IsConst>
inline
ViewIterator<_Array, _IsConst>&
ViewIterator<_Array, _IsConst>::
operator--()
{
#ifdef STLIB_DEBUG
  assert((isValid() || isEnd()) && ! isBegin());
#endif

  // Decrement the iterator.
  _iterator -= _array->stride();

#ifdef STLIB_DEBUG
  assert(isValid());
#endif

  return *this;
}


// Post-decrement.
template<typename _Array, bool _IsConst>
inline
ViewIterator<_Array, _IsConst>
ViewIterator<_Array, _IsConst>::
operator--(int)
{
  ViewIterator tmp(*this);
  --*this;
  return tmp;
}

} // namespace container
}
