// -*- C++ -*-

/*!
  \file ads/indexedPriorityQueue/IndexedPriorityQueueLinearSearchSimple.h
  \brief Indexed priority queue that uses a linear search.
*/

#if !defined(__ads_indexedPriorityQueue_IndexedPriorityQueueLinearSearchSimple_h__)
#define __ads_indexedPriorityQueue_IndexedPriorityQueueLinearSearchSimple_h__

#include "stlib/ext/vector.h"


#include <algorithm>
#include <limits>

namespace stlib
{
namespace ads
{

USING_STLIB_EXT_VECTOR_MATH_OPERATORS;

//! Indexed priority queue that uses a linear search.
/*!
  \param Key is the key type.
*/
template < typename _Key = double,
           bool _UseInfinity = false >
class IndexedPriorityQueueLinearSearchSimple
{
  //
  // Enumerations.
  //
public:

  //! Whether to use the infinity value defined by IEEE 754.
  enum {UseInfinity = _UseInfinity, UsesPropensities = false};

  //
  // Public types.
  //
public:

  //! The key type.
  typedef _Key Key;

  //
  // Member data.
  //
private:

  std::vector<Key> _keys;
  int _topIndex;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the size.
  IndexedPriorityQueueLinearSearchSimple(const std::size_t size) :
    // No valid keys.
    _keys(size, std::numeric_limits<Key>::max()),
    // Invalid index.
    _topIndex(-1)
  {
  }

  // Default copy constructor, assignment operator, and destructor are fine.

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{

  //! Return the key of the specified element.
  Key
  get(const int index) const
  {
    return _keys[index];
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  //! Return the index of the top element.
  int
  top()
  {
#ifdef STLIB_DEBUG
    assert(! _keys.empty());
#endif
    return _topIndex =
             std::min_element(_keys.begin(), _keys.end()) - _keys.begin();
  }

  //! Pop the top element off the queue.
  void
  popTop()
  {
    pop(_topIndex);
  }

  //! Pop the element off the queue.
  void
  pop(const int index)
  {
    pop(index, std::integral_constant<bool, UseInfinity>());
  }

  //! Push the top value into the queue.
  void
  pushTop(const Key key)
  {
    push(_topIndex, key);
  }

  //! Push the value into the queue.
  void
  push(const int index, const Key key)
  {
#ifdef STLIB_DEBUG
    assert(key != std::numeric_limits<Key>::max());
#endif
    _keys[index] = key;
  }

  //! Change the value in the queue.
  void
  set(const int index, const Key key)
  {
#ifdef STLIB_DEBUG
    assert(key != std::numeric_limits<Key>::max());
#endif
    _keys[index] = key;
  }

  //! Clear the queue.
  void
  clear()
  {
    std::fill(_keys.begin(), _keys.end(), std::numeric_limits<Key>::max());
    _topIndex = -1;
  }

  //! Shift the keys by the specified amount.
  void
  shift(const Key x)
  {
    _keys += x;
  }

private:

  //! Pop the element off the queue.
  void
  pop(const int index, std::false_type /*dummy*/)
  {
    _keys[index] = std::numeric_limits<Key>::max();
  }

  //! Pop the element off the queue.
  void
  pop(const int index, std::true_type /*dummy*/)
  {
    _keys[index] = std::numeric_limits<Key>::infinity();
  }

  //@}
};

} // namespace ads
}

#endif
