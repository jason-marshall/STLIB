// -*- C++ -*-

/*!
  \file ads/indexedPriorityQueue/IndexedPriorityQueuePlaceboQueue.h
  \brief Placebo indexed priority queue that uses a FIFO queue.
*/

#if !defined(__ads_indexedPriorityQueue_IndexedPriorityQueuePlaceboQueue_h__)
#define __ads_indexedPriorityQueue_IndexedPriorityQueuePlaceboQueue_h__

#include <limits>
#include <queue>
#include <vector>

#include <cassert>

namespace stlib
{
namespace ads
{

//! Placebo indexed priority queue that uses a FIFO queue.
/*!
  \param Key is the key type.
*/
template < typename _Key = double >
class IndexedPriorityQueuePlaceboQueue
{
  //
  // Enumerations.
  //
public:

  enum {UsesPropensities = false};

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
  std::queue<int> _queue;
  int _topIndex;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the size.
  IndexedPriorityQueuePlaceboQueue(const std::size_t size) :
    _keys(size, std::numeric_limits<Key>::max()),
    // Invalid index.
    _topIndex(-1) {}

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

  //! Return the index of the top element.  Pop that element.
  int
  top()
  {
    _topIndex = _queue.front();
    _queue.pop();
    return _topIndex;
  }

  //! Pop the top element off the queue.  This is not allowed.
  void
  popTop()
  {
    pop(_topIndex);
  }

  //! Pop the element off the queue.  This is not allowed.
  void
  pop(const int index)
  {
    _keys[index] = std::numeric_limits<Key>::max();
    assert(false);
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
    _keys[index] = key;
    _queue.push(index);
  }

  //! Change the value in the queue.
  void
  set(const int index, const Key key)
  {
    _keys[index] = key;
  }

  //! Clear the queue.
  void
  clear()
  {
    std::fill(_keys.begin(), _keys.end(), std::numeric_limits<Key>::max());
    while (! _queue.empty()) {
      _queue.pop();
    }
  }

  //! Shift the keys by the specified amount.
  void
  shift(const Key x)
  {
    _keys += x;
  }

  //@}
};

} // namespace ads
}

#endif
