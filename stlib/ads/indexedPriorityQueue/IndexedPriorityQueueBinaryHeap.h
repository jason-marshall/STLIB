// -*- C++ -*-

/*!
  \file ads/indexedPriorityQueue/IndexedPriorityQueueBinaryHeap.h
  \brief Indexed priority queue that partitions the active and inactive elements.
*/

#if !defined(__ads_indexedPriorityQueue_IndexedPriorityQueueBinaryHeap_h__)
#define __ads_indexedPriorityQueue_IndexedPriorityQueueBinaryHeap_h__

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueBase.h"

#include <iostream>

#include <cstddef>

namespace stlib
{
namespace ads
{

//! Indexed priority queue that partitions the active and inactive elements.
/*!
  \param Key is the key type.
*/
template < typename _Key = double >
class IndexedPriorityQueueBinaryHeap :
  public IndexedPriorityQueueBase<_Key>
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
  // Private types.
  //
private:

  typedef IndexedPriorityQueueBase<Key> Base;
  typedef typename Base::Iterator Iterator;

  //
  // Member data.
  //
protected:

  using Base::_keys;
  using Base::_indices;
  using Base::_queue;
  using Base::_topIndex;
  using Base::_compare;

  //! The end of the heap.
  typename std::vector<Iterator>::iterator _heapEnd;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the size.
  IndexedPriorityQueueBinaryHeap(const std::size_t size) :
    Base(size),
    _heapEnd(_queue.begin())
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the key of the specified element.
  using Base::get;

  //! Return true if the binary heap data struture is valid.
  bool
  isValid() const
  {
    const std::size_t size = _heapEnd - _queue.begin();
    std::size_t parent = 0;
    for (std::size_t child = 1; child < size; ++child) {
      if (_compare(_queue[child], _queue[parent])) {
        return false;
      }
      if ((child & 1) == 0) {
        ++parent;
      }
    }
    return true;
  }

private:

  //! Return the beginning of the queue.
  using Base::getQueueBeginning;

  //! Return the end of the queue.
  typename std::vector<Iterator>::const_iterator
  getQueueEnd() const
  {
    return _heapEnd;
  }

  //! Print the queue.
  void
  printQueue(std::ostream& out) const
  {
    int n = 0;
    for (typename std::vector<Iterator>::const_iterator i =
           getQueueBeginning(); i != _heapEnd; ++i, ++n) {
      out << n << " " << **i << "\n";
    }
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Return the index of the top element.
  int
  top()
  {
#ifdef STLIB_DEBUG
    assert(! _keys.empty());
#endif
    return _topIndex = *getQueueBeginning() - _keys.begin();
  }

  //! Pop the top element off the queue.
  void
  popTop()
  {
#ifdef STLIB_DEBUG
    assert(_keys[_topIndex] != std::numeric_limits<Key>::max());
#endif
    _keys[_topIndex] = std::numeric_limits<Key>::max();
    remove(_topIndex);
#ifdef STLIB_DEBUG
    assert(isValid());
#endif
  }

  //! Pop the element off the queue.
  void
  pop(const int index)
  {
#ifdef STLIB_DEBUG
    // It must be in the active queue.
    assert(_keys[index] != std::numeric_limits<Key>::max());
#endif
    _keys[index] = std::numeric_limits<Key>::max();
    remove(index);
#ifdef STLIB_DEBUG
    assert(isValid());
#endif
  }

  //! Push the value into the queue.
  void
  push(const int index, const Key key)
  {
#ifdef STLIB_DEBUG
    assert(_keys[index] == std::numeric_limits<Key>::max() &&
           key != std::numeric_limits<Key>::max());
#endif
    _keys[index] = key;
    insert(index);
#ifdef STLIB_DEBUG
    assert(isValid());
#endif
  }

  //! Push the value at the top into the queue.
  void
  pushTop(const Key key)
  {
#ifdef STLIB_DEBUG
    assert(key != std::numeric_limits<Key>::max());
#endif
    _keys[_topIndex] = key;
    pushDown(_indices[_topIndex]);
#ifdef STLIB_DEBUG
    assert(isValid());
#endif
  }

  //! Change the value in the queue.
  void
  set(const int index, const Key key)
  {
#ifdef STLIB_DEBUG
    assert(_keys[index] != std::numeric_limits<Key>::max() &&
           key != std::numeric_limits<Key>::max());
#endif
#ifdef STLIB_DEBUG
    assert(isValid());
#endif

    if (key < _keys[index]) {
      _keys[index] = key;
      pushUp(_indices[index]);
    }
    else {
      _keys[index] = key;
      pushDown(_indices[index]);
    }
#ifdef STLIB_DEBUG
    assert(isValid());
#endif
  }

  //! Clear the queue.
  void
  clear()
  {
    Base::clear();
    _heapEnd = _queue.begin();
    _topIndex = -1;
#ifdef STLIB_DEBUG
    assert(isValid());
#endif
  }

  //! Shift the keys by the specified amount.
  using Base::shift;

private:

  //! Return the end of the queue.
  typename std::vector<Iterator>::iterator
  getQueueEnd()
  {
    return _heapEnd;
  }

  //! Swap the two elements' positions in the queue.
  using Base::swap;

  //! Recompute the indices after the queue has changed.
  using Base::recomputeIndices;

  //! Remove the element from the heap.
  /*!
    \pre The element must be in the heap.
  */
  void
  remove(const int index)
  {
    const int i = _indices[index];
#ifdef STLIB_DEBUG
    assert(i < _heapEnd - _queue.begin());
#endif
    --_heapEnd;
    Base::swap(_queue.begin() + i, _heapEnd);
    pushUpOrDown(i);
  }

  //! Insert the element in the heap.
  /*!
    \pre The element must not be in the heap.
  */
  void
  insert(const int index)
  {
#ifdef STLIB_DEBUG
    assert(_indices[index] >= _heapEnd - _queue.begin());
#endif
    Base::swap(_queue.begin() + _indices[index], _heapEnd);
    ++_heapEnd;
    pushUp(_heapEnd - 1 - getQueueBeginning());
  }

  void
  pushUp(ptrdiff_t child)
  {
    ptrdiff_t parent = (child - 1) / 2;

    while (child > 0 && _compare(_queue[child], _queue[parent])) {
      Base::swap(getQueueBeginning() + child, getQueueBeginning() + parent);
      child = parent;
      parent = (child - 1) / 2;
    }

#ifdef STLIB_DEBUG
    assert(isValid());
#endif
  }

  void
  pushDown(ptrdiff_t parent)
  {
    // CONTINUE REMOVE
#ifdef STLIB_DEBUG
    // The parent must not be less than own parent.
    if (parent != 0) {
      ptrdiff_t pp = (parent - 1) / 2;
      assert(! _compare(_queue[parent], _queue[pp]));
    }
#endif
    ptrdiff_t child = getSmallerChild(parent);

    while (child != 0 && _compare(_queue[child], _queue[parent])) {
      Base::swap(getQueueBeginning() + child, getQueueBeginning() + parent);
      parent = child;
      child = getSmallerChild(parent);
    }

#ifdef STLIB_DEBUG
    assert(isValid());
#endif
  }

  //! Update the position.
  void
  pushUpOrDown(ptrdiff_t n)
  {
    ptrdiff_t parent = (n - 1) / 2;
    if (n > 0 && _compare(_queue[n], _queue[parent])) {
      pushUp(n);
    }
    else {
      pushDown(n);
    }
  }

  //! Return the index of the smaller child or 0 if there are no children.
  ptrdiff_t
  getSmallerChild(ptrdiff_t parent) const
  {
    const ptrdiff_t size = getQueueEnd() - getQueueBeginning();
    ptrdiff_t child = 2 * parent + 1;
    // If there are no children.
    if (child >= size) {
      return 0;
    }
    // If the second child is smaller.
    if (child + 1 < size && _compare(_queue[child + 1], _queue[child])) {
      ++child;
    }
    return child;
  }

  //@}
};

} // namespace ads
}

#endif
