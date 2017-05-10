// -*- C++ -*-

/*!
  \file SortedHeap.h
  \brief Implements a class for a sorted heap.

  Not efficient.  Use only for testing correctness of algorithms which require
  a heap data structure.
*/

#if !defined(__SortedHeap_h__)
#define __SortedHeap_h__

#include <vector>
#include <functional>
#include <algorithm>

namespace stlib
{
namespace shortest_paths
{

//! A heap.
template < typename T, class Compare = std::less<T> >
class SortedHeap :
  public std::vector<T>
{
private:

  typedef std::vector<T> base_type;

public:

  //! Value type.
  typedef typename base_type::value_type value_type;
  //! Pointer to the value type.
  typedef typename base_type::pointer pointer;
  //! Const pointer to the value type.
  typedef typename base_type::const_pointer const_pointer;
  //! Iterator to the value type.
  typedef typename base_type::iterator iterator;
  //! Const iterator to the value type.
  typedef typename base_type::const_iterator const_iterator;
  //! Reference to the value type.
  typedef typename base_type::reference reference;
  //! Const reference to the value type.
  typedef typename base_type::const_reference const_reference;
  //! The size type.
  typedef typename base_type::size_type size_type;
  //! Pointer difference type.
  typedef typename base_type::difference_type difference_type;

private:

  //
  // Not implemented.
  //

  // Copy constructor not implemented.
  SortedHeap(const SortedHeap<T, Compare>& heap);

  // Assignment operator not implemented.
  const SortedHeap<T, Compare>& operator=(const SortedHeap<T, Compare>& other);

private:

  //
  // Member data.
  //

  Compare _compare;

public:

  //------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{

  //! Default constructor.
  SortedHeap() :
    base_type() {}

  //! Construct and reserve memory for n elements.
  explicit
  SortedHeap(size_type n) :
    base_type(n)
  {
    base_type::clear();
  }

  //! Destructor.
  virtual
  ~SortedHeap() {}

  //@}
  //------------------------------------------------------------------------
  //! \name Accessors.
  //@{

  //! Return the number of elements in the heap.
  size_type
  size() const
  {
    return base_type::size();
  }

  //! Return a const iterator to the beginning of the elements.
  const_iterator
  begin() const
  {
    return base_type::begin();
  }

  //! Return a const iterator to the end of the elements.
  const_iterator
  end() const
  {
    return base_type::end();
  }

  //! Return the element at the top of the heap.
  const_reference
  top() const
  {
    return base_type::front();
  }

  //@}
  //------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  //! Return an iterator to the beginning of the elements.
  iterator
  begin()
  {
    return base_type::begin();
  }

  //! Return an iterator to the end of the elements.
  iterator
  end()
  {
    return base_type::end();
  }

  //! Push an element onto the heap.
  void
  push(value_type x);

  //! Pop an element off the heap.
  void
  pop();

  //! The key of an element has decreased.  Adjust its position in the heap.
  void
  decrease(pointer iter);

  //@}

protected:

  //! Swap two elements in the vector.
  void
  swap(pointer a, pointer b);

  //! Copy b into a.
  void
  copy(pointer a, pointer b);

  //! Set the heap pointers of the elements.
  void
  set_heap_ptrs();

};


//
// Equality
//

//! Return true if x == y element-wise.
/*! \relates SortedHeap */
template < typename T, class Compare >
inline
bool
operator==(const SortedHeap<T, Compare>& x, const SortedHeap<T, Compare>& y)
{
  return (static_cast< std::vector<T> >(x) ==
          static_cast< std::vector<T> >(y));
}

} // namespace shortest_paths
}

#define __SortedHeap_ipp__
#include "stlib/shortest_paths/SortedHeap.ipp"
#undef __SortedHeap_ipp__

#endif
