// -*- C++ -*-

/*!
  \file BinaryHeap.h
  \brief Implements a class for a sorted heap.
*/

#if !defined(__BinaryHeap_h__)
#define __BinaryHeap_h__

#include <vector>

namespace stlib
{
namespace shortest_paths
{

//! A heap.
template < typename T, class Compare >
class BinaryHeap :
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
  BinaryHeap(const BinaryHeap& heap);

  // Assignment operator not implemented.
  BinaryHeap&
  operator=(const BinaryHeap);

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
  BinaryHeap() :
    base_type() {}

  //! Construct and reserve memory for n elements.
  explicit
  BinaryHeap(size_type n) :
    base_type(n)
  {
    base_type::clear();
  }

  //! Destructor.
  virtual
  ~BinaryHeap() {}

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

  //! Return a const iterator to the end of the elements.
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

private:

  //! Swap two elements in the vector.
  void
  swap(iterator a, iterator b);

  //! Copy b into a.
  void
  copy(pointer a, pointer b);

  //! Set the heap pointers of the elements.
  void
  set_heap_ptrs();

  //! Return the index of the smaller child.
  int
  small_child(const int parent);

};


//
// Equality
//

//! Return true if x == y element-wise.
/*! \relates BinaryHeap */
template <typename T, class Compare>
inline
bool
operator==(const BinaryHeap<T, Compare>& x, const BinaryHeap<T, Compare>& y)
{
  return (static_cast< std::vector<T> >(x) ==
          static_cast< std::vector<T> >(y));
}

} // namespace shortest_paths
}

#define __BinaryHeap_ipp__
#include "stlib/shortest_paths/BinaryHeap.ipp"
#undef __BinaryHeap_ipp__

#endif
