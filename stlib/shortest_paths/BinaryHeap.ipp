// -*- C++ -*-

#if !defined(__BinaryHeap_ipp__)
#error This file is an implementation detail of the class BinaryHeap.
#endif

namespace stlib
{
namespace shortest_paths
{

//
// Manipulators
//

template <typename T, class Compare>
inline
void
BinaryHeap<T, Compare>::
push(value_type x)
{
  if (size() < base_type::capacity()) {
    base_type::push_back(x);
    base_type::back()->set_heap_ptr(&base_type::back());
  }
  else {
    base_type::push_back(x);
    set_heap_ptrs();
  }
  decrease(&base_type::back());
}

template <typename T, class Compare>
inline
void
BinaryHeap<T, Compare>::
pop()
{
  // Store and erase the last element.
  value_type tmp(base_type::back());
  base_type::pop_back();

  // Adjust the heap.
  int parent = 0;
  int child = small_child(parent);
  while (child >= 0 && _compare(*(begin() + child), tmp)) {
    copy(&*begin() + parent, &*begin() + child);
    parent = child;
    child = small_child(parent);
  }

  // Insert the last element.
  copy(&*begin() + parent, &tmp);
}


template <typename T, class Compare>
inline
void
BinaryHeap<T, Compare>::
decrease(pointer iter)
{
  int child = iter - &*begin();
  int parent = (child - 1) / 2;

  while (child > 0 && _compare(*(begin() + child), *(begin() + parent))) {
    swap(begin() + child, begin() + parent);
    child = parent;
    parent = (child - 1) / 2;
  }
}

template <typename T, class Compare>
inline
void
BinaryHeap<T, Compare>::
swap(iterator a, iterator b)
{
  std::swap((*a)->heap_ptr(), (*b)->heap_ptr());
  std::swap(*a, *b);
}

template <typename T, class Compare>
inline
void
BinaryHeap<T, Compare>::
copy(pointer a, pointer b)
{
  *a = *b;
  (*a)->set_heap_ptr(a);
}

template <typename T, class Compare>
inline
void
BinaryHeap<T, Compare>::
set_heap_ptrs()
{
  for (iterator i = begin(); i != end(); ++i) {
    (*i)->set_heap_ptr(&*i);
  }
}

template <typename T, class Compare>
inline
int
BinaryHeap<T, Compare>::
small_child(const int parent)
{
  int child = 2 * parent + 1;
  if (child + 1 < static_cast<int>(size()) &&
      _compare(*(begin() + child + 1), *(begin() + child))) {
    ++child;
  }
  if (child < static_cast<int>(size())) {
    return child;
  }
  return -1;
}

} // namespace shortest_paths
}
