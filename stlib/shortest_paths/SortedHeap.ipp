// -*- C++ -*-

#if !defined(__SortedHeap_ipp__)
#error This file is an implementation detail of the class SortedHeap.
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
SortedHeap<T, Compare>::
push(value_type x)
{
  if (size() < base_type::capacity()) {
    base_type::push_back(x);
    base_type::back()->heap_ptr() = &base_type::back();
  }
  else {
    base_type::push_back(x);
    set_heap_ptrs();
  }
  decrease(&*end() - 1);
}

template <typename T, class Compare>
inline
void
SortedHeap<T, Compare>::
pop()
{
  for (iterator i = begin(); i + 1 < end(); ++i) {
    copy(&*i, &*i + 1);
  }
  base_type::pop_back();
}


template <typename T, class Compare>
inline
void
SortedHeap<T, Compare>::
decrease(pointer ptr)
{
  while (ptr != &*begin() && _compare(*ptr, *(ptr - 1))) {
    std::swap(*(ptr - 1), *ptr);
    --ptr;
  }
}

template <typename T, class Compare>
inline
void
SortedHeap<T, Compare>::
swap(pointer a, pointer b)
{
  std::swap((*a)->heap_ptr(), (*b)->heap_ptr());
  std::swap(*a, *b);
}

template <typename T, class Compare>
inline
void
SortedHeap<T, Compare>::
copy(pointer a, pointer b)
{
  (*a)->heap_ptr() = (*b)->heap_ptr();
  *a = *b;
}

template <typename T, class Compare>
inline
void
SortedHeap<T, Compare>::
set_heap_ptrs()
{
  for (iterator i = begin(); i != end(); ++i) {
    (*i)->heap_ptr() = &*i;
  }
}

} // namespace shortest_paths
}
