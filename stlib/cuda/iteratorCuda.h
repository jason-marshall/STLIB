// -*- C++ -*-

#if !defined(__cuda_iteratorCuda_h__)
#define __cuda_iteratorCuda_h__

namespace std
{

//!  Marking input iterators.
struct input_iterator_tag {};
//!  Marking output iterators.
struct output_iterator_tag {};
//! Forward iterators support a superset of input iterator operations.
struct forward_iterator_tag : public input_iterator_tag {};
//! Bidirectional iterators support a superset of forward iterator operations.
struct bidirectional_iterator_tag : public forward_iterator_tag {};
//! Random-access iterators support a superset of bidirectional iterator operations.
struct random_access_iterator_tag : public bidirectional_iterator_tag {};


//! Common %iterator class.
template<typename _Category, typename _T, typename _Distance = ptrdiff_t,
         typename _Pointer = _T*, typename _Reference = _T&>
struct iterator {
  //! One of the iterator_tags tag types.
  typedef _Category iterator_category;
  //! The type "pointed to" by the iterator.
  typedef _T value_type;
  //! Distance between iterators is represented as this type.
  typedef _Distance difference_type;
  //! This type represents a pointer-to-value_type.
  typedef _Pointer pointer;
  //! This type represents a reference-to-value_type.
  typedef _Reference reference;
};

template<typename _Iterator>
struct iterator_traits {
  typedef typename _Iterator::iterator_category iterator_category;
  typedef typename _Iterator::value_type value_type;
  typedef typename _Iterator::difference_type difference_type;
  typedef typename _Iterator::pointer pointer;
  typedef typename _Iterator::reference reference;
};

template<typename _T>
struct iterator_traits<_T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef _T value_type;
  typedef ptrdiff_t difference_type;
  typedef _T* pointer;
  typedef _T& reference;
};

template<typename _T>
struct iterator_traits<const _T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef _T value_type;
  typedef ptrdiff_t difference_type;
  typedef const _T* pointer;
  typedef const _T& reference;
};

} // namespace std

#endif
