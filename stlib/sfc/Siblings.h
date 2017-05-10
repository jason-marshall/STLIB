// -*- C++ -*-

#if !defined(__sfc_Siblings_h__)
#define __sfc_Siblings_h__

/*!
  \file
  \brief A group of siblings share a single parent.
*/

#include <iterator>

namespace stlib
{
namespace sfc
{


//! A group of siblings share a single parent.
template<std::size_t _Dimension>
class Siblings
{
  //
  // Friends.
  //

  template<std::size_t Dimension_>
  friend
  std::ostream&
  operator<<(std::ostream& out, const Siblings<Dimension_>& x);

  //
  // Types.
  //
public:

  // Types for STL compliance.

  //! The element type.
  typedef std::size_t value_type;

  //! A pointer to an element.
  typedef value_type* pointer;
  //! A pointer to a constant element.
  typedef const value_type* const_pointer;

  //! An iterator in the container.
  typedef value_type* iterator;
  //! A iterator on constant elements in the container.
  typedef const value_type* const_iterator;

  //! A reverse iterator.
  typedef std::reverse_iterator<iterator> reverse_iterator;
  //! A reverse iterator on constant elements.
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  //! A reference to an element.
  typedef value_type& reference;
  //! A reference to a constant element.
  typedef const value_type& const_reference;

  //! The size type.
  typedef std::size_t size_type;
  //! Pointer difference type.
  typedef std::ptrdiff_t difference_type;

  //
  // Constants.
  //
public:

  //! The number of children for a parent.
  BOOST_STATIC_CONSTEXPR std::size_t NumChildren = std::size_t(1) << _Dimension;

  //
  // Member data.
  //
private:

  //! The number of siblings.
  std::size_t _size;
  //! The indices of the siblings.
  /*! For the sake of efficiency, use \c std::array instead of \c std::vector.
   That way we don't need to check the capacity when adding elements. */
  std::array<std::size_t, NumChildren> _data;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  //! Make on empty group of siblings.
  Siblings();

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return true if the range is empty.
  bool
  empty() const
  {
    return _size == 0;
  }

  //! Return the size (number of elements) in the range.
  size_type
  size() const
  {
    return _size;
  }

  //! Return the maximum size of the range.
  size_type
  max_size() const
  {
    return NumChildren;
  }

  //! Return a const pointer to the beginning of the data.
  const_pointer
  data() const
  {
    return &_data[0];
  }

  //! Return a const iterator to the first value.
  const_iterator
  begin() const
  {
    return data();
  }

  //! Return a const iterator to one past the last value.
  const_iterator
  end() const
  {
    return begin() + size();
  }

  //! Return a const reverse iterator to the end of the sequence.
  const_reverse_iterator
  rbegin() const
  {
    return const_reverse_iterator(end());
  }

  //! Return a const reverse iterator to the beginning of the sequence.
  const_reverse_iterator
  rend() const
  {
    return const_reverse_iterator(begin());
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Return a pointer to the beginning of the data.
  pointer
  data()
  {
    return &_data[0];
  }

  //! Return an iterator to the first value.
  iterator
  begin()
  {
    return data();
  }

  //! Return an iterator to one past the last value.
  iterator
  end()
  {
    return begin() + _size;
  }

  //! Return a reverse iterator to the end of the sequence.
  reverse_iterator
  rbegin()
  {
    return reverse_iterator(end());
  }

  //! Return a reverse iterator to the beginning of the sequence.
  reverse_iterator
  rend()
  {
    return reverse_iterator(begin());
  }

  //! Add a new element at the end of the container.
  void
  push_back(const_reference x);

  //! Remove the last element from the container.
  void
  pop_back();

  //! Remove all elements.
  void
  clear()
  {
    _size = 0;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Array indexing.
  //@{
public:

  //! Container indexing.
  const_reference
  operator[](const size_type n) const;

  //! Container indexing.
  reference
  operator[](const size_type n);

  //@}
};


//! Write the sequence of indices.
template<std::size_t _Dimension>
inline
std::ostream&
operator<<(std::ostream& out, const Siblings<_Dimension>& x)
{
  if (x._size) {
    out << x._data[0];
  }
  for (std::size_t i = 1; i < x._size; ++i) {
    out << ' ' << x._data[i];
  }
  return out;
}


} // namespace sfc
}

#define __sfc_Siblings_tcc__
#include "stlib/sfc/Siblings.tcc"
#undef __sfc_Siblings_tcc__

#endif
