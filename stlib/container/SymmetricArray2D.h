// -*- C++ -*-

/*!
  \file SymmetricArray2D.h
  \brief Symmetric 2-D %array that may or may not store the diagonal elements.
*/

#if !defined(__container_SymmetricArray2D_h__)
#define __container_SymmetricArray2D_h__

#include "stlib/ext/vector.h"

namespace stlib
{
namespace container
{

USING_STLIB_EXT_VECTOR_IO_OPERATORS;

//! Symmetric 2-D %array that may or may not store the diagonal elements.
/*!
  \param _T The value type.
  \param _HasDiagonal Whether to store the diagonal elements. By default this
  is true.
*/
template<typename _T, bool _HasDiagonal = true>
class SymmetricArray2D
{
  //
  // Friends.
  //

  template<typename T_, bool HasDiagonal_>
  friend
  std::ostream&
  operator<<(std::ostream& out, const SymmetricArray2D<T_, HasDiagonal_>& x);

  template<typename T_, bool HasDiagonal_>
  friend
  std::istream&
  operator>>(std::istream& in, SymmetricArray2D<T_, HasDiagonal_>& x);

  //
  // Private types.
  //

private:

  typedef std::vector<_T> Container;

  //
  // Public Types.
  //

public:

  //! The value type.
  typedef typename Container::value_type value_type;
  //! Reference to the value type.
  typedef typename Container::reference reference;
  //! Constant reference to the value type.
  typedef typename Container::const_reference const_reference;
  //! Iterator in the container.
  typedef typename Container::iterator iterator;
  //! Constant iterator in the container.
  typedef typename Container::const_iterator const_iterator;
  //! The size type.
  typedef typename Container::size_type size_type;
  //! The pointer difference type.
  typedef typename Container::difference_type difference_type;
  //! Reverse iterator.
  typedef typename Container::reverse_iterator reverse_iterator;
  //! Constant reverse iterator.
  typedef typename Container::const_reverse_iterator const_reverse_iterator;
  //! A pointer to an %array element.
  typedef typename Container::pointer pointer;
  //! A pointer to a constant %array element.
  typedef typename Container::const_pointer const_pointer;

  //
  // Data.
  //

private:

  //! The number of rows is the same as the number of columns.
  size_type _extent;
  //! The non-null %array elements.
  Container _elements;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor, assignment operator, and destructor.
  */
  // @{
public:

  //! Default constructor. Empty data structure.
  SymmetricArray2D() :
    _extent(0),
    _elements()
  {
  }

  //! Size constructor. The extent is both the number of rows and the number of columns.
  SymmetricArray2D(const size_type extent) :
    _extent(extent),
    _elements(calculateSize(extent))
  {
  }

  //! Construct from the size and the fill value.
  SymmetricArray2D(const size_type extent, const value_type& value) :
    _extent(extent),
    _elements(calculateSize(extent), value)
  {
  }

  //! Swap with the argument.
  void
  swap(SymmetricArray2D& other)
  {
    std::swap(_extent, other._extent);
    _elements.swap(other._elements);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  //! Return the extent (both the number of rows and columns).
  size_type
  extent() const
  {
    return _extent;
  }

  //! Return the total number of non-null elements.
  size_type
  size() const
  {
    return _elements.size();
  }

  //! Return true if the total number of elements is zero.
  bool
  empty() const
  {
    return _elements.empty();
  }

  //! Return the size of the largest possible %array.
  size_type
  max_size() const
  {
    return _elements.max_size();
  }

  //! Return the memory size.
  size_type
  getMemoryUsage() const
  {
    return sizeof(size_type) + sizeof(Container) +
           _elements.size() * sizeof(value_type);
  }

  //! Return a const iterator to the first value.
  const_iterator
  begin() const
  {
    return _elements.begin();
  }

  //! Return a const iterator to one past the last value.
  const_iterator
  end() const
  {
    return _elements.end();
  }

  //! Return a const reverse iterator to the last value.
  const_reverse_iterator
  rbegin() const
  {
    return _elements.rbegin();
  }

  //! Return a const reverse iterator to one before the first value.
  const_reverse_iterator
  rend() const
  {
    return _elements.rend();
  }

  //! Return a const reference to the n_th overall element.
  const_reference
  operator[](const size_type n) const
  {
    return _elements[n];
  }

  //! Return a const reference to the specified element
  const_reference
  operator()(const size_type row, const size_type col) const
  {
    return _elements[index(row, col)];
  }

private:

  size_type
  index(size_type row, size_type col) const
  {
    if (row > col) {
      std::swap(row, col);
    }
    return index(row, col, std::integral_constant<bool, _HasDiagonal>());
  }

  size_type
  index(const size_type row, const size_type col,
        std::false_type /*HasDiagonal*/) const
  {
#ifdef STLIB_DEBUG
    assert(row < col && col < _extent);
#endif
    // Column-major order.
    return col * (col - 1) / 2 + row;
  }

  size_type
  index(const size_type row, const size_type col,
        std::true_type /*HasDiagonal*/) const
  {
#ifdef STLIB_DEBUG
    assert(row <= col && col < _extent);
#endif
    // Column-major order.
    return (col + 1) * col / 2 + row;
  }

  size_type
  calculateSize(const size_type extent) const
  {
    return calculateSize(extent, std::integral_constant<bool, _HasDiagonal>());
  }

  size_type
  calculateSize(const size_type extent, std::false_type /*HasDiagonal*/)
  const
  {
    return extent * (extent - 1) / 2;
  }

  size_type
  calculateSize(const size_type extent, std::true_type /*HasDiagonal*/)
  const
  {
    return (extent + 1) * extent / 2;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{
public:

  //! Return an iterator to the first value.
  iterator
  begin()
  {
    return _elements.begin();
  }

  //! Return an iterator to one past the last value.
  iterator
  end()
  {
    return _elements.end();
  }

  //! Clear the %array.
  void
  clear()
  {
    _extent = 0;
    _elements.resize(0);
  }

  //! Return a reference to the n_th overall element.
  reference
  operator[](const size_type n)
  {
    return _elements[n];
  }

  //! Return a reference to the specified element.
  reference
  operator()(const size_type row, const size_type col)
  {
    return _elements[index(row, col)];
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Equality.
  // @{
public:

  //! Return true if the arrays are equal.
  bool
  operator==(const SymmetricArray2D& x) const
  {
    return _extent == x._extent && _elements == x._elements;
  }

  //! Return true if the arrays are not equal.
  bool
  operator!=(const SymmetricArray2D& x) const
  {
    return ! operator==(x);
  }

  // @}
};

//
// File I/O.
//

//! Write a SymmetricArray2D in ascii format.
/*!
  \relates SymmetricArray2D

  Below is the file format.
  \verbatim
  extent
  size
  element_0
  element_1
  ... \endverbatim
*/
template<typename _T, bool _HasDiagonal>
inline
std::ostream&
operator<<(std::ostream& out, const SymmetricArray2D<_T, _HasDiagonal>& x)
{
  out << x.extent() << '\n' << x._elements;
  return out;
}

//! Read a SymmetricArray2D in ascii format.
/*!
  \relates SymmetricArray2D

  Below is the file format.
  \verbatim
  extent
  size
  element_0
  element_1
  ... \endverbatim
*/
template<typename _T, bool _HasDiagonal>
inline
std::istream&
operator>>(std::istream& in, SymmetricArray2D<_T, _HasDiagonal>& x)
{
  in >> x._extent >> x._elements;
#ifdef STLIB_DEBUG
  assert((x.extent() + _HasDiagonal) * (x.extent() - 1 + _HasDiagonal) / 2 ==
         x.size());
#endif
  return in;
}

} // namespace container
}

#endif
