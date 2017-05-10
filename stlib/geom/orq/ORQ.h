// -*- C++ -*-

/*!
  \file geom/orq/ORQ.h
  \brief A base class for a data structure for doing orthogonal range queries
  in N-D.
*/

#if !defined(__geom_ORQ_h__)
#define __geom_ORQ_h__

#include "stlib/geom/kernel/BBox.h"

#include <limits>

namespace stlib
{
namespace geom
{

// CONTINUE REMOVE
#if 0
//! Base class for an orthogonal range query data structure in N-D.
/*!
  \param N The space dimension.
  \param _Record The record type.  All the derived data structures hold
  records.  The record type is most likely a pointer to a class or an iterator
  into a container.
  \param _MultiKey An N-tuple of the key type.  The multi-key type must be
  subscriptable.  For example, the multi-key type could be
  \c std::array<double,3> or \c double*.
  \param _Key The number type.
  \param _MultiKeyAccessor A functor that takes the record type as an argument
  and returns the multi-key for the record.  If possible, it should return
  a constant reference to the multi-key.
*/
template < std::size_t N,
           typename _Record,
           typename _MultiKey,
           typename _Key,
           typename _MultiKeyAccessor >
class ORQ
{
public:

  //
  // Public types.
  //

  //! The record type.
  typedef _Record Record;
  //! The multi-key type.
  typedef _MultiKey MultiKey;
  //! The key type.
  typedef _Key Key;
  //! The multy-key accessor.
  typedef _MultiKeyAccessor MultiKeyAccessor;

  //! A Cartesian point.
  typedef std::array<Key, N> Point;
  //! Bounding box.
  typedef geom::BBox<Key, N> BBox;

private:

  //! Number of records in the data structure.
  std::size_t _size;

protected:

  //! The multi-key accessor.
  MultiKeyAccessor _multiKeyAccessor;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{

  //! Default constructor.
  ORQ() :
    _size(0),
    _multiKeyAccessor()
  {
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Assignment operator.
  // @{

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{

  //! Return the number of records.
  std::size_t
  getSize() const
  {
    return _size;
  }

  //! Return true if the grid is empty.
  bool
  isEmpty() const
  {
    return _size == 0;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  // @{

  //! Print the number of records.
  void
  put(std::ostream& out) const
  {
    out << getSize();
  }

  // @}

protected:

  //! Increment the number of records.
  void
  incrementSize()
  {
    ++_size;
  }

  //! Decrement the number of records.
  void
  decrementSize()
  {
    --_size;
  }

  //! Set the number of records.
  void
  setSize(const std::size_t size)
  {
    _size = size;
  }

  //! Determine an appropriate domain to contain the records.
  /*!
    \param first The first record.
    \param last The last record.
    \param domain The domain that contains the records.

    \note This function assumes that the records are iterators.
  */
  template<class InputIterator>
  void
  computeDomain(InputIterator first, InputIterator last,
                BBox* domain) const;

};


//! Write to a file stream.
/*! \relates ORQ */
template < std::size_t N, typename _Record, typename _MultiKey, typename _Key,
           typename _MultiKeyAccessor >
inline
std::ostream&
operator<<(std::ostream& out,
           const ORQ<N, _Record, _MultiKey, _Key, _MultiKeyAccessor>& x)
{
  x.put(out);
  return out;
}


// Determine an appropriate domain to contain the records.
template < std::size_t N,
           typename _Record,
           typename _MultiKey,
           typename _Key,
           typename _MultiKeyAccessor >
template<class InputIterator>
void
ORQ<N, _Record, _MultiKey, _Key, _MultiKeyAccessor>::
computeDomain(InputIterator first, InputIterator last, BBox* domain) const
{
  // Check the special case that there are no records.
  if (first == last) {
    // Unit box.
    domain->lower = ext::filled_array<MultiKey>(0);
    domain->upper = ext::filled_array<MultiKey>(1);
    return;
  }
  // The number of records must be non-zero.
  assert(first != last);
  // Compute the domain.
  domain->lower = _multiKeyAccessor(first);
  domain->upper = _multiKeyAccessor(first);
  for (++first; first != last; ++first) {
    domain->add(_multiKeyAccessor(first));
  }
  // Because the upper sides are open, expand the domain.
  const Key Epsilon = std::sqrt(std::numeric_limits<Key>::epsilon());
  Point diagonal = domain->upper;
  diagonal -= domain->lower;
  Point upperCorner = domain->upper;
  for (std::size_t n = 0; n != N; ++n) {
    if (diagonal[n] == 0 && upperCorner[n] == 0) {
      upperCorner[n] = Epsilon;
    }
    else {
      upperCorner[n] += Epsilon *
                        std::max(diagonal[n], std::abs(upperCorner[n]));
    }
  }
  domain->upper = upperCorner;
}
#endif


//! Base class for an orthogonal range query data structure in N-D.
/*!
  \param SpaceD The space dimension.
  \param _Location A functor that takes the record type as an argument
  and returns the multi-key for the record.

  The location functor overloads the function call operator to convert
  a record to a Cartesian location. It must define the types \c
  argument_type, which is the record type, and \c result_type, which
  is the point type. The record type is most likely a pointer to a
  class or an iterator into a container. All the derived data
  structures hold records. The point type must be \c
  std::array<T,SpaceD> for some floating-point number type \c _T.
  For the sake of efficiency, it is better if the functor returns a
  const reference to the point. This will cut down on copy constructor
  calls.
*/
template<std::size_t SpaceD, typename _Location>
class Orq
{
  //
  // Constants.
  //
public:
  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t N = SpaceD;

  //
  // Types.
  //
public:
  //! The multi-key accessor.
  typedef _Location Location;
  //! The record type.
  typedef typename Location::argument_type Record;
  //! A Cartesian point. Note that \c result_type could be a reference or const reference.
  typedef typename
  std::remove_const<typename std::remove_reference<typename Location::result_type>::type>::type
  Point;
  //! The key type.
  typedef typename Point::value_type Float;
  //! Bounding box.
  typedef geom::BBox<Float, N> BBox;

  //
  // Member data.
  //
protected:
  //! Number of records in the data structure.
  std::size_t _size;
  //! The multi-key accessor.
  Location _location;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{

  //! Default constructor.
  Orq() :
    _size(0),
    _location()
  {
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{

  //! Return the number of records.
  std::size_t
  size() const
  {
    return _size;
  }

  //! Return true if there are no records.
  bool
  empty() const
  {
    return _size == 0;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  // @{

  //! Print the number of records.
  void
  put(std::ostream& out) const
  {
    out << size();
  }

  // @}

protected:

  //! Determine an appropriate domain to contain the records.
  /*!
    \param first The first record.
    \param last The last record.
    \return The domain that contains the records.

    \note This function assumes that the records are iterators.
  */
  BBox
  computeDomain(Record first, Record last) const;
};


//! Write to a file stream.
/*! \relates Orq */
template<std::size_t SpaceD, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const Orq<SpaceD, _Location>& x)
{
  x.put(out);
  return out;
}


// Determine an appropriate domain to contain the records.
template<std::size_t SpaceD, typename _Location>
typename Orq<SpaceD, _Location>::BBox
Orq<SpaceD, _Location>::
computeDomain(Record first, Record last) const
{
  BBox domain;
  // Check the special case that there are no records.
  if (first == last) {
    // Unit box.
    domain.lower = ext::filled_array<Point>(0);
    domain.upper = ext::filled_array<Point>(1);
    return domain;
  }
  // Compute the domain.
  domain.lower = _location(first);
  domain.upper = _location(first);
  for (++first; first != last; ++first) {
    domain += _location(first);
  }
  // Because the upper sides are open, expand the domain.
  const Float Epsilon = std::sqrt(std::numeric_limits<Float>::epsilon());
  Point diagonal = domain.upper;
  diagonal -= domain.lower;
  Point upperCorner = domain.upper;
  for (std::size_t n = 0; n != N; ++n) {
    if (diagonal[n] == 0 && upperCorner[n] == 0) {
      upperCorner[n] = Epsilon;
    }
    else {
      upperCorner[n] += Epsilon *
                        std::max(diagonal[n], std::abs(upperCorner[n]));
    }
  }
  domain.upper = upperCorner;
  return domain;
}


} // namespace geom
}

#endif
