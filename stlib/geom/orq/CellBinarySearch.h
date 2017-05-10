// -*- C++ -*-

/*!
  \file CellBinarySearch.h
  \brief A class for a cell array coupled with a binary search.
*/

#if !defined(__geom_CellBinarySearch_h__)
#define __geom_CellBinarySearch_h__

#include "stlib/geom/orq/CellSearch.h"

namespace stlib
{
namespace geom
{

//! Binary search of records in the final coordinate.
template<std::size_t N, typename _Location>
class BinarySearch :
  public Search<N, _Location>
{
  //
  // Types.
  //
private:

  typedef Search<N, _Location> Base;

public:

  //! The floating-point number type.
  typedef typename Orq<N, _Location>::Float Float;

  //
  // Functors.
  //
private:

  //! Less than comparison in the final coordinate for a record and a key.
  class LessThanCompareRecordAndKey :
    public std::binary_function<typename Base::Record, Float, bool>
  {
  private:

    _Location _f;

  public:

    //! Less than comparison in the final coordinate.
    bool
    operator()(const typename Base::Record record, const Float key) const
    {
      return _f(record)[N - 1] < key;
    }
  };

  // Compare a record and a key.
  LessThanCompareRecordAndKey _compare;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{

  //! Default constructor.
  BinarySearch() :
    Base(),
    _compare()
  {
  }

  //! Construct and reserve memory for n elements.
  explicit
  BinarySearch(const typename Base::size_type size) :
    Base(size),
    _compare() {}

  //! Construct from a range.
  template <typename _InputIterator>
  BinarySearch(_InputIterator first, _InputIterator last) :
    Base(first, last),
    _compare()
  {
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Memory usage.
  // @{

  //! Return the memory usage.
  typename Base::size_type
  getMemoryUsage() const
  {
    return (sizeof(BinarySearch) +
            Base::size() * sizeof(typename Base::value_type));
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Searching.
  // @{

  //! Do a binary search to find the record.
  typename Base::iterator
  search(const Float x)
  {
    return std::lower_bound(Base::begin(), Base::end(), x, _compare);
  }

  //! Do a binary search to find the record.
  typename Base::const_iterator
  search(const Float x) const
  {
    return std::lower_bound(Base::begin(), Base::end(), x, _compare);
  }

  // @}
};










//! A cell array in combined with a binary search in the final coordinate.
/*!
  The cell array spans N-1 coordinates.  Record access
  is accomplished with array indexing in these coordinates and a
  binary search of a sorted vector in the final coordinate.
*/
template<std::size_t N, typename _Location>
class CellBinarySearch :
  public CellSearch<N, _Location, BinarySearch<N, _Location> >
{
  //
  // Types.
  //
private:

  typedef CellSearch<N, _Location, BinarySearch<N, _Location> > Base;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors.
  //@{

  //! Construct from the size of a cell and a Cartesian domain.
  /*!
    Construct given the cell size and the Cartesian domain that
    contains the records.

    \param delta the suggested size of a cell.
    \param domain the Cartesian domain that contains the records.
  */
  CellBinarySearch(const typename Base::Point& delta,
                   const typename Base::BBox& domain) :
    Base(delta, domain)
  {
  }

  //! Construct from the Cartesian domain and a range of records.
  /*!
    Construct given the cell size, the Cartesian domain that
    contains the records and a range of records.

    \param delta is the suggested size of a cell.
    \param domain is the Cartesian domain that contains the records.
    \param first points to the begining of the range of records.
    \param last points to the end of the semi-open range.
  */
  CellBinarySearch(const typename Base::Point& delta,
                   const typename Base::BBox& domain,
                   typename Base::Record first, typename Base::Record last) :
    Base(delta, domain, first, last)
  {
  }

  //! Construct from a range of records.
  /*!
    Construct given the cell size a range of records. Compute an appropriate
    domain.

    \param delta is the suggested size of a cell.
    \param first points to the begining of the range of records.
    \param last points to the end of the semi-open range.
  */
  CellBinarySearch(const typename Base::Point& delta,
                   typename Base::Record first, typename Base::Record last) :
    Base(delta, first, last)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Window queries.
  //@{

  //! Get the records in the window.  Return the # of records inside.
  template<typename _OutputIterator>
  std::size_t
  computeWindowQuery(_OutputIterator iter,
                     const typename Base::BBox& window) const;

  //@}
};


//
// File I/O
//


//! Write to a file stream.
/*! \relates CellBinarySearch */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const CellBinarySearch<N, _Location>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_CellBinarySearch_ipp__
#include "stlib/geom/orq/CellBinarySearch.ipp"
#undef __geom_CellBinarySearch_ipp__

#endif
