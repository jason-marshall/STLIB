// -*- C++ -*-

/*!
  \file CellForwardSearch.h
  \brief A class for a cell array coupled with a forward search.
*/

#if !defined(__geom_CellForwardSearch_h__)
#define __geom_CellForwardSearch_h__

#include "stlib/geom/orq/CellSearch.h"

namespace stlib
{
namespace geom
{

//! Data structure for performing forward searching on records.
template<std::size_t N, typename _Location>
class ForwardSearch :
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
  // Member data
  //
private:

  _Location _location;
  mutable typename Base::iterator _current;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{

  //! Default constructor.
  ForwardSearch() :
    Base(),
    _location(),
    _current()
  {
  }

  //! Construct and reserve memory for n elements.
  explicit
  ForwardSearch(const typename Base::size_type size) :
    Base(size),
    _location(),
    _current()
  {
  }

  //! Construct from a range.
  template<typename _InputIterator>
  ForwardSearch(_InputIterator first, _InputIterator last) :
    Base(first, last),
    _location(),
    _current()
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
    return (sizeof(ForwardSearch) +
            Base::size() * sizeof(typename Base::value_type));
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Searching.
  // @{

  //! Initialize for a set of queries.
  void
  initialize()
  {
    _current = Base::begin();
  }

  //! Do a forward search to find the record.
  /*!
    \param x is a key in the final coordinate.
  */
  typename Base::iterator
  search(const Float x)
  {
    // Move the current pointer to the specified value.
    for (; _current != Base::end() && _location(*_current)[N - 1] < x;
         ++_current)
      ;
    return _current;
  }

  //! Do a forward search to find the record.
  /*!
    \param x is a key in the final coordinate.
  */
  typename Base::const_iterator
  search(const Float x) const
  {
    // Move the current pointer to the specified value.
    for (; _current != Base::end() &&
         _location(*_current)[N - 1] < x; ++_current)
      ;
    return _current;
  }
};









//! A cell array in combined with a forward search in the final coordinate.
/*!
  The cell array spans N-1 coordinates.  Record access
  is accomplished with array indexing in these coordinates and a
  forward search of a sorted vector in the final coordinate.
*/
template<std::size_t N, typename _Location>
class CellForwardSearch :
  public CellSearch<N, _Location, ForwardSearch<N, _Location> >
{
  //
  // Types.
  //
private:

  typedef CellSearch<N, _Location, ForwardSearch<N, _Location> > Base;

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
  CellForwardSearch(const typename Base::Point& delta,
                    const typename Base::BBox& domain) :
    Base(delta, domain)
  {
  }

  //! Construct from the Cartesian domain and a range of records.
  /*!
    Construct given the cell size, the Cartesian domain that
    contains the records and a range of records.

    \param delta the suggested size of a cell.
    \param domain the Cartesian domain that contains the records.
    \param first points to the begining of the range of records.
    \param last points to the end of the semi-open range.
  */
  CellForwardSearch(const typename Base::Point& delta,
                    const typename Base::BBox& domain,
                    typename Base::Record first, typename Base::Record last) :
    Base(delta, domain, first, last)
  {
  }

  //! Construct from a range of records.
  /*!
    Construct given the cell size a range of records. Compute an appropriate
    domain.

    \param delta the suggested size of a cell.
    \param first points to the begining of the range of records.
    \param last points to the end of the semi-open range.
  */
  CellForwardSearch(const typename Base::Point& delta,
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
/*! \relates CellForwardSearch */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const CellForwardSearch<N, _Location>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_CellForwardSearch_ipp__
#include "stlib/geom/orq/CellForwardSearch.ipp"
#undef __geom_CellForwardSearch_ipp__

#endif
