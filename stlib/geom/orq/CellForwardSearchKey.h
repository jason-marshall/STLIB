// -*- C++ -*-

/*!
  \file CellForwardSearchKey.h
  \brief A class for a sparse cell array in 3-D.
*/

#if !defined(__geom_CellForwardSearchKey_h__)
#define __geom_CellForwardSearchKey_h__

#include "stlib/geom/orq/CellSearch.h"

namespace stlib
{
namespace geom
{

//! Data structure for performing forward searching on the keys of records.
template<std::size_t N, typename _Location>
class ForwardSearchKey :
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

  //! The key container.
  typedef std::vector<Float> KeyContainer;
  //! An iterator on const keys in the key container.
  typedef typename KeyContainer::const_iterator KeyConstIterator;

  //
  // Member data
  //
private:

  //! The last coordinate of the record's multi-key
  KeyContainer _lastKeys;
  //! Index in the vector of record pointers.
  mutable std::size_t _current;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors.
  //@{

  //! Default constructor.
  ForwardSearchKey() :
    Base(),
    _lastKeys(),
    _current(0)
  {
  }

  //! Construct and reserve memory for n elements.
  explicit
  ForwardSearchKey(const typename Base::size_type size) :
    Base(size),
    _lastKeys(),
    _current(0)
  {
  }

  //! Construct from a range.
  template<typename _InputIterator>
  ForwardSearchKey(_InputIterator first, _InputIterator last) :
    Base(first, last),
    _lastKeys(),
    _current(0)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{

  const KeyContainer&
  getLastKeys() const
  {
    return _lastKeys;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Sorting and searching.
  //@{

  //! Sort the record pointers in the last coordinate.
  void
  sort()
  {
    // Sort the records.
    Base::sort();
    // Copy the last keys.
    _lastKeys.clear();
    _lastKeys.reserve(Base::size());
    _Location location;
    for (typename Base::iterator i = Base::begin(); i != Base::end(); ++i) {
      _lastKeys.push_back(location(*i)[N - 1]);
    }
  }

  //! Initialize for a set of queries.
  void
  initialize() const
  {
    _current = 0;
  }

  //! Do a forward search to find the index of the record.
  std::size_t
  search(const Float x) const
  {
    KeyConstIterator i = _lastKeys.begin() + _current;
    KeyConstIterator iEnd = _lastKeys.end();
    while (i != iEnd && *i < x) {
      ++i;
    }
    _current = i - _lastKeys.begin();
    return _current;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Memory usage.
  //@{

  //! Return the memory usage.
  typename Base::size_type
  getMemoryUsage() const
  {
    return (sizeof(ForwardSearchKey)
            + Base::size() * sizeof(typename Base::value_type)
            + _lastKeys.size() * sizeof(Float));
  }

  //@}
};








//! A cell array in combined with a forward search on keys in the final coordinate.
/*!
  The cell array spans N-1 coordinates.  Record access
  is accomplished with array indexing in these coordinates and a
  forward search of a sorted vector in the final coordinate.
*/
template<std::size_t N, typename _Location>
class CellForwardSearchKey :
  public CellSearch<N, _Location, ForwardSearchKey<N, _Location> >
{
  //
  // Types.
  //
private:

  typedef CellSearch<N, _Location, ForwardSearchKey<N, _Location> > Base;

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
  CellForwardSearchKey(const typename Base::Point& delta,
                       const typename Base::BBox& domain) :
    Base(delta, domain)
  {
  }

  //! Construct from a domain and a range of records.
  /*!
    Construct given the cell size, the Cartesian domain that
    contains the records and a range of records.

    \param delta the suggested size of a cell.
    \param domain the Cartesian domain that contains the records.
    \param first points to the begining of the range of records.
    \param last points to the end of the semi-open range.
  */
  CellForwardSearchKey(const typename Base::Point& delta,
                       const typename Base::BBox& domain,
                       typename Base::Record first,
                       typename Base::Record last) :
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
  CellForwardSearchKey(const typename Base::Point& delta,
                       typename Base::Record first,
                       typename Base::Record last) :
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


//! Write to a file stream.
/*! \relates CellForwardSearchKey */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const CellForwardSearchKey<N, _Location>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_CellForwardSearchKey_ipp__
#include "stlib/geom/orq/CellForwardSearchKey.ipp"
#undef __geom_CellForwardSearchKey_ipp__

#endif
