// -*- C++ -*-

// CONTINUE: Add constructors that compute the domain to the rest of the
// classes.
// CONTINUE: I can use a 1-D array of cells instead of an N-D Array.  Just
// convert the multi-index to a single index.
/*!
  \file CellArray.h
  \brief A class for a cell array in N-D.
*/

#if !defined(__geom_CellArray_h__)
#define __geom_CellArray_h__

#include "stlib/geom/orq/CellArrayBase.h"

#include "stlib/container/MultiArray.h"

namespace stlib
{
namespace geom
{

//! A cell array in N-D.
/*!
  A dense cell array in N-D.
*/
template<std::size_t N, typename _Location>
class CellArray :
  public CellArrayBase<N, _Location>
{
  //
  // Types.
  //
private:

  typedef CellArrayBase<N, _Location> Base;
  typedef container::MultiArray<typename Base::Cell, N> DenseArray;

  //
  // Data
  //
private:

  // The array of cells.
  DenseArray _cellArray;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{

  //! Construct from the size of a cell and a Cartesian domain.
  /*!
    Construct a cell array given the grid size and the Cartesian domain
    that contains the records.

    \param delta the suggested size of a cell.
    \param domain the Cartesian domain that contains the records.
  */
  CellArray(const typename Base::Point& delta,
            const typename Base::BBox& domain) :
    Base(delta, domain),
    _cellArray(Base::getExtents())
  {
  }

  //! Construct from a range of records.
  /*!
    Construct a cell grid given the array size, the Cartesian domain
    and a range of records.

    \pre The records must lie in the specified domain.
    \pre There must be a non-zero number of records.

    \param delta The suggested size of a cell.
    \param domain The Cartesian domain that contains the records.
    \param first The first record.
    \param last The last record.

    \note This function assumes that the records are iterators.
  */
  CellArray(const typename Base::Point& delta,
            const typename Base::BBox& domain,
            typename Base::Record first, typename Base::Record last) :
    Base(delta, domain),
    _cellArray(Base::getExtents())
  {
    // Insert the grid elements in the range.
    insert(first, last);
  }

  //! Construct from the cell size and a range of records.
  /*!
    \pre There must be a non-zero number of records.

    \param delta The suggested size of a cell.
    \param first The first record.
    \param last The last record.

    \c first and \c last must be forward iterators because there is one pass
    to compute an appropriate domain and one pass to insert the records.

    \note This function assumes that the records are iterators.
  */
  CellArray(const typename Base::Point& delta, typename Base::Record first,
            typename Base::Record last) :
    Base(delta, first, last),
    _cellArray(Base::getExtents())
  {
    // Insert the grid elements in the range.
    insert(first, last);
  }

  //! Construct from the number of cells and a range of records.
  /*!
    \pre There must be a non-zero number of records.

    \param numberOfCells the suggested number of cells.
    \param first The first record.
    \param last The last record.

    \c first and \c last must be forward iterators because there is one pass
    to compute an appropriate domain and one pass to insert the records.

    \note This function assumes that the records are iterators.
  */
  CellArray(const std::size_t numberOfCells, typename Base::Record first,
            typename Base::Record last) :
    Base(numberOfCells, first, last),
    _cellArray(Base::getExtents())
  {
    // Insert the grid elements in the range.
    insert(first, last);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Insert/Erase records.
  // @{

  //! Insert a single record.
  void
  insert(const typename Base::Record record)
  {
    typename Base::Cell& b = (*this)(Base::_location(record));
    b.push_back(record);
    ++Base::_size;
  }

  //! Insert a number of records.
  /*!
    \param first The first record.
    \param last The last record.
    \note This function assumes that the records are iterators.
  */
  void
  insert(typename Base::Record first, typename Base::Record last)
  {
    while (first != last) {
      insert(first);
      ++first;
    }
  }

  //! Clear all records.
  void
  clear()
  {
    // Clear each of the cells.
    for (typename DenseArray::iterator i = _cellArray.begin();
         i != _cellArray.end(); ++i) {
      i->clear();
    }
    // There are now no records.
    Base::_size = 0;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Memory usage.
  // @{

  //! Return the memory usage.
  std::size_t
  getMemoryUsage() const;

  // @}
  //--------------------------------------------------------------------------
  //! \name Window Queries.
  // @{

  //! Get the records in the window.  Return the # of records inside.
  template<typename _OutputIterator>
  std::size_t
  computeWindowQuery(_OutputIterator iter,
                     const typename Base::BBox& window) const;

  // @}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  // @{

  //! Print the records.
  void
  put(std::ostream& out) const;

  // @}

private:

  //
  // Cell indexing.
  //

  //! Return a reference to the cell that would hold the point.
  /*!
    Indexing by location.  Return a reference to a cell.
    The multi-key must be in the domain of the cell array.
  */
  template<typename _AnyMultiKeyType>
  typename Base::Cell&
  operator()(const _AnyMultiKeyType& multiKey)
  {
    typename Base::IndexList mi;
    Base::convertMultiKeyToIndices(multiKey, &mi);
#ifdef STLIB_DEBUG
    // Check that the cell exists.  If does not, then the record must be
    // outside the domain spanned by the cells.  Note that if the debugging
    // code is turned on in the array class, it would also catch this error.
    assert(isIn(_cellArray.range(), mi));
#endif
    return _cellArray(mi);
  }

};


//
// File I/O
//


//! Write to a file stream.
/*! \relates CellArray */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const CellArray<N, _Location>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_CellArray_ipp__
#include "stlib/geom/orq/CellArray.ipp"
#undef __geom_CellArray_ipp__

#endif
