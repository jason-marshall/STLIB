// -*- C++ -*-

/*!
  \file CellSearch.h
  \brief A base class for a cell array with searching in N-D.
*/

#if !defined(__geom_CellSearch_h__)
#define __geom_CellSearch_h__

#include "stlib/geom/orq/ORQ.h"

#include "stlib/container/MultiArray.h"

#include <vector>
#include <algorithm>

namespace stlib
{
namespace geom
{

//! Base class for a search structure in the final coordinate.
template<std::size_t N, typename _Location>
class Search :
  public std::vector<typename Orq<N, _Location>::Record>
{
  //
  // Types.
  //
private:

  typedef std::vector<typename Orq<N, _Location>::Record> Base;

public:

  //! The record type.
  typedef typename Orq<N, _Location>::Record Record;

private:

  //
  // Functors.
  //

  //! Less than comparison in the final coordinate.
  class LessThanCompare :
    public std::binary_function<Record, Record, bool>
  {
  private:

    _Location _f;

  public:

    //! Less than comparison in the final coordinate.
    bool
    operator()(const Record x, const Record y) const
    {
      return _f(x)[N - 1] < _f(y)[N - 1];
    }
  };

public:

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{

  //! Default constructor.
  Search() :
    Base()
  {
  }

  //! Construct and reserve memory for n elements.
  explicit
  Search(const typename Base::size_type size) :
    Base()
  {
    reserve(size);
  }

  //! Construct from a range.
  template<typename _InputIterator>
  Search(_InputIterator first, _InputIterator last) :
    Base(first, last)
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
    return (sizeof(Search) +
            Base::size() * sizeof(typename Base::value_type));
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Sorting and searching.
  // @{

  //! Sort the records.
  void
  sort()
  {
    // CONTINUE: This causes warnings with the PGI compiler.
    LessThanCompare compare;
    std::sort(Base::begin(), Base::end(), compare);
  }

  //! Initialize for a set of queries.
  void
  initialize()
  {
  }

  //@}
};









//! Base class for a cell array combined with a search structure.
/*!
  A base class for a cell array in N-D.
  This class implements the common functionality of data structures
  which have cell arrays in the first N-1 coordinates and have a search
  data structure in the final coordinate.  It does not store pointers to
  the records.  Instead it has info on the number and size of the cells.
*/
template<std::size_t N, typename _Location, typename _Cell>
class CellSearch :
  public Orq<N, _Location>
{
  //
  // Types.
  //
private:

  //! The base class.
  typedef Orq<N, _Location> Base;

protected:

  //! The cell type.
  typedef _Cell Cell;
  //! The cell array type.
  typedef container::MultiArray < Cell, N - 1 > DenseArray;
  //! A single index.
  typedef typename DenseArray::Index Index;
  //! A multi-index into the cell array.
  typedef typename DenseArray::IndexList IndexList;
  //! A type to describe the extents of the cell array.
  typedef typename DenseArray::SizeList SizeList;
  //! The (N-1)-dimensional domain spanned by the cell array.
  typedef geom::BBox < typename Base::Float, N - 1 > Domain;

  //
  // Member data.
  //
private:

  //! The semi-open domain spanned by the grid.
  Domain _domain;
  //! The (N-1)-D array of cells that span the first N-1 coordinates.
  DenseArray _cellArray;
  //! The inverse cell sizes.
  std::array < typename Base::Float, N - 1 > _inverseCellSizes;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors.
  //@{

  //! Construct from the size of a cell and a Cartesian domain.
  /*!
    Construct given the cell size and the Cartesian domain
    that contains the records.

    \param delta is the suggested size of a cell.  The final coordinate
    is ignored.
    \param domain is the Cartesian domain spanned by the records.
    The final coordinate is ignored.
  */
  CellSearch(const typename Base::Point& delta,
             const typename Base::BBox& domain);

  //! Construct from the cell size, the Cartision domain and a range of records.
  /*!
    Construct given the cell size, the Cartesian domain and a range of records.

    \param delta is the suggested size of a cell.
    \param domain is the Cartesian domain that contains the records.
    \param first points to the begining of the range of records.
    \param last points to the end of the range.
  */
  CellSearch(const typename Base::Point& delta, const typename Base::BBox& domain,
             typename Base::Record first, typename Base::Record last);

  //! Construct from the cell size and a range of records.
  /*!
    Construct given the cell size and a range of records. An appropriate
    domain will be computed.

    \param delta is the suggested size of a cell.
    \param first points to the begining of the range of records.
    \param last points to the end of the range.
  */
  CellSearch(const typename Base::Point& delta, typename Base::Record first,
             typename Base::Record last);

  //@}
  //--------------------------------------------------------------------------
  //! \name Insert records.
  //@{

  //! Insert a single record.
  void
  insert(const typename Base::Record record)
  {
    getCell(Base::_location(record)).push_back(record);
    ++Base::_size;
  }

  //! Insert a range of records.
  /*!
    The input iterators are to a container of records.
  */
  void
  insert(typename Base::Record first, typename Base::Record last)
  {
    while (first != last) {
      insert(first);
      ++first;
    }
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Memory usage.
  //@{

  //! Return the memory usage.
  std::size_t
  getMemoryUsage() const;

  //@}
  //--------------------------------------------------------------------------
  //! \name Sorting and searching.
  //@{

  //! Sort the records.
  void
  sort();

  //! Initialize for a set of queries.
  void
  initialize();

  //@}
  //--------------------------------------------------------------------------
  //! \name File I/O
  //@{

  //! Print the data structure.
  void
  put(std::ostream& out) const;

  //@}

protected:

  //! Determine an appropriate domain to contain the records.
  void
  computeDomain(typename Base::Record first, typename Base::Record last)
  {
    setDomain(Base::computeDomain(first, last));
  }

  //! Convert the multikey to a cell array index.
  template<typename _AnyMultiKeyType>
  void
  convertMultiKeyToIndices(const _AnyMultiKeyType& multiKey, IndexList* mi)
  const
  {
    for (std::size_t n = 0; n != N - 1; ++n) {
      (*mi)[n] = Index((multiKey[n] - _domain.lower[n]) *
                       _inverseCellSizes[n]);
    }
  }

  //! Return the extents of the cell array.
  const SizeList&
  getCellArrayExtents() const
  {
    return _cellArray.extents();
  }

  //! Return a reference to the specified cell.
  Cell&
  getCell(const IndexList& index)
  {
    return _cellArray(index);
  }

  //! Return a const reference to the specified cell.
  const Cell&
  getCell(const IndexList& index) const
  {
    return _cellArray(index);
  }

  //! Return a const reference to the search structure that would hold the point.
  template<typename _AnyMultiKeyType>
  const Cell&
  getCell(const _AnyMultiKeyType& multiKey) const
  {
    IndexList mi;
    convertMultiKeyToIndices(multiKey, &mi);
    return _cellArray(mi);
  }

  //! Return a reference to the search structure that would hold the point.
  template<typename _AnyMultiKeyType>
  Cell&
  getCell(const _AnyMultiKeyType& multiKey)
  {
    IndexList mi;
    convertMultiKeyToIndices(multiKey, &mi);
    return _cellArray(mi);
  }

private:

  //! Compute the array extents and the sizes for the cells.
  void
  computeExtentsAndSizes(const typename Base::Point& suggestedCellSize);

  //! Set the domain. Ignore the last coordinate.
  void
  setDomain(const typename Base::BBox& domain);
};


//
// File I/O
//


//! Write to a file stream.
/*! \relates CellSearch */
template<std::size_t N, typename _Location, typename _Cell>
inline
std::ostream&
operator<<(std::ostream& out, const CellSearch<N, _Location, _Cell>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_CellSearch_ipp__
#include "stlib/geom/orq/CellSearch.ipp"
#undef __geom_CellSearch_ipp__

#endif
