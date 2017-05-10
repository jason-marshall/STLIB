// -*- C++ -*-

/*!
  \file SparseCellArray.h
  \brief A class for a sparse cell array in N-D.
*/

#if !defined(__geom_SparseCellArray_h__)
#define __geom_SparseCellArray_h__

#include "stlib/geom/orq/CellArrayBase.h"

#include "stlib/container/MultiArray.h"

#include <algorithm>

namespace stlib
{
namespace geom
{

//! An index and cell for holding records.
template<typename _Cell>
struct IndexAndCell {
  //! The index of the cell.
  std::size_t index;
  //! The cell containing pointers to records.
  _Cell cell;
};


//! Less than comparison for indices.
template<typename _Cell>
inline
bool
operator<(const IndexAndCell<_Cell>& a, const IndexAndCell<_Cell>& b)
{
  // Note: The more obvious implementation may cause compilation errors.
  //return a.index < b.index;
  return b.index > a.index;
}


//! A vector of sparse cells.
template<typename _Cell>
class SparseCellVector :
  public std::vector<IndexAndCell<_Cell> >
{
  //
  // Types.
  //
private:

  typedef std::vector<IndexAndCell<_Cell> > Base;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{

  //! Default constructor.
  SparseCellVector() :
    Base()
  {
  }

  //! Construct and reserve memory for n elements.
  explicit
  SparseCellVector(const typename Base::size_type n) :
    Base(n) {}

  //! Construct from a range.
  template<typename _InputIterator>
  SparseCellVector(_InputIterator first, _InputIterator last) :
    Base(first, last)
  {
  }

  //--------------------------------------------------------------------------
  //! \name Searching.
  //@{

  // Return a const iterator to the first index and cell with index >= i.
  typename Base::const_iterator
  lower_bound(const std::size_t i) const
  {
    IndexAndCell<_Cell> val;
    val.index = i;
    return std::lower_bound(Base::begin(), Base::end(), val);
  }

  // Return an iterator to the first index and cell with index >= i.
  typename Base::iterator
  lower_bound(const std::size_t i)
  {
    IndexAndCell<_Cell> val;
    val.index = i;
    return std::lower_bound(Base::begin(), Base::end(), val);
  }

  // Return an iterator to cell i.
  _Cell&
  find(const std::size_t i)
  {
    IndexAndCell<_Cell> val;
    val.index = i;
    typename Base::iterator iter =
      std::lower_bound(Base::begin(), Base::end(), val);

    // If the cell does not yet exist, make it.
    if (iter == Base::end() || iter->index != i) {
      iter = Base::insert(iter, val);
    }

    return iter->cell;
  }

  //@}
};




//! A sparse cell array in N-D.
/*!
  A sparse cell array in N-D.
  The array is sparse in the last dimension.  Cell access is accomplished
  with array indexing in the N-1 directions and a binary search
  of a sorted vector in the final direction.
*/
template<std::size_t N, typename _Location>
class SparseCellArray :
  public CellArrayBase<N, _Location>
{
  //
  // Types.
  //
private:

  typedef CellArrayBase<N, _Location> Base;

protected:

  //! (N-1)-D array of 1-D sparce cell vectors.
  typedef container::MultiArray < SparseCellVector<typename Base::Cell>, N - 1 >
  VectorArray;

public:

  //! A Cartesian point.
  typedef typename Base::Point Point;
  //! A bounding box.
  typedef typename Base::BBox BBox;
  //! The cell type.
  typedef typename Base::Cell Cell;

private:

  //
  // Member data
  //

  //! The array of vectors.
  VectorArray _vectorArray;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{

  //! Construct from the size of a cell and a Cartesian domain.
  /*!
    Construct a cell grid given the grid size and the Cartesian domain that
    the grid spans.

    \param delta the suggested size of a cell.
    \param domain the Cartesian domain spanned by the records.
  */
  SparseCellArray(const Point& delta,
                  const BBox& domain) :
    Base(delta, domain),
    _vectorArray()
  {
    build();
  }

  //! Construct from the domain and a range of records.
  /*!
    Construct a cell grid given the array size, the Cartesian domain
    and a range of records.

    \param delta the suggested size of a cell.
    \param domain the Cartesian domain that contains the records.
    \param first points to the begining of the range of records.
    \param last points to the end of the semi-open range.
  */
  SparseCellArray(const Point& delta,
                  const BBox& domain,
                  typename Base::Record first, typename Base::Record last) :
    Base(delta, domain),
    _vectorArray()
  {
    build();
    // Insert the grid elements in the range.
    insert(first, last);
  }

  //! Construct from a range of records.
  /*!
    Construct a cell grid given the array size and a range of records.
    Compute an appropriate domain.

    \param delta the suggested size of a cell.
    \param first points to the begining of the range of records.
    \param last points to the end of the semi-open range.
  */
  SparseCellArray(const Point& delta,
                  typename Base::Record first, typename Base::Record last) :
    Base(delta, first, last),
    _vectorArray()
  {
    build();
    // Insert the grid elements in the range.
    insert(first, last);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Insert records.
  // @{

  //! Insert a single record.
  void
  insert(const typename Base::Record record)
  {
    Cell& b = (*this)(Base::_location(record));
    b.push_back(record);
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

  //! Get the grid points in the window.  Return the # of grid pts inside.
  template<class _OutputIterator>
  std::size_t
  computeWindowQuery(_OutputIterator iter,
                     const BBox& window) const;

  // @}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  // @{

  //! Print the records.
  void
  put(std::ostream& out) const;

  //@}

private:

  //
  // Accesors: Cell Indexing
  //

  //! Return a reference to the cell that would hold the point.
  /*!
    Indexing by location.  Return a reference to a cell.
    The multi-key must be in the domain of the cell array.
  */
  Cell&
  operator()(const Point& multiKey);

  void
  build();
};

//
// File I/O
//

//! Write to a file stream.
/*! \relates SparseCellArray */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const SparseCellArray<N, _Location>& x)
{
  x.put(out);
  return out;
}


} // namespace geom
}

#define __geom_SparseCellArray_ipp__
#include "stlib/geom/orq/SparseCellArray.ipp"
#undef __geom_SparseCellArray_ipp__

#endif
