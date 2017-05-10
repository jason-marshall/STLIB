// -*- C++ -*-

/*!
  \file CellArrayStatic.h
  \brief A class for a static cell array in N-D.
*/

#if !defined(__geom_CellArrayStatic_h__)
#define __geom_CellArrayStatic_h__

#include "stlib/geom/orq/CellArrayBase.h"

#include "stlib/container/StaticArrayOfArrays.h"

namespace stlib
{
namespace geom
{

//! A static cell array in N-D.
/*!
  A static, dense cell array in N-D.
*/
template<std::size_t N, typename _Location>
class CellArrayStatic :
  public CellArrayBase<N, _Location>
{
  //
  // Types.
  //
private:

  typedef CellArrayBase<N, _Location> Base;

  //
  // Data
  //
private:

  // The strides are used for array indexing.
  typename Base::IndexList _strides;
  // The array of cells.
  container::StaticArrayOfArrays<typename Base::Record> _cellArray;

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{
public:

  //! Construct from the cell size and the domain.
  /*!
    Construct a cell grid given the cell size and the Cartesian domain.

    \param delta The suggested size of a cell.
    \param domain The Cartesian domain that contains the records.
  */
  CellArrayStatic(const typename Base::Point& delta,
                  const typename Base::BBox& domain) :
    Base(delta, domain),
    _strides(),
    _cellArray()
  {
  }

  //! Construct from a range of records.
  /*!
    Construct a cell grid given the cell size, the Cartesian domain
    and a range of records.

    \pre The records must lie in the specified domain.
    \pre There must be a non-zero number of records.

    \param delta The suggested size of a cell.
    \param domain The Cartesian domain that contains the records.
    \param first The first record.
    \param last The last record.

    \note This function assumes that the records are iterators.
  */
  CellArrayStatic(const typename Base::Point& delta,
                  const typename Base::BBox& domain,
                  typename Base::Record first, typename Base::Record last) :
    Base(delta, domain),
    _strides(),
    _cellArray()
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
  CellArrayStatic(const typename Base::Point& delta,
                  typename Base::Record first,
                  typename Base::Record last) :
    Base(delta, first, last),
    _strides(),
    _cellArray()
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
  CellArrayStatic(const std::size_t numberOfCells,
                  typename Base::Record first,
                  typename Base::Record last) :
    Base(numberOfCells, first, last),
    _strides(),
    _cellArray()
  {
    // Insert the grid elements in the range.
    insert(first, last);
  }

  //! Construct from a range of records.
  /*!
    \pre There must be a non-zero number of records.

    \param first The first record.
    \param last The last record.

    \c first and \c last must be forward iterators because there is one pass
    to compute an appropriate domain and one pass to insert the records.
    The number of cells will be approximately the number of records.

    \note This function assumes that the records are iterators.
  */
  CellArrayStatic(typename Base::Record first, typename Base::Record last) :
    Base(std::distance(first, last), first, last),
    _strides(),
    _cellArray()
  {
    // Insert the grid elements in the range.
    insert(first, last);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Insert/Erase records.
  // @{
public:

  //! Insert a number of records.
  /*!
    \param first The first record.
    \param last The last record.
    \pre The set of records must be empty.
    \note This function assumes that the records are iterators.
  */
  void
  insert(typename Base::Record first, typename Base::Record last)
  {
    assert(Base::empty());
    // Set the size.
    Base::_size = std::distance(first, last);
    // Compute the strides.
    computeStrides();
    // Determine where each record should be placed and the sizes for the
    // cells.
    std::vector<std::size_t> indices(Base::size());
    std::vector<std::size_t> sizes(ext::product(Base::getExtents()),
                                   std::size_t(0));
    typename Base::Record record = first;
    for (std::size_t i = 0; i != indices.size(); ++i) {
      indices[i] = computeIndex(Base::_location(record++));
      ++sizes[indices[i]];
    }
    // Allocate the static array of arrays that represent the cell array.
    _cellArray.rebuild(sizes.begin(), sizes.end());
    // Copy the records into the array.
    std::vector<typename container::StaticArrayOfArrays<typename Base::Record>::
    iterator> positions(sizes.size());
    for (std::size_t i = 0; i != positions.size(); ++i) {
      positions[i] = _cellArray.begin(i);
    }
    record = first;
    for (std::size_t i = 0; i != indices.size(); ++i) {
      *positions[indices[i]]++ = record++;
    }
  }

  //! Clear all records.
  void
  clear()
  {
    _cellArray.clear();
    // There are now no records.
    Base::_size = 0;
  }

  //! Rebuild the data structure for a new set of records or following modifications to the record multi-keys.
  template<class ForwardIterator>
  void
  rebuild(ForwardIterator first, ForwardIterator last)
  {
    clear();
    Base::rebuild(std::distance(first, last), first, last);
    // Insert the grid elements in the range.
    insert(first, last);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Memory usage.
  // @{
public:

  //! Return the memory usage.
  std::size_t
  getMemoryUsage() const
  {
    return _cellArray.getMemoryUsage();
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Window Queries.
  // @{
public:

  //! Get the records in the window.  Return the # of records inside.
  template<typename _OutputIterator>
  std::size_t
  computeWindowQuery(_OutputIterator iter,
                     const typename Base::BBox& window) const;

  // @}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  // @{
public:

  //! Print the records.
  void
  put(std::ostream& out) const;

  // @}

protected:

  //! Convert the multikey to a cell index.
  template<typename _AnyMultiKeyType>
  std::size_t
  computeIndex(const _AnyMultiKeyType& multiKey) const
  {
    std::size_t index = 0;
    for (std::size_t n = 0; n != N; ++n) {
      index += _strides[n] *
               typename Base::Index((multiKey[n] - Base::getDomain().lower[n]) *
                                    Base::getInverseCellSizes()[n]);
    }
    return index;
  }

private:

  // Compute the strides from the extents.
  void
  computeStrides()
  {
    _strides[0] = 1;
    for (std::size_t n = 1; n != N; ++n) {
      _strides[n] = _strides[n - 1] * Base::getExtents()[n - 1];
    }
  }

};

//
// File I/O
//

//! Write to a file stream.
/*! \relates CellArrayStatic */
template<std::size_t N, typename _Location>
inline
std::ostream&
operator<<(std::ostream& out, const CellArrayStatic<N, _Location>& x)
{
  x.put(out);
  return out;
}

} // namespace geom
}

#define __geom_CellArrayStatic_ipp__
#include "stlib/geom/orq/CellArrayStatic.ipp"
#undef __geom_CellArrayStatic_ipp__

#endif
