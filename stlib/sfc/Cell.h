// -*- C++ -*-

#if !defined(__sfc_Cell_h__)
#define __sfc_Cell_h__

/*!
  \file
  \brief Functions for linked information in cells.
*/

#include "stlib/geom/kernel/ExtremePoints.h"

#include <vector>

#include <cassert>

namespace stlib
{
namespace sfc
{


//! Build a cell from a range of objects.
/*! \note This functor is not defined for generic types. It must be 
  specialized for each cell type in which you build SFC cell data structures
  from objects. */
template<typename _Cell>
struct BuildCell;


//! Specialization for ExtremePoints.
template<typename _T, std::size_t _D>
struct BuildCell<geom::ExtremePoints<_T, _D> > {
  //! Build an ExtremePoints.
  template<typename _ForwardIterator>
  geom::ExtremePoints<_T, _D>
  operator()(_ForwardIterator begin, _ForwardIterator end)
  {
    return geom::extremePoints<geom::ExtremePoints<_T, _D> >(begin, end);
  }
};


//! Specialization for BBox.
template<typename _T, std::size_t _D>
struct BuildCell<geom::BBox<_T, _D> > {
  //! Build a BBox.
  template<typename _ForwardIterator>
  geom::BBox<_T, _D>
  operator()(_ForwardIterator begin, _ForwardIterator end)
  {
    return geom::specificBBox<geom::BBox<_T, _D> >(begin, end);
  }
};


//! When cells are not stored in the SFC data structures, this is used for the cell type.
/*! This class should not be used outside of the implementation of the SFC
 data structures. */
struct DummyCell
{
  //! Define merging for dummy cells.
  DummyCell&
  operator+=(DummyCell const& /*other*/)
  {
    return *this;
  }

  //! Define the equality operator.
  bool
  operator==(DummyCell const& /*other*/) const
  {
    return true;
  }
};


//! Don't print anything for dummy cells.
inline
std::ostream&
operator<<(std::ostream& out, DummyCell const& /*x*/)
{
  return out;
}


//! Specialization for DummyCell.
template<>
struct BuildCell<DummyCell> {
  //! Build a DummyCell.
  template<typename _ForwardIterator>
  DummyCell
  operator()(_ForwardIterator /*begin*/, _ForwardIterator /*end*/)
  {
    return DummyCell{};
  }
};


//! Construct by merging a group of cells, specified by a range of indices.
/*!
  Cells are merged using the += operator. Thus, this operator must be 
  defined for any cell type in which cells are merged.
*/
template<typename _Container, typename _ForwardIterator>
inline
typename _Container::value_type
cellMerge(_Container const& cells, _ForwardIterator begin,
          _ForwardIterator end)
{
#ifdef STLIB_DEBUG
  assert(begin != end);
#endif
  typename _Container::value_type cell = cells[*begin++];
  while (begin != end) {
    cell += cells[*begin++];
  }
  return cell;
}


} // namespace sfc
} // namespace stlib

#endif

