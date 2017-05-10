// -*- C++ -*-

/*!
  \file amr/PatchAdjacent.h
  \brief A patch that stores links to its adjacent neighbors in the orthree.
*/

#if !defined(__amr_PatchAdjacent_h__)
#define __amr_PatchAdjacent_h__

#include "stlib/amr/Patch.h"
#include "stlib/amr/Orthtree.h"

namespace stlib
{
namespace amr
{

//! A patch that stores links to its adjacent neighbors in the orthree.
/*!
  This class stores an iterator for each signed direction. If the neighbor
  is at the same or a lower level, the iterator points to that neighbor.
  If the neighbors are at higher levels, the iterator points to the lower
  corner of the block of neighbors.
*/
template<class _PatchData, class _Traits>
class PatchAdjacent : public Patch<_PatchData, _Traits>
{
  //
  // Private types.
  //
private:

  typedef Patch<_PatchData, _Traits> Base;

  //
  // Public types.
  //
public:

  //! The patch data.
  typedef _PatchData PatchData;
  //! A single index.
  typedef typename _Traits::Index Index;
  //! A multiindex.
  typedef typename _Traits::IndexList IndexList;
  //! A list of sizes.
  typedef typename Base::SizeList SizeList;
  //! A spatial index.
  typedef typename _Traits::SpatialIndex SpatialIndex;
  //! A neighbor is a pair of the spatial index and the patch.
  typedef std::pair<const SpatialIndex, PatchAdjacent> Neighbor;

  //
  // Member data.
  //
public:

  std::array<Neighbor*, 2 * _Traits::Dimension> adjacent;

  //
  // Not implemented.
  //
private:

  // Default constructor not implemented.
  PatchAdjacent();

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Allocate memory for the patch data. Set the adjacent link to a null value.
  PatchAdjacent(const SpatialIndex& spatialIndex, const SizeList& extents) :
    Base(spatialIndex, extents),
    adjacent()
  {
    std::fill(adjacent.begin(), adjacent.end(), static_cast<Neighbor*>(0));
  }

  //! Allocate memory for the patch data. The adjacent links are copied.
  /*! Use the example patch to determine any necessary parameters. */
  PatchAdjacent(const SpatialIndex& spatialIndex, const PatchAdjacent& patch) :
    Base(spatialIndex, patch),
    adjacent(patch.adjacent)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  using Base::getPatchData;
  using Base::getMessageStreamSize;

  //@}
  //--------------------------------------------------------------------------
  //! \name Equality.
  //@{
public:

  //! Return true if the data structures are equal.
  bool
  operator==(const PatchAdjacent& other) const
  {
    return adjacent == other.adjacent && Base::operator==(other);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Message stream I/O.
  //@{
public:

  using Base::write;
  using Base::read;

  //@}
};

} // namespace amr
}

#endif
