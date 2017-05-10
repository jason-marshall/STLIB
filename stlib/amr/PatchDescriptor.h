// -*- C++ -*-

/*!
  \file amr/PatchDescriptor.h
  \brief Describe a patch.
*/

#if !defined(__amr_PatchDescriptor_h__)
#define __amr_PatchDescriptor_h__

#include "stlib/amr/FieldDescriptor.h"

namespace stlib
{
namespace amr
{

//! Describe a patch.
/*!
  \param _Traits Traits for the orthtree.
*/
template<typename _Traits>
class PatchDescriptor
{
  //
  // Public types.
  //
public:

  //! The spatial index.
  typedef typename _Traits::SpatialIndex SpatialIndex;
  //! A list of sizes.
  typedef typename _Traits::SizeList SizeList;

  //
  // Member data.
  //
private:

  //! The index extents of a patch.
  SizeList _extents;
  //! The ghost cell width.
  std::size_t _ghostCellWidth;
  //! The field descriptor.
  FieldDescriptor _fieldDescriptor;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
private:

  //! The default constructor is not implemented.
  PatchDescriptor();

public:

  //! Construct from the array extents and the ghost cell width.
  PatchDescriptor(const SizeList& extents, const std::size_t ghostCellWidth,
                  const FieldDescriptor& fieldDescriptor) :
    _extents(extents),
    _ghostCellWidth(ghostCellWidth),
    _fieldDescriptor(fieldDescriptor)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Get the grid index extents.
  const SizeList&
  getExtents() const
  {
    return _extents;
  }

  //! Return the ghost cell width.
  std::size_t
  getGhostCellWidth() const
  {
    return _ghostCellWidth;
  }

  //! Get the field descriptor.
  const FieldDescriptor&
  getFieldDescriptor() const
  {
    return _fieldDescriptor;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Equality.
  //@{
public:

  //! Return true if the data structures are equal.
  bool
  operator==(const PatchDescriptor& other)
  {
    return _extents == other._extents &&
           _ghostCellWidth == other._ghostCellWidth;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Mathematical.
  //@{
public:

  //! Compute the index lower bounds for the patch.
  SizeList
  computeLowerBounds(const SpatialIndex& spatialIndex) const
  {
    return _extents * spatialIndex.getCoordinates();
  }

  //@}
};

} // namespace amr
}

#define __amr_PatchDescriptor_ipp__
#include "stlib/amr/PatchDescriptor.ipp"
#undef __amr_PatchDescriptor_ipp__

#endif
