// -*- C++ -*-

/*!
  \file amr/Patch.h
  \brief A patch is a node in the orthtree. It holds patch data.
*/

#if !defined(__amr_Patch_h__)
#define __amr_Patch_h__

#include "stlib/amr/MessageInputStream.h"
#include "stlib/amr/MessageOutputStreamChecked.h"

#include <iostream>

namespace stlib
{
namespace amr
{

//! A patch is a node in the orthtree. It holds patch data.
/*!
  Right now a patch holds a single PatchData. Later it will hold an array of
  PatchData. Consider using a typelist to describe the patchdata.
*/
template<class _PatchData, class _Traits>
class Patch
{
  //
  // Public types.
  //
public:

  //! The patch data.
  typedef _PatchData PatchData;
  //! A list of sizes.
  typedef typename _Traits::SizeList SizeList;
  //! A spatial index.
  typedef typename PatchData::SpatialIndex SpatialIndex;

  //
  // Member data.
  //
private:

  PatchData _patchData;

  //
  // Not implemented.
  //
private:

  // Default constructor not implemented.
  Patch();

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Allocate memory for the patch data.
  Patch(const SpatialIndex& spatialIndex, const SizeList& extents) :
    _patchData(spatialIndex, extents)
  {
  }

  //! Allocate memory for the patch data.
  /*! Use the example patch to determine any necessary parameters. */
  Patch(const SpatialIndex& spatialIndex, const Patch& patch) :
    _patchData(spatialIndex, patch._patchData)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return a const reference to the patch data.
  const PatchData&
  getPatchData() const
  {
    return _patchData;
  }

  //! Get the message stream size for this object.
  /*! \pre This must be initialized. */
  std::size_t
  getMessageStreamSize() const
  {
    return _patchData.getMessageStreamSize();
  }

  //! Get the message stream size for this object.
  static
  std::size_t
  getMessageStreamSize(const SizeList& extents)
  {
    return PatchData::getMessageStreamSize(extents);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Equality.
  //@{
public:

  //! Return true if the data structures are equal.
  bool
  operator==(const Patch& other) const
  {
    return _patchData == other._patchData;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Return a reference to the data.
  PatchData&
  getPatchData()
  {
    return _patchData;
  }

  //@}
  //--------------------------------------------------------------------------
  //! Prolongation, restriction, and synchronization.
  //@{
public:

  //! Prolongation from patch that is one level lower.
  void
  prolong(const Patch& source)
  {
    _patchData.prolong(source._patchData);
  }

  //! Restriction from patch that is one level higher.
  void
  restrict(const Patch& source)
  {
    _patchData.restrict(source._patchData);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Message stream I/O.
  //@{
public:

  //! Write to the message stream.
  void
  write(MessageOutputStream& out) const
  {
    out << _patchData;
  }

  //! Write to the checked message stream.
  void
  write(MessageOutputStreamChecked& out) const
  {
    out << _patchData;
  }

  //! Read from the message stream.
  void
  read(MessageInputStream& in)
  {
    in >> _patchData;
  }

  //@}
};

//! Write the patch data.
/*!
  \relates Patch
*/
template<class _PatchData, class _Traits>
inline
std::ostream&
operator<<(std::ostream& out, const Patch<_PatchData, _Traits>& x)
{
  out << x.getPatchData();
  return out;
}

} // namespace amr
}

#define __amr_Patch_ipp__
#include "stlib/amr/Patch.ipp"
#undef __amr_Patch_ipp__

#endif
