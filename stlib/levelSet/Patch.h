// -*- C++ -*-

#if !defined(__levelSet_Patch_h__)
#define __levelSet_Patch_h__

#include "stlib/levelSet/count.h"
#include "stlib/container/EquilateralArrayRef.h"

namespace stlib
{
namespace levelSet
{

//! A patch is a multi-array that has equal extents in each dimension.
/*!
  \param _T The value type.
  \param _D The dimension
  \param N The extent in each dimension.

  A patch is either refined, or unrefined. For refined patches, storage
  has been externally allocated for the array. Internally, this condition
  is tested by checking the address of the array memory. A nonzero value
  indicated that the storage has been allocated and the patch is refined.
  For an unrefined patch, the fillValue data member represents the value
  at all grid points. Typically, one only uses refined patches
  in a narrow band around the zero iso-surface. One uses a fill value of
  \f$\pm \infty\f$ to denote large positive or negative function values.
  A fill value of \c NaN indicates that the grid points in the patch have
  unknown values.

  We inherit from container::EquilateralArrayRef. Hence, this class inherits
  the interface of an equilateral array. Note that we inherit from
  container::EquilateralArrayRef and not container::EquilateralArray because this class
  does not own the storage for the %array.
  When this class is used by Grid, the memory for all
  of the patches is stored in one contiguous block.
*/
template<typename _T, std::size_t _D, std::size_t N>
class Patch :
  public container::EquilateralArrayRef<_T, _D, N>
{
  //
  // Types.
  //
private:

  //! The base class.
  typedef container::EquilateralArrayRef<_T, _D, N> Base;

  //
  // Member data.
  //
public:

  //! The fill value is used for unrefined patches.
  typename Base::value_type fillValue;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor, assignment operator, and
    destructor. */
  // @{
public:

  //! Default constructor. Unrefined with NaN for the fill value.
  Patch() :
    Base(),
    fillValue(std::numeric_limits<typename Base::value_type>::quiet_NaN())
  {
  }

  //! Construct from a pointer to the array data.
  Patch(typename Base::iterator data) :
    Base(data),
    fillValue(0)
  {
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  bool
  isValid() const
  {
    // Either the data pointer must be valid, or there must be a valid
    // (non-zero) fill value.
    return (Base::data() == 0 && fillValue != 0) ||
           (Base::data() != 0 && fillValue == 0);
  }

  //! Return true if the patch is refined.
  bool
  isRefined() const
  {
    return Base::data();
  }

  //! Return true if the patch should be coarsened.
  bool
  shouldBeCoarsened() const
  {
    return isRefined() && allSame(Base::begin(), Base::end());
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{
public:

  //! Reference an externally allocated array.
  void
  refine(const typename Base::iterator data)
  {
    Base::setData(data);
    fillValue = 0;
  }

  //! Coarsen the patch. Set the fill value to the first grid value.
  /*! \pre The patch must be refined. */
  void
  coarsen()
  {
    assert(Base::data());
    fillValue = *Base::begin();
    Base::setData(0);
  }

  //! Clear the patch. Set the data to 0 and the fill value to NaN.
  void
  clear()
  {
    Base::setData(0);
    fillValue = std::numeric_limits<typename Base::value_type>::quiet_NaN();
  }

private:

  // Hide setData().
  void
  setData();

  // @}
};


} // namespace levelSet
}

#endif
