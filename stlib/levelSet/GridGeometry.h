// -*- C++ -*-

/*!
  \file levelSet/GridGeometry.h
  \brief Geometric information about an AMR grid that is a multi-array of patches.
*/

#if !defined(__levelSet_GridGeometry_h__)
#define __levelSet_GridGeometry_h__

#include "stlib/container/StaticArrayOfArrays.h"
#include "stlib/container/SimpleMultiIndexRangeIterator.h"
#include "stlib/geom/kernel/BBox.h"
#include "stlib/numerical/constants/Exponentiation.h"
#include "stlib/numerical/constants/Logarithm.h"

#include <iostream>

namespace stlib
{
namespace levelSet
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_ARRAY_IO_OPERATORS;

//! Geometric information about an AMR grid that is a multi-array of patches.
/*!
  \param _D The dimension.
  \param N The extent in each dimension for a patch.
  \param _R The real number type.
*/
template<std::size_t _D, std::size_t N, typename _R>
class GridGeometry
{
  //
  // Constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _D;

  //
  // Types.
  //
public:

  //! A Cartesian point.
  typedef std::array<_R, Dimension> Point;
  //! A bounding box.
  typedef geom::BBox<_R, Dimension> BBox;
  //! An array index is the same as the size type.
  typedef std::size_t Index;
  //! A list of indices.
  typedef std::array<std::size_t, Dimension> IndexList;
  //! An index range.
  typedef container::SimpleMultiIndexRange<Dimension> Range;
  //! A pair of grid/patch indices.
  typedef std::pair<IndexList, IndexList> DualIndices;

  //
  // Member data.
  //
public:

  //! The number of patches in each dimension.
  const IndexList gridExtents;
  //! The Cartesian coordinates of the lower corner of the grid.
  const Point lowerCorner;
  //! The grid spacing for the patches.
  const _R spacing;

private:

  //! The strides for indexing the array of patches.
  const IndexList _strides;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default destructor. The copy constructor and assignment
    operator are disabled.
  */
  // @{
public:

  //! Construct from the Cartesian domain and the suggested grid patch spacing.
  /*!
    The grid spacing will be no greater than the suggested grid spacing and
    is the same in all dimensions. The domain will be expanded in the upper
    limits to exactly accomodate the grid.
  */
  GridGeometry(const BBox& domain, const _R targetSpacing);

  //! Construct from the Cartesian domain and the grid extents (number of patches).
  GridGeometry(const BBox& domain, const IndexList& extents);

private:

  //! Calculate the extents.
  static
  IndexList
  calculateExtents(const BBox& domain, const _R targetSpacing);

  //! Compute the strides.
  /*! This is static so it can be called in the initializer list. */
  static
  IndexList
  computeStrides(const IndexList& extents)
  {
    IndexList strides;
    strides[0] = 1;
    for (std::size_t i = 1; i != Dimension; ++i) {
      strides[i] = strides[i - 1] * extents[i - 1];
    }
    return strides;
  }

  //! The copy constructor is not implemented.
  GridGeometry(const GridGeometry&);

  //! The assignment operator is not implemented.
  GridGeometry&
  operator=(const GridGeometry&);

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  //! Return the %array index for the given index list.
  /*! This index is in the range [0..numberOfPatches-1]. */
  Index
  arrayIndex(const IndexList& indices) const
  {
    return ext::dot(_strides, indices);
  }

  //! Return the Cartesian domain spanned by the grid.
  BBox
  domain() const;

  //! Return the length of a side of a voxel patch.
  _R
  getVoxelPatchLength() const
  {
    return spacing * N;
  }

  //! Return the domain for the specified vertex patch.
  BBox
  getVertexPatchDomain(const IndexList& i) const;

  //! Return the lower corner of the Cartesian domain for the specified patch.
  /*! Note that it doesn't matter whether one is referring to a vertex patch
   or a voxel patch. */
  Point
  getPatchLowerCorner(const IndexList& i) const
  {
    return lowerCorner +
      getVoxelPatchLength() * stlib::ext::convert_array<_R>(i);
  }

  //! Return the center of the Cartesian domain for the specified vertex patch.
  Point
  getVertexPatchCenter(const IndexList& i) const
  {
    return getPatchLowerCorner(i) + _R(0.5) * spacing * (N - 1);
  }

  //! Return the Cartesian position of the specified vertex.
  Point
  indexToLocation(const IndexList& patch, const IndexList& index) const;

  //! Report the specified set of grid points as patch/grid multi-index pairs.
  template<typename _OutputIterator>
  void
  report(const IndexList& patch, const Range& range,
         _OutputIterator neighbors) const;

  //! Return true if the grid is valid.
  bool
  isValid() const
  {
    return spacing > 0;
  }

  // @}
};


//! Determine the objects whose bounding boxes overlap each patch.
/*! \relates GridGeometry */
template<std::size_t _D, std::size_t N, typename _R, typename _InputIterator>
void
patchDependencies(const GridGeometry<_D, N, _R>& grid, _InputIterator begin,
                  _InputIterator end,
                  container::StaticArrayOfArrays<unsigned>* dependencies);


//! Determine the patches that intersect the bounding box.
/*! \relates GridGeometry */
template<std::size_t _D, std::size_t N, typename _R, typename _OutputIterator>
void
getIntersectingPatches(const GridGeometry<_D, N, _R>& grid,
                       geom::BBox<_R, _D> box, _OutputIterator indices);


//! Print information about the grid.
/*! \relates GridGeometry */
template<std::size_t _D, std::size_t N, typename _R>
void
printInfo(const GridGeometry<_D, N, _R>& grid, std::ostream& out);


} // namespace levelSet
}

#define __levelSet_GridGeometry_ipp__
#include "stlib/levelSet/GridGeometry.ipp"
#undef __levelSet_GridGeometry_ipp__

#endif
