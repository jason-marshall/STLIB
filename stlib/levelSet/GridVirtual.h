// -*- C++ -*-

#if !defined(__levelSet_GridVirtual_h__)
#define __levelSet_GridVirtual_h__

#include "stlib/container/SimpleMultiIndexRangeIterator.h"
#include "stlib/container/EquilateralArray.h"
#include "stlib/container/StaticArrayOfArrays.h"
#include "stlib/geom/kernel/BBox.h"
//#include "stlib/numerical/constants/Logarithm.h"

//#include <vector>
//#include <fstream>
//#include <iomanip>
#include <iostream>
//#include <sstream>

namespace stlib
{
namespace levelSet
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! A virtual grid stores information about the patches that cover the domain.
/*!
  \param _T The value type.
  \param _D The dimension
  \param N The extent in each dimension for a patch.
*/
template<typename _T, std::size_t _D, std::size_t N>
class GridVirtual
{
  //
  // Constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _D;
#if 0
  //! The number of vertices per patch.
  BOOST_STATIC_CONSTEXPR std::size_t NumVerticesPerPatch =
    numerical::Exponentiation<std::size_t, N, _D>::Result;
  //! The logarithm (base 2) of the patch extent.
  BOOST_STATIC_CONSTEXPR std::size_t LogarithmOfExtent =
    numerical::Logarithm<std::size_t, 2, N>::Result;
#endif

public:

  //! A patch shares points with adjacent neighbors.
  typedef container::EquilateralArray<_T, _D, N> Patch;
  //! A Cartesian point.
  typedef std::array<_T, Dimension> Point;
  //! A bounding box.
  typedef geom::BBox<_T, Dimension> BBox;
  //! A single index.
  typedef std::size_t Index;
  //! A list of indices.
  typedef std::array<Index, Dimension> IndexList;

  //
  // Member data.
  //
public:

  //! The number of patches in each dimension.
  const IndexList extents;
  //! The Cartesian coordinates of the lower corner of the grid.
  const Point lowerCorner;
  //! The grid spacing for the patches.
  const _T spacing;

protected:

  //! The strides for indexing the multi-array of patches.
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
  GridVirtual(const BBox& domain, _T targetSpacing);

private:

  //! Calculate the extents.
  static
  IndexList
  calculateExtents(const BBox& domain, _T targetSpacing);

  //! Calculate the strides from the extents.
  static
  IndexList
  calculateStrides(const IndexList& ext);

  //! The copy constructor is not implemented.
  GridVirtual(const GridVirtual&);

  //! The assignment operator is not implemented.
  GridVirtual&
  operator=(const GridVirtual&);

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  //! Return the Cartesian domain spanned by the grid.
  BBox
  domain() const;

  //! Return the length of a side of a voxel patch.
  _T
  getPatchLength() const
  {
    return spacing * (N - 1);
  }

  //! Return the domain for the specified patch.
  BBox
  getPatchDomain(const IndexList& i) const;

  //! Return the lower corner of the Cartesian domain for the specified patch.
  Point
  getPatchLowerCorner(const IndexList& i) const;

  //! Return the %array index for the given index list.
  Index
  arrayIndex(const IndexList& indices) const
  {
    return ext::dot(_strides, indices);
  }

  //! Return the Cartesian position of the specified vertex.
  Point
  indexToLocation(const IndexList& patch,
                  const IndexList& index) const;

  // @}
};


//! Determine the objects whose bounding boxes overlap each patch.
/*! \relates GridVirtual */
template<typename _T, std::size_t _D, std::size_t N, typename _InputIterator>
void
patchDependencies(const GridVirtual<_T, _D, N>& grid, _InputIterator begin,
                  _InputIterator end,
                  container::StaticArrayOfArrays<unsigned>* dependencies);


//! Determine the patches that intersect the bounding box.
/*! \relates GridVirtual */
template<typename _T, std::size_t _D, std::size_t N,
         typename _OutputIterator>
void
getIntersectingPatches(const GridVirtual<_T, _D, N>& grid,
                       geom::BBox<_T, _D> box, _OutputIterator indices);


//! Print information about the grid.
/*! \relates GridVirtual */
template<typename _T, std::size_t _D, std::size_t N>
void
printInfo(const GridVirtual<_T, _D, N>& grid, std::ostream& out);


} // namespace levelSet
}

#define __levelSet_GridVirtual_ipp__
#include "stlib/levelSet/GridVirtual.ipp"
#undef __levelSet_GridVirtual_ipp__

#endif
