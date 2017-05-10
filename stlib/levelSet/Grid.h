// -*- C++ -*-

/*!
  \file levelSet/Grid.h
  \brief A grid is a multi-array of patches.
*/

#if !defined(__levelSet_Grid_h__)
#define __levelSet_Grid_h__

#include "stlib/levelSet/GridGeometry.h"
#include "stlib/levelSet/Patch.h"
#include "stlib/levelSet/GridUniform.h"

#include "stlib/container/EquilateralArray.h"
#include "stlib/container/SimpleMultiArray.h"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace stlib
{
namespace levelSet
{

//! A grid is a multi-array of patches.
/*!
  \param _T The value type.
  \param _D The dimension
  \param N The extent in each dimension for a patch.
  \param _R The real number type. By default it is _T.
*/
template<typename _T, std::size_t _D, std::size_t N, typename _R = _T>
class Grid :
  public GridGeometry<_D, N, _R>,
  public container::SimpleMultiArray<Patch<_T, _D, N>, _D>
{
  //
  // Constants.
  //
public:

  //! The number of vertices (and also the number of voxels) per patch.
  BOOST_STATIC_CONSTEXPR std::size_t NumVerticesPerPatch =
    numerical::Exponentiation<std::size_t, N, _D>::Result;
  //! The logarithm (base 2) of the patch extent.
  BOOST_STATIC_CONSTEXPR std::size_t LogarithmOfExtent =
    numerical::Logarithm<std::size_t, 2, N>::Result;

  //
  // Types.
  //
private:

  //! The grid geometry.
  typedef GridGeometry<_D, N, _R> GeometryBase;
  //! A multi-array of patches.
  typedef container::SimpleMultiArray<Patch<_T, _D, N>, _D> ArrayBase;

public:

  //! A bounding box.
  typedef typename GeometryBase::BBox BBox;
  //! A pair of grid/patch indices.
  typedef typename GeometryBase::DualIndices DualIndices;

  //! An array index is the same as the size type.
  typedef typename ArrayBase::Index Index;
  //! A list of indices.
  typedef typename ArrayBase::IndexList IndexList;
  //! An index range.
  typedef typename ArrayBase::Range Range;

  //! A patch without ghost elements.
  /*! This is used for working with vertices. */
  typedef Patch<_T, _D, N> VertexPatch;
  //! A patch that has been padded on the upper sides with one layer of ghost elements.
  /*! This is used for working with voxels. */
  typedef container::EquilateralArray < _T, _D, N + 1 > VoxelPatch;

  //
  // Member data.
  //
protected:

  //! The data for all of the patches is contiguous in memory.
  std::vector<_T> _data;

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
  Grid(const BBox& domain, const _R targetSpacing);

  //! Construct from the Cartesian domain and the grid extents (number of patches).
  Grid(const BBox& domain, const IndexList& extents);

private:

  //! The copy constructor is not implemented.
  Grid(const Grid&);

  //! The assignment operator is not implemented.
  Grid&
  operator=(const Grid&);

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  using ArrayBase::arrayIndex;

  //! The data for all of the patches is contiguous in memory.
  const _T*
  data() const
  {
    return &_data[0];
  }

  using ArrayBase::operator();

  //! Return the value at the specified grid point.
  _T
  operator()(const IndexList& patch,
             const IndexList& index) const;

  //! Return the value at the specified grid point.
  _T
  operator()(const DualIndices& pair) const
  {
    return operator()(pair.first, pair.second);
  }

  //! Get the specified voxel patch.
  /*! The values on the upper boundaries are copied from the adjacent, refined
   vertex patches. If there is no refined, adjacent vertex patch, the
   nearest value in the vertex patch is used.

   \pre The requested patch must be refined.
  */
  void
  getVoxelPatch(const IndexList& i, VoxelPatch* patch) const;

  //! Get the number of refined patches.
  std::size_t
  numRefined() const;

  //! Get the total number of vertices in the refined patches.
  std::size_t
  numVertices() const
  {
    return _data.size();
  }

  //! Report the set of adjacent neighbors in the specified direction.
  template<typename _OutputIterator>
  void
  adjacentNeighbors(const DualIndices& pair,
                    const std::size_t direction,
                    _OutputIterator neighbors) const
  {
    adjacentNeighbors(pair.first, pair.second, direction, neighbors);
  }

  //! Report the set of adjacent neighbors in the specified direction.
  template<typename _OutputIterator>
  void
  adjacentNeighbors(const IndexList& patch,
                    const IndexList& index,
                    const std::size_t direction,
                    _OutputIterator neighbors) const;

  //! Report the set of all adjacent neighbors.
  template<typename _OutputIterator>
  void
  adjacentNeighbors(const DualIndices& pair, _OutputIterator neighbors) const;

  //! Report the set of all neighbors.
  /*! For the refined case, there are 3^_D - 1 neighbors. */
  template<typename _OutputIterator>
  void
  allNeighbors(const DualIndices& pair, _OutputIterator neighbors) const;

  //! Return true if the grid is valid.
  bool
  isValid() const;

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{
public:

  //! The data for all of the patches is contiguous in memory.
  _T*
  data()
  {
    return &_data[0];
  }

  //! Return a reference to the value at the specified grid point.
  _T&
  operator()(const IndexList& patch,
             const IndexList& index);

  //! Return the value at the specified grid point.
  _T&
  operator()(const DualIndices& pair)
  {
    return operator()(pair.first, pair.second);
  }

  //! Make all of the grids unrefined.
  /*! The grid extents and domain are not altered. */
  void
  clear();

  //! Refine the specified patches.
  void
  refine(const std::vector<std::size_t>& indices);

  //! Refine the patches that have one or more dependencies.
  void
  refine(const container::StaticArrayOfArrays<unsigned>& dependencies);

  //! Remove unecessary refinement.
  void
  coarsen();

  // @}
};


//! Add a constant to all vertices and fill values.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
Grid<_T, _D, N, _R>&
operator+=(Grid<_T, _D, N, _R>& grid, _T x);


//! Subtract a constant from all vertices and fill values.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
Grid<_T, _D, N, _R>&
operator-=(Grid<_T, _D, N, _R>& grid, _T x);


//! Write the grid in VTK XML format.
/*! \relates Grid */
template<typename _T, std::size_t N, typename _R>
void
writeVtkXml(const Grid<_T, 3, N, _R>& grid, std::ostream& out);


//! Write the grid in VTK XML format.
/*! \relates Grid */
template<typename _T, std::size_t N, typename _R>
void
writeVtkXml(const Grid<_T, 2, N, _R>& grid, std::ostream& out);


//! Print information about the grid.
/*! \relates Grid */
template<typename _T, std::size_t _D, std::size_t N, typename _R>
void
printInfo(const Grid<_T, _D, N, _R>& grid, std::ostream& out);


} // namespace levelSet
}

#define __levelSet_Grid_ipp__
#include "stlib/levelSet/Grid.ipp"
#undef __levelSet_Grid_ipp__

#endif
