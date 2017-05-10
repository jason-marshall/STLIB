// -*- C++ -*-

/*!
  \file geom/mesh/quadrilateral/file_io.h
  \brief Implements file I/O operations for QuadMesh.
*/

#if !defined(__geom_mesh_quadrilateral_file_io_h__)
#define __geom_mesh_quadrilateral_file_io_h__

#include "stlib/geom/mesh/quadrilateral/QuadMesh.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup quadrilateral_file_io File I/O for QuadMesh
  A quadrilateral mesh can be written to a file in ascii or binary format.

  The first number identifies the space dimension.
  Then the vertex coordinates are enumerated, followed by the tuples of
  vertex indices that comprise the cells.  These indices are in the range
  [0..numberOfVertices).
  The file format is:
  \verbatim
  spaceDimension
  numberOfVertices
  vertex_0_coord_0 vertex_0_coord_1 ... vertex_0_coord_N-1
  vertex_1_coord_0 vertex_1_coord_1 ... vertex_1_coord_N-1
  ...
  numberOfCells
  cell_0_index_0 cell_0_index_1 ... cell_0_index_M
  cell_1_index_0 cell_1_index_1 ... cell_1_index_M
  ... \endverbatim

  For example, a mesh of the unit square is:
  \verbatim
  2
  4
  0.0 0.0
  1.0 0.0
  1.0 1.0
  0.0 1.0
  1
  0 1 2 3 \endverbatim
*/
//@{

//! Write a quad mesh in ascii format.
template<std::size_t N, typename VertForIter, typename CellForIter>
void
writeQuadMeshAscii(std::ostream& out,
                   VertForIter verticesBeginning, VertForIter verticesEnd,
                   CellForIter cellsBeginning, CellForIter cellsEnd);


//! Write a quad mesh in ascii format.
/*! \relates QuadMesh */
template<std::size_t N, typename T>
inline
void
writeAscii(std::ostream& out, const QuadMesh<N, T>& x) {
   writeQuadMeshAscii<N>(out, x.getVerticesBeginning(), x.getVerticesEnd(),
                         x.getIndexedFacesBeginning(), x.getIndexedFacesEnd());
}


//! Write a quad mesh in binary format.
/*! \relates QuadMesh */
template<std::size_t N, typename T>
void
writeBinary(std::ostream& out, const QuadMesh<N, T>& x);


//! Write a quad mesh in binary format.
template<std::size_t N, typename VertForIter, typename CellForIter>
void
writeQuadrilateralBinary(std::ostream& out,
                         VertForIter verticesBeginning, VertForIter verticesEnd,
                         CellForIter cellsBeginning, CellForIter cellsEnd);


//! Read a quad mesh in ascii format.
/*! \relates QuadMesh */
template<std::size_t N, typename T>
void
readAscii(std::istream& in, QuadMesh<N, T>* x);


//! Read a quad mesh in binary format.
/*! \relates QuadMesh */
template<std::size_t N, typename T>
void
readBinary(std::istream& in, QuadMesh<N, T>* x);


//@}

} // namespace geom
}

#define __geom_mesh_quadrilateral_file_io_ipp__
#include "stlib/geom/mesh/quadrilateral/file_io.ipp"
#undef __geom_mesh_quadrilateral_file_io_ipp__

#endif
