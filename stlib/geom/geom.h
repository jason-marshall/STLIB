// -*- C++ -*-

#if !defined(__geom_h__)
#define __geom_h__

#include "stlib/geom/grid.h"
#include "stlib/geom/kernel.h"
#include "stlib/geom/mesh.h"
#include "stlib/geom/orq.h"
#include "stlib/geom/polytope.h"
#include "stlib/geom/shape.h"
#include "stlib/geom/spatialIndexing.h"
#include "stlib/geom/tree.h"

namespace stlib
{
//! All classes and functions in the computational geometry package are defined in the geom namespace.
namespace geom
{

/*!
\mainpage Computational Geometry Package

This package provides computational geometry algorithms and data structures.
This is a templated class library. Thus
there is no library to compile or link with. Just include the
appropriate header files in your application code when you compile.
All classes and functions are in the \c geom namespace.

The geom package is composed of the following sub-packages:
- The \ref geom_kernel contains geometric primitives like line segments
  and planes.
- The \ref geom_mesh has simplicial mesh data structures and algorithms for
generating and optimizing meshes.
- The \ref geom_neighbors has classes for performing various neighbor
searches for points in N-D.
- The \ref geom_polytope has polygons and polyhedra.
- The \ref geom_grid has grids which store objects and support mathematical
operations.
- The \ref geom_orq has data structures for doing orthogonal range queries.
- The \ref geom_spatialIndexing has an orthtree (quadtree, octree, etc.) data structure.
- The \ref geom_tree has a bounding box tree.
- The \ref geom_shape has a function for ordinary Procrustes analysis.

Note that I don't have a point or vector class. For this
functionality I use the std::array class.
*/

/*!
\page geom_bibliography Bibliography

- \anchor geom_edelsbrunner2001
Herbert Edelsbrunner,
"Geometry and Topology for Mesh Generation,"
Cambridge University Press, 2001.
- \anchor geom_press2007
William H. Press, Saul A. Teukolsky, William T. Vetterling, and
Brian P. Flannery,
"Numerical Recipes. The Art of Scientific Computing. Third Edition,"
Cambridge University Press, 2007.
- \anchor geom_ridders1982
Ridders, C. J. F., "Accurate computation of F'(x) and F'(x)F''(x),"
Advances in Engineering Software, vol. 4, no. 2, pp. 75-76.

*/

} // namespace geom
}

#endif
