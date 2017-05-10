// -*- C++ -*-

/*!
  \file simplicial.h
  \brief Includes the classes for topological optimization of simplicial meshes.
*/

/*!
  \page simplicial Simplicial Mesh Package

  \section mesh_simplicial_Simple Simple Topological Transformations

  In 3-D, local topological transformations replace a set of tetrahedra with a
  different set that fills the same domain.  (On the boundary they may
  fill only approximately the same domain.)

  A 2-3 flip removes a face and adds a new edge.  The inverse is a 3-2 flip.
  Two boundary faces are replaced in a 2-2 flip.  The faces should be
  approximately co-planar.

  \image html flip23_22.jpg
  <!--CONTINUE: I don't like hard coding the size.-->
  \image latex flip23_22.pdf width=4in

  \section mesh_simplicial_Composite Composite Topological Transformations

  We have been implementing the algorithms presented in
  ``Two Discrete Optimization Algorithms for the Topological Improvement
  of Tetrahedral Meshes'' by Jonathan Shewchuk.  He presents two
  \e composite topological operations: <em>edge removal</em>
  and <em>multi-face removal</em>.  These are more general than
  simple flips.

  Edge removal and multi-face removal operations can be represented by
  sequences of 2-3, 3-2, 2-2 and 4-4 flips.
  Although a given composite operation may improve the mesh,
  some of its individual flips may temporarily reduce the quality.
  (They may even invert elements.)
  Thus using such composite operations is likely to produce better
  results than using only simple topological changes.


  \section mesh_simplicial_Edge_Removal_Interior Edge Removal for Interior Edges

  For interior edges, edge removal replaces \f$ n \f$ tetrahedra with
  \f$ 2(n-2) \f$ tetrahedra.  This includes 3-2, 4-4, 5-6 and other flips.

  The ring of vertices around the edge are triangulated using Klincsek's
  algorithm to produce an optimal triangulation.

  \image html edge_removal.jpg "Edge removal that produces a 6-8 flip."
  \image latex edge_removal.pdf "Edge removal that produces a 6-8 flip." width=0.9\textwidth



  \section mesh_simplicial_Edge_Removal_Boundary Edge Removal for Boundary Edges

  For boundary edges, an edge is swapped on two approximately co-planar
  boundary faces.  It replaces \f$ n \f$ tetrahedra with \f$ 2(n-1) \f$
  tetrahedra.  This includes 2-2, 3-4, 4-6 and other flips.

  Boundary edge removal changes the volume of the mesh.  It is performed
  only if the surface is approximately planar.
  In our implementation, boundary edges may be removed if the angle
  between the normals of the two adjacent boundary faces is less than a
  user specified constant.

  \image html edge_removal_boundary.jpg "Edge removal that produces a 4-6 flip.  An edge removal that changes the volume of the mesh."
  \image latex edge_removal_boundary.pdf "Edge removal that produces a 4-6 flip.  An edge removal that changes the volume of the mesh." width=0.9\textwidth

  We have implemented edge removal for interior and boundary edges.



  \section mesh_simplicial_Multi_Face_Removal Multi-Face Removal

  Multi-face removal is the inverse of edge removal.  It removes the faces
  that are sandwiched between two vertices and inserts an edge between the
  vertices.  The optimal set of faces is determined with Shewchuk's
  algorithm.

  \image html face_removal.jpg "Face removal that produces an 8-6 flip."
  \image latex face_removal.pdf "Face removal that produces an 8-6 flip." width=0.9\textwidth

  Currently in our implementation only one or two faces can be removed
  by face removal.  We will implement the general multi-face removal algorithm.



  \section mesh_simplicial_Topological_Optimization Topological Optimization

  We apply local topological transformations in a hill-climbing method.
  We can use this to find a (local) maximum quality mesh.

  One can repeatedly sweep over all the edges and faces, applying edge and
  face removal.  However this is very inefficient.  A better approach is to
  keep track of the edges and faces upon which local topological changes could
  possibly improve the mesh.

  We maintain a set of active tetrahedra.
  Tetrahedra are added to the active set upon insertion in the mesh.
  Tetrahedra are removed from the set when they are removed from the mesh
  or when it is determined that edge or face removal on its edges and
  faces will not improve the mesh.



  \section mesh_simplicial_DS Data Structures

  The geom::SimpMeshRed class is a reduced representation of a simplicial
  mesh.


  Use the classs by including the file mesh/simplicial.h or by
  including mesh.h.
*/

#if !defined(__geom_mesh_simplicial_h__)
#define __geom_mesh_simplicial_h__

#include "stlib/geom/mesh/simplicial/SmrCell.h"
#include "stlib/geom/mesh/simplicial/SmrNode.h"
#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"

#include "stlib/geom/mesh/simplicial/accessors.h"
#include "stlib/geom/mesh/simplicial/build.h"
#include "stlib/geom/mesh/simplicial/coarsen.h"
#include "stlib/geom/mesh/simplicial/file_io.h"
#include "stlib/geom/mesh/simplicial/flip.h"
#include "stlib/geom/mesh/simplicial/geometry.h"
#include "stlib/geom/mesh/simplicial/inc_opt.h"
#include "stlib/geom/mesh/simplicial/insert.h"
#include "stlib/geom/mesh/simplicial/laplacian.h"
#include "stlib/geom/mesh/simplicial/manipulators.h"
#include "stlib/geom/mesh/simplicial/quality.h"
#include "stlib/geom/mesh/simplicial/refine.h"
#include "stlib/geom/mesh/simplicial/set.h"
#include "stlib/geom/mesh/simplicial/tile.h"
#include "stlib/geom/mesh/simplicial/topologicalOptimize.h"
#include "stlib/geom/mesh/simplicial/topology.h"
#include "stlib/geom/mesh/simplicial/transform.h"
#include "stlib/geom/mesh/simplicial/valid.h"

#endif
