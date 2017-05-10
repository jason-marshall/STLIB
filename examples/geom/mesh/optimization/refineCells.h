// -*- C++ -*-

/*!
  \file refineCells.h
  \brief Refine the specified cells by splitting edges.
*/

/*!
  \page examples_geom_mesh_refineCells Refine the specified cells by splitting edges.

  <!-------------------------------------------------------------------------->
  \section examples_geom_mesh_refineCellsUsage Usage.

  The angle options differ depending on the space dimension and
  simplex dimension.  For triangle meshes in 2-D, the angle is used to
  determine corner features.

  \verbatim
  refineCells22.exe [-length=length] [-manifold=manifold] [-featureDistance=d]
    [-angle=maxAngleDeviation]
    cells inputMesh outputMesh
  \endverbatim

  For triangle meshes in 3-D:
  The dihedral angle is used to determine edge features.
  The solid angle is used to determine corner features.
  The boundary angle is used to determine boundary corner features.

  \verbatim
  refineCells32.exe [-length=length] [-manifold=manifold] [-featureDistance=d]
    [-dihedralAngle=maxDihedralAngleDeviation]
    [-solidAngle=maxSolidAngleDeviation]
    cells inputMesh outputMesh
  \endverbatim

  For tetrahedron meshes in 3-D,
  the dihedral angle is used to determine edge
  features and the solid angle is used to determine corner features.

  \verbatim
  refineCells33.exe [-length=length] [-manifold=manifold] [-featureDistance=d]
    [-dihedralAngle=maxDihedralAngleDeviation]
    [-solidAngle=maxSolidAngleDeviation]
    [-boundaryAngle=maxBoundaryAngleDeviation]
    cells inputMesh outputMesh
  \endverbatim

  - length is used to specify the maximum allowed edge length.
    If specified, only cells whose longest edge exceeds this length
    will be refined.
  - cells contains the indices of the cells to split.
  - inputMesh is the input mesh.
  - outputMesh is the output mesh.



  <!-------------------------------------------------------------------------->
  \section examples_geom_mesh_refineCells2D 2-D Examples.

  Recall that a cell refines by splitting its longest edge.  This refines all
  cells incident to the edge.  (In 2-D, there are either one or two incident
  cells.  The 3-D, there may be one or more incident cells.)  Further,
  a cell may only be refined by splitting its longest edge.  This means
  that in order to refine a particular cell, others may have to be refined
  first.

  To illustrate this, we start with a nautilus mesh.

  \verbatim
  cp ../../../data/geom/mesh/22/nautilus.txt .
  \endverbatim

  \image html refine_cells_2_nautilus.jpg "A nautilus mesh."
  \image latex refine_cells_2_nautilus.pdf "A nautilus mesh."

  We refine the smallest triangle.  This causes recursive refinement of all
  the other triangles.

  \verbatim
  refineCells22.exe indices nautilus.txt nautilus_r.txt
  \endverbatim

  The indices file specifies that the first cell should be split.  The
  contents of the file are shown below.  The first line specifies the
  number of cells; the second gives the cell index.

  \verbatim
  1
  0
  \endverbatim

  \image html refine_cells_2_nautilus_r.jpg "After refining the smallest triangle."
  \image latex refine_cells_2_nautilus_r.pdf "After refining the smallest triangle."



  An adjacent neighbor may have to be refined multiple times before the target
  cell may undergo edge splitting.
  We start with a wedge-shaped mesh.

  \verbatim
  cp ../../../data/geom/mesh/22/wedge.txt .
  \endverbatim

  \image html refine_cells_2_wedge.jpg "A wedge-shaped mesh."
  \image latex refine_cells_2_wedge.pdf "A wedge-shaped mesh."

  We refine the smaller triangle.  The adjacent cell must be refined three
  times before the the shared edge may be split.

  \verbatim
  refineCells22.exe indices wedge.txt wedge_r.txt
  \endverbatim

  \image html refine_cells_2_wedge_r.jpg "After refining the smaller triangle."
  \image latex refine_cells_2_wedge_r.pdf "After refining the smaller triangle."



  Although adjacent neighbors may be recursively refined in refining the
  target cell, the refinement typically stays local.
  To demonstrate this, we start with a mesh of the letter A and repeatedly
  refine the first cell in the mesh.

  \verbatim
  cp ../../../data/geom/mesh/22/a_23.txt a.txt
  \endverbatim

  \image html refine_cells_2_a.jpg "The initial mesh."
  \image latex refine_cells_2_a.pdf "The initial mesh."

  The refinement stays local when we repeatedly refine the first cell in
  the mesh.

  \verbatim
  refineCells22.exe indices a.txt a_r.txt
  refineCells22.exe indices a_r.txt a_r.txt
  refineCells22.exe indices a_r.txt a_r.txt
  refineCells22.exe indices a_r.txt a_r.txt
  refineCells22.exe indices a_r.txt a_r.txt
  refineCells22.exe indices a_r.txt a_r.txt
  refineCells22.exe indices a_r.txt a_r.txt
  refineCells22.exe indices a_r.txt a_r.txt
  refineCells22.exe indices a_r.txt a_r.txt
  refineCells22.exe indices a_r.txt a_r.txt
  \endverbatim

  \image html refine_cells_2_a_r.jpg "The refined mesh."
  \image latex refine_cells_2_a_r.pdf "The refined mesh."






  <!-------------------------------------------------------------------------->
  \section examples_geom_mesh_refineCellsSimple3D Simple 3-D Example.

  We demonstrate cell refinement in 3-D.  We start with an equilateral
  tetrahedron.  In the figures below we show the modified condition number
  quality metric.

  \verbatim
  cp ../../../data/geom/mesh/33/equilateral.txt initial.txt
  \endverbatim

  \image html tetrahedronRefineInitial.jpg "The initial tetrahedron mesh."
  \image latex tetrahedronRefineInitial.pdf "The initial tetrahedron mesh."

  Refining the first cell will split one of the edges.

  \verbatim
  refineCells33.exe indices initial.txt refined01.txt
  \endverbatim

  \image html tetrahedronRefine01.jpg "One step of cell refinement."
  \image latex tetrahedronRefine01.pdf "One step of cell refinement."

  Below we show the mesh after ten and twenty cell refinement steps.
  For each mesh, the first cell is in one of the corners of the tetrahedron.
  We note that the refinement stays local (that is, the mesh is not uniformly
  refined).  The longest edge rule helps to maintain the quality of the
  mesh.  For the final mesh with 87 cells, the minimum modified condition
  number is 0.42; the mean is 0.67.

  \verbatim
  refineCells33.exe indices refined01.txt refined02.txt
  ...
  refineCells33.exe indices refined19.txt refined20.txt
  \endverbatim

  \image html tetrahedronRefine10.jpg "Ten steps of cell refinement."
  \image latex tetrahedronRefine10.pdf "Ten steps of cell refinement."

  \image html tetrahedronRefine20.jpg "Twenty steps of cell refinement."
  \image latex tetrahedronRefine20.pdf "Twenty steps of cell refinement."



  <!-------------------------------------------------------------------------->
  \section examples_geom_mesh_refineCellsCylinder Cylinder Example.

  Now for a more sophisticated example.  We start with a coarse mesh of a
  cylinder with a radius of 1 cm and a height of 5 cm.  The mesh was generated
  in Cubit using a target edge length of 1 cm.

  \verbatim
  cp ../../../data/geom/mesh/33/cylinderR1H5L1.txt mesh.txt
  \endverbatim

  \image html cylinderRefineCellsMesh.jpg "The initial mesh.  The minimum modified condition number is 0.85; the mean is 0.94.  There are 102 elements."
  \image latex cylinderRefineCellsMesh.pdf "The initial mesh.  The minimum modified condition number is 0.85; the mean is 0.94.  There are 102 elements."

  For the boundary, we use a surface mesh with a target edge length of 1 mm.
  We use a dihedral angle of 0.5 to define edge features.

  \verbatim
  cp ../../../data/geom/mesh/32/cylinderR1H5L0.1.txt boundary.txt
  utility/extractFeatures32.exe -dihedralAngle=0.5 boundary.txt edges.txt corners.txt
  \endverbatim

  \image html cylinderRefineCellsBoundary.jpg "The boundary mesh and the edge features."
  \image latex cylinderRefineCellsBoundary.pdf "The boundary mesh and the edge features."

  Suppose that we will do an impact experiment with the cylinder and wish
  to have a fine mesh near the base and a coarse mesh near the top.
  Specifically, we will refine the mesh so that the cells in the bottom
  2 cm have a maximum
  edge length of 5 mm, the cells in the bottom 1 cm have a maximum edge
  length of 2.5 mm, and the cells in the bottom 5 mm have a maximum edge
  length of 1.25 mm.

  First we compute the centroids of the cells in the mesh.

  \verbatim
  utility/computeCentroids33.exe mesh.txt centroids.txt
  \endverbatim

  We use simple mesh that is a square positioned at the base of the cylinder.
  The distance from the square is height along the cylinder.

  \verbatim
  utility/computeSignedDistance3.exe square centroids.txt tmp.txt
  \endverbatim

  We add an attribute name to the distance file.

  \verbatim
  echo "Signed Distance" >name.txt
  cat name.txt tmp.txt >signedDistance.txt
  \endverbatim

  We select the cells with signed distance no greater than 2 cm.

  \verbatim
  utility/selectCells.exe -upper=2 signedDistance.txt indices.txt
  \endverbatim

  Then we refine the selected cells whose edge length exceed 5 mm.

  \verbatim
  optimization/refineCells33.exe -length=0.5 -manifold=boundary.txt -dihedralAngle=0.5 -featureDistance=0.1 indices.txt mesh.txt refined.txt
  \endverbatim

  Each of these cells will be refined at
  least once.  Because most of the cells have multiple edges that are too long,
  the above sequence of commands must be repeated until refineCells33.exe
  produces no further refinement.  Below is the result of refining the mesh in
  the bottom 2 cm.

  Following the refinement, we apply a sweep of topological and then goemetric
  optimization.

  \verbatim
  optimization/topologicalOptimize3.exe -manifold=boundary.txt -angle=0.5 refined.txt refined.txt
  optimization/geometricOptimize3.exe -boundary=boundary.txt -dihedralAngle=0.5 refined.txt refined.txt
  \endverbatim

  \image html cylinderRefineCellsRefine1.jpg "Cells refined in the bottom 2 cm.  The minimum modified condition number is 0.49; the mean is 0.82.  There are 2,594 elements."
  \image latex cylinderRefineCellsRefine1.pdf "Cells refined in the bottom 2 cm.  The minimum modified condition number is 0.49; the mean is 0.82.  There are 2,594 elements."

  Note that the mesh conforms better to the boundary as we refine it.
  Next we refine the cells in the bottom 1 cm using a maximum edge length
  of 2.5 mm.

  \image html cylinderRefineCellsRefine2.jpg "Cells refined in the bottom 1 cm.  The minimum modified condition number is 0.44; the mean is 0.83.  There are 12,084 elements."
  \image latex cylinderRefineCellsRefine2.pdf "Cells refined in the bottom 1 cm.  The minimum modified condition number is 0.44; the mean is 0.83.  There are 12,084 elements."

  Finally we refine the cells in the bottom 5 mm using a maximum edge length
  of 1.25 mm.

  \image html cylinderRefineCellsRefine3.jpg "Cells refined in the bottom 5 mm.  The minimum modified condition number is 0.49; the mean is 0.84.  There are 49,714 elements."
  \image latex cylinderRefineCellsRefine3.pdf "Cells refined in the bottom 5 mm.  The minimum modified condition number is 0.49; the mean is 0.84.  There are 49,714 elements."




  <!-------------------------------------------------------------------------->
  \section examples_geom_mesh_refineCellsEnterprise Enterprise Example.

  Now suppose that we were going to do an impact experiment with the
  Enterprise and a wall.  We use Cubit to generate meshes for the the
  two objects.  In each, the tetrahedron edges lengths are about 50 meters.
  The meshes are shown below.  The Enterprise mesh has 4,926 elements; the
  mesh for the wall has 3,128 elements.

  \verbatim
  cp ../../../data/geom/mesh/33/enterpriseL50.txt enterprise.txt
  cp ../../../data/geom/mesh/33/enterpriseWallL50.txt wall.txt
  \endverbatim

  \image html enterpriseWall.jpg "The Enterprise and the wall.  The minimum modified condition numbers are 0.069 and 0.51; the means are 0.82 and 0.82."
  \image latex enterpriseWall.pdf "The Enterprise and the wall.  The minimum modified condition numbers are 0.069 and 0.51; the means are 0.82 and 0.82."

  We will refine the meshes near the impact region.  Because the Enterprise
  has interesting geometry, we will use a finer mesh of the boundary in
  the refinement process.  This boundary mesh is shown below.  Because the
  wall does not have any curved surfaces, there is no need for a better
  representation of its surface.  We will simply extract a triangle surface
  mesh from the tetrahedron mesh.

  \verbatim
  cp ../../../data/geom/mesh/32/enterpriseL20.txt enterpriseBoundary.txt
  utility/boundary33.exe wall.txt wallBoundary.txt
  \endverbatim

  \image html enterpriseBoundary.jpg "A finer boundary mesh for the Enterprise."
  \image latex enterpriseBoundary.pdf "A finer boundary mesh for the Enterprise."

  Next we determine a suitable dihedral angle parameter for characterizing
  the edge and corner features of the Enterprise model.
  We check that an angle deviation of 0.5 captures the relevant features
  without introducing spurious edges and corners.  Below the edge features
  are shown in white and the corner features are shown in red.

  \verbatim
  utility/extractFeatures32.exe -dihedralAngle=0.5 enterpriseBoundary.txt enterpriseEdges.txt enterpriseCorners.txt
  \endverbatim

  \image html enterpriseFeatures.jpg "Edge and corner features for the Enterprise for a dihedral angle deviation of 0.5."
  \image latex enterpriseFeatures.pdf "Edge and corner features for the Enterprise for a dihedral angle deviation of 0.5."

  Before we refine the meshes near the impact region, we will improve the
  quality of the Enterprise mesh.  Overall, it has high quality elements,
  but it has a couple problem areas due to the geometry of the object.  The
  thin rectangular beams are one issue.  The target edge length (50 meters)
  is too long to mesh the beams with high quality elements.  Neither geometric
  nor topological optimization can fix this problem.  The only solution is to
  refine the mesh.  However, we don't want to refine the entire mesh, we
  only want to refine in the problem areas.  To achieve this, we select the
  cells with relatively poor quality (modified condition number less than 0.4)
  and refine any cell whose maximum edge length exceeds 10 meters.  We repeat
  this until no more cells are refined.  We also apply topological and
  geometric optimization between refinements.  Below is an example step in
  the procedure.

  \verbatim
  optimization/topologicalOptimize3.exe -function=c -manifold=enterpriseBoundary.txt -angle=0.5 enterprise.txt enterprise.txt
  optimization/geometricOptimize3.exe -function=c -boundary=enterpriseBoundary.txt -dihedralAngle=0.5 enterprise.txt enterprise.txt
  utility/cellAttributes33.exe -mcn enterprise.txt mcn.txt
  utility/selectCells.exe -upper=0.4 mcn.txt indices.txt
  optimization/refineCells33.exe -length=10 -manifold=enterpriseBoundary.txt -dihedralAngle=0.5 indices.txt enterprise.txt enterprise.txt
  \endverbatim

  \image html enterpriseFixed.jpg "The improved mesh for the Enterprise.  The minimum modified condition number is 0.36; the mean is 0.83.  The mesh has 7,065 elements."
  \image latex enterpriseFixed.pdf "The improved mesh for the Enterprise.  The minimum modified condition number is 0.36; the mean is 0.83.  The mesh has 7,065 elements."

  Now we will refine the meshes near the impact region.  To define what is
  near the impact region in the mesh of the enterprise, we compute signed
  distance from the boundary of the wall.  Likewise, to define what is
  near the impact region in the mesh of the wall, we compute signed
  distance from the boundary of the Enterprise.  We constrain that cells
  within 200 meters of the other object should have edge lengths no longer
  than 40 meters.  For cells within 100, 50, and 25 meters we use maximum
  edge lengths of 20, 10, and 5 meters, respectively.  Below is an
  example command of how we refine the Enterprise mesh.  For more details
  check out the makefile in the stlib/results/geom/mesh/3/enterpriseWall
  directory.

  \verbatim
  utility/computeCentroids33.exe enterprise.txt centroids.txt
  utility/computeSignedDistance3.exe wallBoundary.txt centroids.txt tmp.txt
  echo "Signed Distance" >name.txt
  cat name.txt tmp.txt >signedDistance.txt
  utility/selectCells.exe -upper=200 signedDistance.txt indices.txt
  optimization/refineCells33.exe -length=40 -manifold=enterpriseBoundary.txt -dihedralAngle=0.5 indices.txt enterprise.txt enterpriseRefine.txt
  \endverbatim

  We take analogous steps to refine the mesh for the wall.  The only difference
  is that we don't supply a boundary mesh for the refinement.

  \verbatim
  utility/computeCentroids33.exe wall.txt centroids.txt
  utility/computeSignedDistance3.exe enterpriseBoundary.txt centroids.txt tmp.txt
  echo "Signed Distance" >name.txt
  cat name.txt tmp.txt >signedDistance.txt
  utility/selectCells.exe -upper=200 signedDistance.txt indices.txt
  optimization/refineCells33.exe -length=40 indices.txt wall.txt wallRefine.txt
  \endverbatim

  Below we show the meshes after each of the four stages of refinement.

  \image html enterpriseWallRefine1.jpg "Refinement in the cells within 200 meters of contact.  The minimum modified condition numbers are 0.36 and 0.38; the means are 0.81 and 0.76.  The meshes have 10,320 and 14,063 elements, respectively."
  \image latex enterpriseWallRefine1.pdf "Refinement in the cells within 200 meters of contact.  The minimum modified condition numbers are 0.36 and 0.38; the means are 0.81 and 0.76.  The meshes have 10,320 and 14,063 elements, respectively."

  \image html enterpriseWallRefine2.jpg "Refinement in the cells within 100 meters of contact.  The minimum modified condition numbers are 0.36 and 0.41; the means are 0.79 and 0.77.  The meshes have 18,519 and 42,014 elements, respectively."
  \image latex enterpriseWallRefine2.pdf "Refinement in the cells within 100 meters of contact.  The minimum modified condition numbers are 0.36 and 0.41; the means are 0.79 and 0.77.  The meshes have 18,519 and 42,014 elements, respectively."

  \image html enterpriseWallRefine3.jpg "Refinement in the cells within 50 meters of contact.  The minimum modified condition numbers are 0.36 and 0.41; the means are 0.78 and 0.76.  The meshes have 36,829 and 98,303 elements, respectively."
  \image latex enterpriseWallRefine3.pdf "Refinement in the cells within 50 meters of contact.  The minimum modified condition numbers are 0.36 and 0.41; the means are 0.78 and 0.76.  The meshes have 36,829 and 98,303 elements, respectively."

  \image html enterpriseWallRefine4.jpg "Refinement in the cells within 25 meters of contact.  The minimum modified condition numbers are 0.36 and 0.35; the means are 0.78 and 0.75.  The meshes have 66,504 and 178,010 elements, respectively."
*/

#include "../smr_io.h"
#include "../iss_io.h"

#include "stlib/geom/mesh/simplicial/refine.h"
#include "stlib/geom/mesh/simplicial/quality.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <string>
#include <sstream>

#include <cassert>

using namespace stlib;

namespace {

// Global variables.

//! The program name.
std::string programName;

// Local functions.

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " [-length=l] [-manifold=m] [-featureDistance=d]\n"
#if SPACE_DIMENSION == 2 && SIMPLEX_DIMENSION == 2
         << "[-angle=maxAngleDeviation] \n"
#elif SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 3
         << "[-dihedralAngle=maxDihedralAngleDeviation] \n"
         << "[-solidAngle=maxSolidAngleDeviation] \n"
#else // SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 2
         << "[-dihedralAngle=maxDihedralAngleDeviation] \n"
         << "[-solidAngle=maxSolidAngleDeviation] \n"
         << "[-boundaryAngle=maxBoundaryAngleDeviation] \n"
#endif
         << "cells inputMesh outputMesh\n"
         << "- length is used to specify the maximum allowed edge length.\n"
         << "  If specified, only cells whose longest edge exceeds this length\n"
         << "  will be refined.\n"
         << "- If specified, inserted nodes are moved to lie on the manifold.\n"
         << "- The feature distance determines how close a vertex must be to a \n"
         << "  corner or edge to be placed on that feature.\n"
#if SPACE_DIMENSION == 2 && SIMPLEX_DIMENSION == 2
         << "- The angle is used to determine corner features.\n"
#elif SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 3
         << "- The dihedral angle is used to determine edge features.\n"
         << "- The solid angle is used to determine corner features.\n"
#else // SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 2
         << "- The dihedral angle is used to determine edge features.\n"
         << "- The solid angle is used to determine corner features.\n"
         << "- The boundary angle is used to determine boundary corner features.\n"
#endif
         << "- cells contains the indices of the cells to split.\n"
         << "- inputMesh is the input mesh.\n"
         << "- outputMesh is the output mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
  static_assert((SPACE_DIMENSION == 2 && SIMPLEX_DIMENSION == 2) ||
                (SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 2) ||
                (SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 3),
                "Those dimensions are not supported.");

   typedef geom::SimpMeshRed<SPACE_DIMENSION, SIMPLEX_DIMENSION> Mesh;
   typedef Mesh::Vertex Vertex;
#if SPACE_DIMENSION == SIMPLEX_DIMENSION
   typedef geom::IndSimpSet < SPACE_DIMENSION, SIMPLEX_DIMENSION - 1 >
   BoundaryMesh;
   typedef geom::PointsOnManifold < SPACE_DIMENSION, SIMPLEX_DIMENSION - 1, 1 >
   BoundaryManifold;
#else // SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 2
   typedef geom::IndSimpSet<SPACE_DIMENSION, SIMPLEX_DIMENSION>
   BoundaryMesh;
   typedef geom::PointsOnManifold<SPACE_DIMENSION, SIMPLEX_DIMENSION, 1>
   BoundaryManifold;
#endif

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   // The maximum edge length.
   double length = 0;
   if (parser.getOption("length", &length) && length <= 0) {
      std::cerr << "Bad length.  You specified " << length << "\n"
                << "The length must be positive.\n";
      exitOnError();
   }

   double featureDistance = 0;
   if (parser.getOption("featureDistance", &featureDistance)) {
      if (featureDistance <= 0) {
         std::cerr << "Bad feature distance.\n";
         exitOnError();
      }
   }

   // If they did not specify the input mesh, the cells and output mesh.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the cell indices.
   std::vector<std::size_t> cells;
   {
      std::ifstream file(parser.getArgument().c_str());
      if (! file) {
         std::cerr << "Bad input file.  Exiting...\n";
         exitOnError();
      }

      std::size_t size;
      file >> size;
      cells.resize(size);
      for (std::size_t n = 0; n != size; ++n) {
         cells[n] = -1;
         file >> cells[n];
      }
   }

   // Read the input mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   //
   // Build the boundary manifold.
   //
   std::string boundaryFile;
   BoundaryManifold* boundaryManifold = 0;
   if (parser.getOption("manifold", &boundaryFile)) {
      // Read the mesh describing the boundary.
      BoundaryMesh boundaryMesh;
      readAscii(boundaryFile.c_str(), &boundaryMesh);

#if SPACE_DIMENSION == 2 && SIMPLEX_DIMENSION == 2
      // Get the angle deviation for corner features.
      double maxAngleDeviation = -1;
      parser.getOption("angle", &maxAngleDeviation);
      // Build the boundary manifold.
      boundaryManifold = new BoundaryManifold(boundaryMesh, maxAngleDeviation);
      // If the feature distance was specified.
      if (featureDistance != 0) {
         // Set the maximum corner distance.
         boundaryManifold->setMaxCornerDistance(featureDistance);
      }
      // Register the boundary vertices of the mesh.
      boundaryManifold->insertBoundaryVertices(&mesh);

#elif SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 3
      double maxDihedralAngleDeviation = -1;
      parser.getOption("dihedralAngle", &maxDihedralAngleDeviation);
      double maxSolidAngleDeviation = -1;
      parser.getOption("solidAngle", &maxSolidAngleDeviation);
      // Build the boundary manifold.
      boundaryManifold = new BoundaryManifold(boundaryMesh,
                                              maxDihedralAngleDeviation,
                                              maxSolidAngleDeviation);
      // If the feature distance was specified.
      if (featureDistance != 0) {
         // Set the maximum corner distance.
         boundaryManifold->setMaxCornerDistance(featureDistance);
         // Set the maximum edge distance.
         boundaryManifold->setMaxEdgeDistance(featureDistance);
      }
      // Register the boundary vertices and edges of the mesh.
      boundaryManifold->insertBoundaryVerticesAndEdges
      (&mesh, maxDihedralAngleDeviation);

#else // SPACE_DIMENSION == 3 && SIMPLEX_DIMENSION == 2
      double maxDihedralAngleDeviation = -1;
      parser.getOption("dihedralAngle", &maxDihedralAngleDeviation);
      double maxSolidAngleDeviation = -1;
      parser.getOption("solidAngle", &maxSolidAngleDeviation);
      double maxBoundaryAngleDeviation = -1;
      parser.getOption("boundaryAngle", &maxBoundaryAngleDeviation);
      // Build the boundary manifold.
      boundaryManifold = new BoundaryManifold(boundaryMesh,
                                              maxDihedralAngleDeviation,
                                              maxSolidAngleDeviation,
                                              maxBoundaryAngleDeviation);
      // If the feature distance was specified.
      if (featureDistance != 0) {
         // Set the maximum corner distance.
         boundaryManifold->setMaxCornerDistance(featureDistance);
         // Set the maximum edge distance.
         boundaryManifold->setMaxEdgeDistance(featureDistance);
      }
      // Register the vertices and edges of the surface mesh.
      boundaryManifold->insertVerticesAndEdges(&mesh, maxDihedralAngleDeviation);
#endif
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Print quality measures for the input mesh.
   std::cout << "Quality of the input mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);
   std::cout << "Number of edges = " << mesh.computeEdgesSize() << "\n";
   geom::printEdgeLengthStatistics(std::cout, mesh);

   std::cout << "\nRefining the mesh...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Refine the mesh.
   std::size_t count;
   if (length != 0) {
      count = geom::refine(&mesh, boundaryManifold, cells.begin(), cells.end(),
                           ads::constructUnaryConstant<Vertex, double>(length));
   }
   else {
      count = geom::refine(&mesh, boundaryManifold, cells.begin(), cells.end());
   }

   double elapsedTime = timer.toc();
   std::cout << "done.\n"
             << "Number of edges split = " << count << "\n"
             << "Refining took " << elapsedTime << " seconds.\n";

   // Print quality measures for the refined mesh.
   std::cout << "\nQuality of the refined mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);
   std::cout << "Number of edges = " << mesh.computeEdgesSize() << "\n";
   geom::printEdgeLengthStatistics(std::cout, mesh);

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   // Delete the boundary manifold if it was allocated.
   if (boundaryManifold != 0) {
      delete boundaryManifold;
      boundaryManifold = 0;
   }

   return 0;
}
