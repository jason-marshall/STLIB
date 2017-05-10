// -*- C++ -*-

/*!
  \file refine.h
  \brief Refine the mesh by splitting edges.
*/

/*!
  \page examples_geom_mesh_refine Refine the mesh by splitting edges.

  The maximum allowed edge length is specified on the command line.
  The mesh is refined until no edges exceed this length.


  \section examples_geom_mesh_refine_usage Usage.

  The angle options differ depending on the space dimension and
  simplex dimension.  For triangle meshes in 2-D, the angle is used to
  determine corner features.

  \verbatim
  refine22.exe [-length=length] [-manifold=manifold]
    [-mesh=mesh -field=field] [-featureDistance=d]
    [-angle=maxAngleDeviation]
    inputMesh outputMesh
  \endverbatim

  For triangle meshes in 3-D:
  The dihedral angle is used to determine edge features.
  The solid angle is used to determine corner features.
  The boundary angle is used to determine boundary corner features.

  \verbatim
  refine32.exe [-length=length] [-manifold=manifold]
    [-mesh=mesh -field=field] [-featureDistance=d]
    [-dihedralAngle=maxDihedralAngleDeviation]
    [-solidAngle=maxSolidAngleDeviation]
    inputMesh outputMesh
  \endverbatim

  For tetrahedron meshes in 3-D,
  the dihedral angle is used to determine edge
  features and the solid angle is used to determine corner features.

  \verbatim
  refine33.exe [-length=length] [-manifold=manifold]
    [-mesh=mesh -field=field] [-featureDistance=d]
    [-dihedralAngle=maxDihedralAngleDeviation]
    [-solidAngle=maxSolidAngleDeviation]
    [-boundaryAngle=maxBoundaryAngleDeviation]
    inputMesh outputMesh
  \endverbatim

  One can either specify a length or a mesh and a field to define
  the maximum allowed edge length.
  - length is used to specify the maximum edge length.
    If no length is specified, the maximum edge length will be set to
    0.99 times the average input edge length.
  - If specified, inserted nodes are moved to lie on the manifold.
  - mesh and field are used to define the maximum edge length function.
  - The feature distance determines how close a vertex must be to a
    corner or edge to be placed on that feature.
  - inputMesh is the input mesh.
  - outputMesh is the output mesh.




  \section examples_geom_mesh_refine_semiAnnulus 3-D Example: Refinement Using a Boundary Description

  We start with a coarse mesh of a semi-annulus.  The inner radius is 1,
  the outer radius is 2.  It has unit height and the edge lengths are close
  to 1.  The mesh has 39 elements.  In the figures below, we show the modified
  condition number of the elements.

  \verbatim
  cp ../../../data/geom/mesh/33/semiAnnulusR1R2H1E1.txt initial.txt
  \endverbatim

  \image html semiAnnulusInitial.jpg "The initial mesh."
  \image latex semiAnnulusInitial.pdf "The initial mesh."

  We have a higher resolution description of the boundary of the semi-annulus.
  For this surface mesh, the edge lengths are close to 0.1.

  \verbatim
  cp ../../../data/geom/mesh/32/semiAnnulusR1R2H1E0.1.txt boundary.txt
  \endverbatim

  \image html semiAnnulusBoundary.jpg "The boundary mesh."
  \image latex semiAnnulusBoundary.pdf "The boundary mesh."

  First we use geometric optimization to improve the quality of the initial
  mesh.

  \verbatim
  geometricOptimize3.exe -boundary=boundary.txt -featureDistance=0.01 -dihedralAngle=1 -sweeps=5 initial.txt mesh.txt
  \endverbatim

  \image html semiAnnulusMesh.jpg "The mesh after geometric optimization."
  \image latex semiAnnulusMesh.pdf "The mesh after geometric optimization."

  Now we are going to refine the mesh.  As we do so, we will use our high
  resolution description of the boundary.  When a boundary edge is split,
  the new vertex will either be inserted into an edge feature or a surface
  feature of the boundary mesh.  We perform edge refinement with maximum
  edge lengths of 0.9, 0.5 and 0.2.  This results in meshes with 350, 1997,
  and 28051 elements, respectively.

  \verbatim
  refine33.exe -featureDistance=0.001 -length=0.9 -manifold=boundary.txt -dihedralAngle=1 mesh.txt refine0.9.txt
  refine33.exe -featureDistance=0.001 -length=0.5 -manifold=boundary.txt -dihedralAngle=1 mesh.txt refine0.5.txt
  refine33.exe -featureDistance=0.001 -length=0.2 -manifold=boundary.txt -dihedralAngle=1 mesh.txt refine0.2.txt
  \endverbatim

  \image html semiAnnulusRefine0.9.jpg "The refined mesh with a maximum allowed edge length of 0.9."
  \image latex semiAnnulusRefine0.9.pdf "The refined mesh with a maximum allowed edge length of 0.9."

  \image html semiAnnulusRefine0.5.jpg "The refined mesh with a maximum allowed edge length of 0.5."
  \image latex semiAnnulusRefine0.5.pdf "The refined mesh with a maximum allowed edge length of 0.5."

  \image html semiAnnulusRefine0.2.jpg "The refined mesh with a maximum allowed edge length of 0.2."
  \image latex semiAnnulusRefine0.2.pdf "The refined mesh with a maximum allowed edge length of 0.2."

  We see that as we refine the mesh, we get a more accurate representation of
  the shape of the semi-annulus.  We can improve the quality of the refined
  mesh with topological and geometric optimization.

  \verbatim
  topologicalOptimize3.exe -manifold=boundary.txt -angle=1 refine0.2.txt optimized.txt
  geometricOptimize3.exe -boundary=boundary.txt -dihedralAngle=1 optimized.txt optimized.txt
  topologicalOptimize3.exe -manifold=boundary.txt -angle=1 optimized.txt optimized.txt
  geometricOptimize3.exe -boundary=boundary.txt -dihedralAngle=1 optimized.txt optimized.txt
  \endverbatim

  \image html semiAnnulusOptimized.jpg "The refined mesh after topological and geometric optimization."
  \image latex semiAnnulusOptimized.pdf "The refined mesh after topological and geometric optimization."








  \section examples_geom_mesh_refine_example2 2-D Example: Uniform Refinement

  We start with a coarse mesh.
  This mesh has 23 vertices and 23 triangles.  The minimum condition
  number is 0.53; the mean is 0.71.

  \verbatim
  cp ../../../data/geom/mesh/22/a_23.txt mesh.txt
  \endverbatim

  \image html refine_max_length_2_mesh.jpg "The coarse mesh."
  \image latex refine_max_length_2_mesh.pdf "The coarse mesh."

  We refine the mesh so no edge is longer than 0.1.  95 edges are split to
  yield 118 vertices and 157 triangles.  The minimum condition
  number is 0.53; the mean is 0.79.

  \verbatim
  refine2.exe -lengh=0.1 mesh.txt mesh_0.1.txt
  \endverbatim

  \image html refine_max_length_2_mesh_0.1.jpg "No edge longer than 0.1"
  \image latex refine_max_length_2_mesh_0.1.pdf "No edge longer than 0.1"

  If we choose a smaller edge lengths we obtain similar results.
  For a maximum edge length of 0.05, 369 edges are split to
  yield 392 vertices and 624 triangles.  The minimum condition
  number is 0.53; the mean is 0.80.

  \verbatim
  refine2.exe -length=0.05 mesh.txt mesh_0.05.txt
  \endverbatim

  \image html refine_max_length_2_mesh_0.05.jpg "No edge longer than 0.05"
  \image latex refine_max_length_2_mesh_0.05.pdf "No edge longer than 0.05"

  For a still smaller maximum edge length, the mesh quality improves further.
  For an edge length of 0.02, 2193 edges are split to
  yield 2216 vertices and 4096 triangles.  The minimum condition
  number is 0.67; the mean is 0.84.

  \verbatim
  refine2.exe -length=0.02 mesh.txt mesh_0.02.txt
  \endverbatim

  \image html refine_max_length_2_mesh_0.02.jpg "No edge longer than 0.02"
  \image latex refine_max_length_2_mesh_0.02.pdf "No edge longer than 0.02"




  \section examples_geom_mesh_refine_square 2-D Example: Non-uniform Refinement

  We start with a coarse mesh of the unit square.

  \verbatim
  cp ../../../data/geom/mesh/22/square1.txt mesh.txt
  \endverbatim

  \image html refine22_square.jpg "The coarse mesh."
  \image latex refine22_square.pdf "The coarse mesh."

  We use a mesh with vertices at (0,0), (1,0), and (0,1) with values
  0.01, 0.1, and 0.4, respectively, to define a linear function.
  We use this as the maximum allowed edge length function.

  \verbatim
  refine22.exe -mesh=function_mesh.txt -field=function_values.txt mesh.txt mesh_refined.txt
  \endverbatim

  \image html refine22_square_refined.jpg "The refined mesh."
  \image latex refine22_square_refined.pdf "The refined mesh."





  \section examples_geom_mesh_refine_brick 3-D Example: Non-uniform Refinement

  We start with a coarse mesh of a brick.  It has 179 elements.
  The minimum condition number is 0.33, the mean is 0.79.

  \verbatim
  cp ../../../data/geom/mesh/33/brick.txt mesh.txt
  \endverbatim

  \image html refine33_brick.jpg "The coarse mesh."
  \image latex refine33_brick.pdf "The coarse mesh."

  The lower and upper corners of the brick are (-5,-5,-5) and (5,5,5),
  respectively.
  We use a mesh with vertices at (-5,-5,-5), (5,-5,-5), (-5,5,-5), and
  (-5,-5,5), with values 0.1, 1, 2, and 4, respectively, to define a
  linear function.  We use this as the maximum allowed edge length function.

  \verbatim
  refine33.exe -mesh=function_mesh.txt -field=function_values.txt mesh.txt mesh_refined.txt
  \endverbatim

  The refined mesh has 8711 elements.
  The minimum condition number is 0.30, the mean is 0.73.

  \image html refine33_brick_refined.jpg "The refined mesh."
  \image latex refine33_brick_refined.pdf "The refined mesh."
*/

#include "../smr_io.h"
#include "../iss_io.h"

#include "stlib/geom/mesh/simplicial/refine.h"
#include "stlib/geom/mesh/simplicial/quality.h"
#include "stlib/geom/mesh/iss/ISS_Interpolate.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <string>
#include <sstream>

#include <cassert>

USING_STLIB_EXT_VECTOR_IO_OPERATORS;
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
         << programName << " [-length=length] [-manifold=manifold]\n"
         << "[-mesh=mesh -field=field] [-featureDistance=d] \n"
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
         << "inputMesh outputMesh\n"
         << "One can either specify a length or a mesh and a field to define\n"
         << "the maximum allowed edge length.\n"
         << "- length is used to specify the maximum edge length.\n"
         << "  If no length is specified, the maximum edge length will be set to\n"
         << "  0.99 times the average input edge length.\n"
         << "- If specified, inserted nodes are moved to lie on the manifold.\n"
         << "- mesh and field are used to define the maximum edge length function.\n"
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
   typedef geom::IndSimpSet<SPACE_DIMENSION, SPACE_DIMENSION> InterpolationMesh;
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
   // Input files.
   std::string meshFile;
   std::string fieldFile;
   // If they do not specify the length.
   if (! parser.getOption("length", &length)) {
      // They can specify a mesh and length as a field on that mesh.
      if (parser.getOption("mesh", &meshFile)) {
         if (! parser.getOption("field", &fieldFile)) {
            std::cerr << "If you specify the edge length mesh, you must also\n"
                      << "specify the edge field.\n";
            exitOnError();
         }
      }
   }

   double featureDistance = 0;
   if (parser.getOption("featureDistance", &featureDistance)) {
      if (featureDistance <= 0) {
         std::cerr << "Bad feature distance.\n";
         exitOnError();
      }
   }

   // If they did not specify the input and output mesh.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n";
      exitOnError();
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

   if (meshFile.empty() && length == 0) {
      double minLength, maxLength, meanLength;
      geom::computeEdgeLengthStatistics(mesh, &minLength, &maxLength,
                                        &meanLength);
      length = 0.99 * meanLength;
      std::cout << "\nWill split edges longer than " << length << ".\n";
   }

   // Print information about the manifold data structure.
   if (boundaryManifold != 0) {
      std::cout << "\n";
      boundaryManifold->printInformation(std::cout);
      std::cout << "\n";
   }

   std::cout << "Refining the mesh...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Refine the mesh.
   int count;
   if (meshFile.empty()) {
      count = geom::refine(&mesh, boundaryManifold,
                           ads::constructUnaryConstant<Vertex, double>(length));
   }
   else {
      // Read the interpolation mesh.
      InterpolationMesh interpolationMesh;
      {
         std::ifstream file(meshFile.c_str());
         geom::readAscii(file, &interpolationMesh);
      }
      // Read the interpolation field.
      std::vector<double> interpolationField;
      {
         std::ifstream file(fieldFile.c_str());
         file >> interpolationField;
      }
      // The interpolation functor.
      geom::ISS_Interpolate<InterpolationMesh>
      edgeLength(interpolationMesh, &interpolationField[0]);
      // Refine the mesh.
      count = geom::refine(&mesh, boundaryManifold, edgeLength);
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

   // If a boundary manifold was allocated.
   if (boundaryManifold) {
      // Free that memory.
      delete boundaryManifold;
      boundaryManifold = 0;
   }

   return 0;
}
