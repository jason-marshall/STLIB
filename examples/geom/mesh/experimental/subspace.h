// -*- C++ -*-

/*!
  \file subspace.h
  \brief Subspace method.
*/

/*!
  \page examples_geom_mesh_subspace The subspace method.

  \section examples_geom_mesh_subspace_algorithm The Algorithm

  In the process of mesh optimization, one searches in the space of all meshes
  for a mesh of high quality.  The space of all meshes is very large.
  First there is a size complexity: the number of nodes.
  Next is a combinatorial complexity: the nodes are connected to form
  simplices.  Finally, there is the geometric complexity: the positions
  of each of the nodes.

  It is the complexity of the search space that makes mesh
  generation/optimization so difficult.  The subspace method tries to
  make mesh optimization more tractible by searching in only a subspace
  of all possible meshes.

  One could use many criteria to select a subspace.  For this example,
  we use incidence optimization and Laplacian smoothing.  For the incidence
  optimization we specify that each internal node should ideally have
  six incident cells.  Boundary nodes should have one incident cell per
  each \f$\pi / 3\f$ of internal angle.  We use edge flips to minimize the
  deviation from the ideal.  The laplacian smoothing determines the positions
  of the nodes.  The position of each internal node is the arithmetic mean
  of its neighbors' positions.  Note that the incidence optimization does
  not depend on the positions of the internal nodes.  Thus the incidence
  optimization and the Laplacian smoothing steps are not coupled.  They
  may be applied sequentially, without iteration.

  Alternatively, one could use Delaunay triangulation with Laplacian smoothing,
  Delaunay triangulation with geometric optimization of the condition number,
  or any combination that can be used to compute the topology and geometry
  of the mesh in the selected subspace.  However, in most cases the
  topological criterion and the geometric criterion will be coupled.
  In this case, one would have to iterate back and forth between the
  topological criterion and the geometric criterion to converge to a
  state that satisfies both.

  With the two criteria controlling the topology and geometry of the mesh,
  we use edge splitting and edge collapsing to obtain the desired edge
  lengths and the desired element shapes.  (Currently splitting and collapsing
  are only used to obtain the desired edge lengths.  I will be investigating
  element shapes.)  After splitting or collapsing and edge, the criteria
  are applied to return to the selected subspace.  The criteria need only
  be applied locally.



  \section examples_geom_mesh_subspace_sin Example: A Distorted Square

  We start with a mesh of the unit square.

  \verbatim
  cp ../../../data/geom/mesh/22/square4.txt .
  \endverbatim

  \image html core_square4.jpg "A mesh of the unit square."
  \image latex core_square4.pdf "A mesh of the unit square."

  We distort the square by mapping the vertices with
  \f$ (x,y) \rightarrow (0.4 x + 0.2 \sin(2 \pi y), y)\f$.
  (Here \c sin_function is the file that contains this function.)

  \verbatim
  python.exe ../../../data/geom/mesh/utilities/vertices_map.py square4.txt sin.txt sin_function
  \endverbatim

  \image html core_sin.jpg "A mesh of the distorted square."
  \image latex core_sin.pdf "A mesh of the distorted square."

  We generate a coarse mesh of the object.  We specify lower and upper bounds
  on the edge length of 0.1 and 0.25, respectively.
  With the -a option, we apply the constraint that boundary nodes
  should not be moved if the have angle sharper than
  \f$ 3 \pi / 4 \approx 2.36 \f$.  The mesh is generated with two sweeps over
  the cells.

  \verbatim
  subspace2.exe -lower=0.1 -upper=0.25 -angle=2.36 -sweeps=2 sin.txt sin-c.txt
  \endverbatim

  \image html core_sin-c.jpg "A coarse mesh of the distorted square."
  \image latex core_sin-c.pdf "A coarse mesh of the distorted square."

  Next we generate a finer mesh.  We halve the edge lengths.  First we show
  the effect of a single sweep over the cells.

  \verbatim
  subspace2.exe -lower=0.05 -upper=0.125 -angle=2.36 -sweeps=1 sin.txt sin-f1.txt
  \endverbatim

  \image html core_sin-f1.jpg "A finer mesh generated with 1 sweep."
  \image latex core_sin-f1.pdf "A finer mesh generated with 1 sweep."

  Five sweeps is sufficient to bring the mesh to its final form.
  (Subsequent sweeps will have no effect.)

  \verbatim
  subspace2.exe -lower=0.05 -upper=0.125 -angle=2.36 -sweeps=5 sin.txt sin-f.txt
  \endverbatim

  \image html core_sin-f.jpg "A finer mesh of the distorted square."
  \image latex core_sin-f.pdf "A finer mesh of the distorted square."







  \section examples_geom_mesh_subspace_maze Example: A 2-D Maze

  We start with the boundary of a maze.  (Well, not much of a maze,
  but that's what we'll call it.)

  \verbatim
  cp ../../../data/geom/mesh/21/maze.txt maze_boundary.txt
  \endverbatim

  \image html core_maze_boundary.jpg "The boundary of a maze."
  \image latex core_maze_boundary.pdf "The boundary of a maze."

  We generate a mesh of very poor quality by connecting the boundary edges to
  a center point.  About half the cells have negative area.

  \verbatim
  centerPointMesh2.exe maze_boundary.txt maze.txt
  \endverbatim

  \image html core_maze.jpg "The initial mesh of the maze."
  \image latex core_maze.pdf "The initial mesh of the maze."

  We perform 1 sweep of the coarsening/refining algorithm.  Since the width
  of the path in the maze is 0.1, we try to get edge lengths between 0.05
  and 0.1.

  \verbatim
  subspace2.exe -lower=0.05 -upper=0.1 -angle=2.36 -sweeps=1 maze.txt maze-1.txt
  \endverbatim

  \image html core_maze-1.jpg "The mesh after 1 sweep."
  \image latex core_maze-1.pdf "The mesh after 1 sweep."


  Since this is a complicated geometry, it will take a few sweeps over the
  cells to get the mesh roughly sorted out.

  \verbatim
  subspace2.exe -lower=0.05 -upper=0.1 -angle=2.36 -sweeps=1 maze-1.txt maze-2.txt
  \endverbatim

  \image html core_maze-2.jpg "The mesh after 2 sweeps."
  \image latex core_maze-2.pdf "The mesh after 2 sweeps."


  \verbatim
  subspace2.exe -lower=0.05 -upper=0.1 -angle=2.36 -sweeps=1 maze-2.txt maze-3.txt
  \endverbatim

  \image html core_maze-3.jpg "The mesh after 3 sweeps."
  \image latex core_maze-3.pdf "The mesh after 3 sweeps."


  \verbatim
  subspace2.exe -lower=0.05 -upper=0.1 -angle=2.36 -sweeps=1 maze-3.txt maze-4.txt
  \endverbatim

  \image html core_maze-4.jpg "The mesh after 4 sweeps."
  \image latex core_maze-4.pdf "The mesh after 4 sweeps."

  Five sweeps is sufficient to obtain a mesh that fills the correct region
  with reasonable triangles.

  \verbatim
  subspace2.exe -lower=0.05 -upper=0.1 -angle=2.36 -sweeps=1 maze-4.txt maze-5.txt
  \endverbatim

  \image html core_maze-5.jpg "The mesh after 5 sweeps."
  \image latex core_maze-5.pdf "The mesh after 5 sweeps."


  Eight sweeps is enough to obtain a quality mesh.  The minimum condition
  number is 0.79; the mean is 0.92.

  \verbatim
  subspace2.exe -lower=0.05 -upper=0.1 -angle=2.36 -sweeps=3 maze-5.txt maze-8.txt
  \endverbatim

  \image html core_maze-8.jpg "The mesh after 8 sweeps."
  \image latex core_maze-8.pdf "The mesh after 8 sweeps."



  We could also generate a coarse mesh.
  For the mesh below, the minimum condition number is 0.72; the mean is 0.86.

  \verbatim
  subspace2.exe -lower=0.1 -upper=0.25 -angle=2.36 -sweeps=6 maze.txt maze-coarse.txt
  \endverbatim

  \image html core_maze-coarse.jpg "Six sweeps to generate a coarse mesh."
  \image latex core_maze-coarse.pdf "Six sweeps to generate a coarse mesh."


  Finally, we refine this coarse mesh.
  For the fine mesh, the minimum condition number is 0.48; the mean is 0.89.

  \verbatim
  subspace2.exe -lower=0.02 -upper=0.05 -angle=2.36 -sweeps=10 maze-coarse.txt maze-fine.txt
  \endverbatim

  \image html core_maze-fine.jpg "A fine mesh."
  \image latex core_maze-fine.pdf "A fine mesh."
*/

#ifndef SPACE_DIMENSION
#error SPACE_DIMENSION must be defined to compile this program.
#endif
#ifndef SIMPLEX_DIMENSION
#error SIMPLEX_DIMENSION must be defined to compile this program.
#endif

#include "../smr_io.h"
#include "../iss_io.h"

#include "stlib/geom/mesh/simplicial/refine.h"
#include "stlib/geom/mesh/simplicial/quality.h"
#include "stlib/geom/mesh/simplicial/laplacian.h"
#include "stlib/geom/mesh/iss/build.h"
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
         << programName << " [-boundary] [-lower=minimumLength] [-upper=maximumLength] [-angle=minimumAngle] [sweeps=n] [-norm=m] in out\n"
         << "-boundary and -angle are only supported for 2-2 meshes.\n"
         << "The norm should by 0, 1, or 2 for max norm, 1-norm, or 2-norm.\n"
         << "in is the input mesh.\n"
         << "out is the output mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   static_assert(SPACE_DIMENSION == 2 || SPACE_DIMENSION == 3,
                 "The space dimension must be 2 or 3.");
   static_assert(SIMPLEX_DIMENSION == 2, "The simplex dimension must be 2.");

   typedef geom::SimpMeshRed<SPACE_DIMENSION, SIMPLEX_DIMENSION> Mesh;

#if SPACE_DIMENSION == 2
   typedef geom::IndSimpSetIncAdj<SPACE_DIMENSION, SIMPLEX_DIMENSION> ISS;
   typedef geom::IndSimpSet < SPACE_DIMENSION, SIMPLEX_DIMENSION - 1 > Boundary;
   // The functor for computing signed distance.
   typedef geom::ISS_SignedDistance<Boundary> ISS_SD;
   // The functor for computing the closest point.
   // CONTINUE: Add a choice between these two.
   //typedef geom::ISS_SD_ClosestPoint<Boundary> ISS_SD_CP;
   typedef geom::ISS_SD_ClosestPointDirection<Boundary> ISS_SD_CP;
#else
   typedef geom::IndSimpSet<SPACE_DIMENSION, SIMPLEX_DIMENSION> ISS;
   // The functor for computing signed distance.
   typedef geom::ISS_SignedDistance<ISS> ISS_SD;
   // The functor for computing the closest point.
   // CONTINUE: Add a choice between these two.
   //typedef geom::ISS_SD_ClosestPoint<ISS> ISS_SD_CP;
   typedef geom::ISS_SD_ClosestPointDirection<ISS> ISS_SD_CP;
#endif

   typedef Mesh::Vertex Vertex;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   // The minimum boundary angle for moving a node.
   double minimumAngle = 0;
   parser.getOption("angle", &minimumAngle);

   // The minimum and maximum edge length.
   double minimumLength = 0, maximumLength = 0;
   parser.getOption("lower", &minimumLength);
   parser.getOption("upper", &maximumLength);
   if (minimumLength == 0 || maximumLength == 0) {
      std::cerr << "Sorry, you must specify the lower and upper bounds.\n";
      exitOnError();
   }

   // By default one sweep is performed.
   int numberOfSweeps = 1;
   parser.getOption("sweeps", &numberOfSweeps);

   int norm = 0;
   if (parser.getOption("norm", &norm)) {
      if (norm < 0 || norm > 2) {
         std::cerr << "Bad value for the norm.\n";
         exitOnError();
      }
   }

#if SPACE_DIMENSION == 2
   // The boundary.
   Boundary boundary;
   std::string boundaryFileName;
   if (parser.getOption("boundary", &boundaryFileName)) {
      // Read the boundary from the file.
      readAscii(boundaryFileName.c_str(), &boundary);
   }
#endif

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the input and output mesh.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

#if SPACE_DIMENSION == 2
   // If they did not specify a boundary file, use the boundary of the mesh.
   if (boundary.vertices.size() == 0) {
      // Make an indexed simplex set.
      ISS iss;
      buildIndSimpSetFromSimpMeshRed(mesh, &iss);
      // Make the boundary mesh.
      geom::buildBoundary(iss, &boundary);
   }
   // The data structure and functor that computes the signed distance and
   // closest point.
   ISS_SD signedDistance(boundary);
#else
   // Make an indexed simplex set.
   ISS iss;
   buildIndSimpSetFromSimpMeshRed(mesh, &iss);
   // The data structure and functor that computes the signed distance and
   // closest point.
   ISS_SD signedDistance(iss);
#endif
   // The functor that returns the closest point.
   ISS_SD_CP closestPoint(signedDistance);

   // Print quality measures for the input mesh.
   std::cout << "Quality of the input mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);
   std::cout << "Number of edges = " << mesh.computeEdgesSize() << "\n";
   geom::printEdgeLengthStatistics(std::cout, mesh);

   std::cout << "Applying the subspace method to the mesh...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Coarsen and refine the mesh.
#if SPACE_DIMENSION == 2
   std::pair<std::size_t, std::size_t> counts = geom::optimizeWithSubspaceMethod
         (&mesh, closestPoint, minimumAngle,
          ads::constructUnaryConstant<Vertex, double>(minimumLength),
          ads::constructUnaryConstant<Vertex, double>(maximumLength),
          norm, numberOfSweeps);
#else
   std::pair<std::size_t, std::size_t> counts = geom::optimizeWithSubspaceMethod
         (&mesh, closestPoint,
          ads::constructUnaryConstant<Vertex, double>(minimumLength),
          ads::constructUnaryConstant<Vertex, double>(maximumLength),
          norm, numberOfSweeps);
#endif

   double elapsedTime = timer.toc();
   std::cout << "done.\n"
             << "Number of collapses = " << counts.first << "\n"
             << "Number of splits = " << counts.second << "\n"
             << "Subspace method took " << elapsedTime
             << " seconds.\n";

   // Print quality measures for the refined mesh.
   std::cout << "\nQuality of the output mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);
   std::cout << "Number of edges = " << mesh.computeEdgesSize() << "\n";
   geom::printEdgeLengthStatistics(std::cout, mesh);

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
