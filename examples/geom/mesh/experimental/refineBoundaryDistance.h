// -*- C++ -*-

/*!
  \file refineBoundaryDistance.h
  \brief Refine according to the distance from the boundary.
*/

/*!
  \page examples_geom_mesh_refineBoundaryDistance Refine according to the distance from the boundary.

  This executable refines the elements of mesh using a linear function of
  the distance to the boundary as the maximum allow edge length.

  Usage:
  \verbatim
  refineBoundaryDistance22.exe -a=a -b=b in out
  \endverbatim
  - The maximim allowed edge length is \f$a * d + b\f$ where \f$d\f$ is the
  distance to the boundary.
  - in is the input mesh.
  - out is the output mesh.

  \section examples_geom_mesh_refineBoundaryDistance_square 2-D Example: Square



  We start with a mesh of the unit square.
  This mesh has 4 vertices and 2 triangles.

  \verbatim
  cp ../../../data/geom/mesh/22/square1.txt mesh.txt
  \endverbatim

  \image html rbd22_square.jpg "The coarse mesh."
  \image latex rbd22_square.pdf "The coarse mesh."



  We refine the mesh so that the maximum edge length function has a value
  of 0.01 at the boundary and there is a gradual transition from fine
  to coarse.

  \verbatim
  refineBoundaryDistance22.exe -a=0.5 -b=0.01 mesh.txt mesh_refined.txt
  \endverbatim

  The resulting mesh has 2265 vertices and 4016 triangles.

  \image html rbd22_square-0.5.jpg "The refined mesh.  Gradual transition from fine to coarse."
  \image latex rbd22_square-0.5.pdf "The refined mesh.  Gradual transition from fine to coarse."



  If we increase the slope of the maximum edge length function, we will
  get a faster transition from fine to coarse cells.

  \verbatim
  refineBoundaryDistance22.exe -a=1 -b=0.01 mesh.txt mesh_refined.txt
  \endverbatim

  The resulting mesh has 1385 vertices and 2256 triangles.

  \image html rbd22_square-1.jpg "Faster transition from fine to coarse."
  \image latex rbd22_square-1.pdf "Faster transition from fine to coarse."



  If we increase the slope to 2, we will
  get a rapid transition from fine to coarse cells.

  \verbatim
  refineBoundaryDistance22.exe -a=2 -b=0.01 mesh.txt mesh_refined.txt
  \endverbatim

  This mesh has 1221 vertices and 1928 triangles.

  \image html rbd22_square-2.jpg "Rapid transition from fine to coarse."
  \image latex rbd22_square-2.pdf "Rapid transition from fine to coarse."



  \section examples_geom_mesh_refineBoundaryDistance_a 2-D Example: A



  We start with a mesh of the letter "A" that has unit height and width.
  This mesh has 23 vertices and 23 triangles.

  \verbatim
  cp ../../../data/geom/mesh/22/a_23.txt mesh.txt
  \endverbatim

  \image html rbd22_a.jpg "The coarse mesh."
  \image latex rbd22_a.pdf "The coarse mesh."



  We refine the mesh with the same maximum edge length function as
  for the square.

  \image html rbd22_a-0.5.jpg "Gradual transition from fine to coarse."
  \image latex rbd22_a-0.5.pdf "Gradual transition from fine to coarse."

  \image html rbd22_a-1.jpg "Faster transition from fine to coarse."
  \image latex rbd22_a-1.pdf "Faster transition from fine to coarse."

  \image html rbd22_a-2.jpg "Rapid transition from fine to coarse."
  \image latex rbd22_a-2.pdf "Rapid transition from fine to coarse."



  \section examples_geom_mesh_refineBoundaryDistance_code 3-D Example: Cone


  We start with a surface mesh of the top of a cone.

  \verbatim
  cp ../../../data/geom/mesh/32/cone_top.txt mesh.txt
  \endverbatim

  \image html rbd32_cone_top.jpg "The coarse mesh."
  \image latex rbd32_cone_top.pdf "The coarse mesh."



  We refine the mesh so that the maximum edge length function has a value
  of 0.01 at the boundary and there is a transition from fine
  to coarse as one moves away from the boundary.

  \verbatim
  refineBoundaryDistance32.exe -a=1 -b=0.01 mesh.txt mesh_refined.txt
  \endverbatim

  \image html rbd32_cone_top_refined.jpg "The refined mesh."
  \image latex rbd32_cone_top_refined.pdf "The refined mesh."




  \section examples_geom_mesh_refineBoundaryDistance_code 3-D Example: Tetrahedron


  We start with a solid mesh of a tetrahedron.

  \verbatim
  cp ../../../data/geom/mesh/33/tetrahedron.txt mesh.txt
  \endverbatim

  \image html rbd33_tet.jpg "The coarse mesh."
  \image latex rbd33_tet.pdf "The coarse mesh."

  We refine the mesh so that the maximum edge length function has a value
  of 0.1 at the boundary and there is a transition from fine
  to coarse as one moves away from the boundary.

  \verbatim
  refineBoundaryDistance33.exe -a=1 -b=0.1 mesh.txt mesh_refined.txt
  \endverbatim

  \image html rbd33_tet_refined.jpg "The refined mesh."
  \image latex rbd33_tet_refined.pdf "The refined mesh."
*/

#ifndef SPACE_DIMENSION
#error SPACE_DIMENSION must be defined to compile this program.
#endif
#ifndef SIMPLEX_DIMENSION
#error SIMPLEX_DIMENSION must be defined to compile this program.
#endif

#include "../smr_io.h"

#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/simplicial/build.h"
#include "stlib/geom/mesh/simplicial/quality.h"
#include "stlib/geom/mesh/simplicial/refine.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/functor/linear.h"

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
         << programName << " [-a=a] [-b=b] in out\n"
         << "The maximim allowed edge length is a * d + b where d is the distance\n"
         << "to the boundary.\n"
         << "in is the input mesh.\n"
         << "out is the output mesh.\n";
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
   typedef geom::IndSimpSetIncAdj<SPACE_DIMENSION, SIMPLEX_DIMENSION> ISS;
   typedef geom::IndSimpSet < SPACE_DIMENSION, SIMPLEX_DIMENSION - 1 > Boundary;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   // The maximum edge length.
   double a = 0, b = 0;
   parser.getOption("a", &a);
   parser.getOption("b", &b);
   if (a <= 0 || b <= 0) {
      std::cerr << "Bad coefficient values.\n";
      exitOnError();
   }

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

   // Get the boundary of the mesh.
   Boundary boundary;
   {
      ISS iss;
      geom::buildIndSimpSetFromSimpMeshRed(mesh, &iss);
      geom::buildBoundary(iss, &boundary);
   }


   // The functor for the distance to the boundary.
   geom::ISS_SimplexQuery<Boundary> simplexQuery(boundary);
   geom::ISS_Distance<Boundary> distance(simplexQuery);
   // The linear function.
   ads::UnaryLinear<double> linear(a, b);
   // The linear function of the distance.
   ads::unary_compose_unary_unary < ads::UnaryLinear<double>,
       geom::ISS_Distance<Boundary> > linearDistance(linear, distance);

   // Print quality measures for the input mesh.
   std::cout << "Quality of the input mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);
   std::cout << "Number of edges = " << mesh.computeEdgesSize() << "\n";
   geom::printEdgeLengthStatistics(std::cout, mesh);

   std::cout << "Refining the mesh...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Refine the mesh.
   int count = geom::refine(&mesh, linearDistance);

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

   return 0;
}
