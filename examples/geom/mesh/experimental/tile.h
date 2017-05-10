// -*- C++ -*-

/*!
  \file tile.h
  \brief Tile a faceted curve or surface.
*/

/*!
  \page examples_geom_mesh_tile Tile a faceted curve or surface.



  \section examples_geom_mesh_tile_example2 2-D Example

  We start with a boundary that forms the letter 'A'.

  \verbatim
  cp ../../../data/geom/mesh/21/a.txt boundary.txt
  \endverbatim

  \image html tile_2_boundary.jpg "The boundary to mesh."
  \image latex tile_2_boundary.pdf "The boundary to mesh."

  The boundary mesh lies within the unit square [0..1]x[0..1].  We tile
  the region with triangles of length 0.04.  The centroids of the triangles
  in the resulting mesh lie inside the boundary mesh.

  \verbatim
  tile2.exe -length=0.04 boundary.txt mesh.txt
  \endverbatim

  \image html tile_2_mesh.jpg "A tiling of the object."
  \image latex tile_2_mesh.pdf "A tiling of the object."

  The resulting mesh is pretty jagged.  It has many boundary triangles with
  two boundary faces.  It we moved the boundary vertices to lie on the boundary
  mesh, most of these triangles would get flattened.  If we remove the
  triangles with low adjacencies we obtain a mesh with a smoother boundary.

  \verbatim
  tile2.exe -smooth -length=0.04 boundary.txt mesh_s.txt
  \endverbatim

  \image html tile_2_mesh_s.jpg "Tiling with a smoothed boundary."
  \image latex tile_2_mesh_s.pdf "Tiling with a smoothed boundary."

  Finally we move the boundary vertices of the solid mesh to lie on the
  boundary mesh.

  \verbatim
  tile2.exe -smooth -move -length=0.04 boundary.txt mesh_sm.txt
  \endverbatim

  \image html tile_2_mesh_sm.jpg "The meshed object."
  \image latex tile_2_mesh_sm.pdf "The meshed object."

  From here, we could optimize the mesh to improve the quality of the
  triangles and the fit to the boundary curve.
*/

#ifndef DIMENSION
#error DIMENSION must be defined to compile this program.
#endif

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/tile.h"
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
         << programName << " [-closestPoint] [-smooth] [-move] [-length=l] boundary mesh\n"
         << "- The closestPoint option specifies that the closest point (instead of the\n"
         << "  closest point in the normal direction) should be used\n"
         << "  for moving the boundary nodes.\n"
         << "-smooth specifies that the boundary should be smoothed.\n"
         << "   Simplices with low adjacencies will be removed.\n"
         << "-move indicates that the boundary nodes of the solid mesh will\n"
         << "   be moved to lie on the boundary mesh.\n"
         << "-length is used to specify the simplex edge length.\n"
         << "boundary is the file name for the boundary mesh.\n"
         << "mesh is the file name for the solid mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   static_assert(DIMENSION == 2 || DIMENSION == 3,
                 "The dimension must be 2 or 3.");
   typedef geom::IndSimpSetIncAdj<DIMENSION, DIMENSION> Mesh;
   typedef geom::IndSimpSet < DIMENSION, DIMENSION - 1 > Boundary;
   // The functor for computing signed distance.
   typedef geom::ISS_SignedDistance<Boundary> ISS_SD;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   // Use the closest point (instead of the closest point along the normal
   // direction) for the boundary condition.
   bool areUsingClosestPoint = parser.getOption("closestPoint");

   // Smooth the boundary of the mesh by removing low adacency simplices.
   bool smooth = parser.getOption("smooth");

   // Move the mesh boundary points to the curve/surface.
   bool move = parser.getOption("move");

   // The maximum edge length of the simplices.
   double length = 0;
   parser.getOption("length", &length);

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the input boundary and the output mesh.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input boundary mesh.
   Boundary boundary;
   readAscii(parser.getArgument().c_str(), &boundary);
   std::cout << "The boundary mesh has " << boundary.vertices.size()
             << " vertices and " << boundary.indexedSimplices.size()
             << " simplices.\n";


   // Make a bounding box around the boundary.
   geom::BBox<double, DIMENSION> domain =
     geom::specificBBox<geom::BBox<double, DIMENSION> >
     (boundary.vertices.begin(), boundary.vertices.end());
   std::cout << "The bounding box for the boundary is\n" << domain << "\n";

   // The data structure and functor that computes the signed distance.
   ISS_SD signedDistance(boundary);

   // If they did not specify the edge length, try to choose an appropriate one.
   if (length == 0) {
      // Fill the bounding box with about 1000 simplices.
      if (DIMENSION == 2) {
         // The area of the unit triangle is sqrt(3) / 4.
         length = std::sqrt(content(domain) / 1000. * 4. / std::sqrt(3.));
      }
      else if (DIMENSION == 3) {
         // The area of the unit tetrahedron is 1 / 12.
         length = std::pow(content(domain) / 1000. * 12., 1. / 3.);
      }
      else {
         assert(false);
      }
   }
   if (length <= 0) {
      std::cerr << "Bad edge length.\n";
      exitOnError();
   }

   // Make the mesh.
   Mesh mesh;

   std::cout << "Building the mesh...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Tile the object.
   geom::tile(domain, length, signedDistance, &mesh);

   double elapsedTime = timer.toc();
   std::cout << "done.\nBuild took " << elapsedTime
             << " seconds.\n";

   // Smooth the boundary of the mesh.
   if (smooth) {
      geom::removeLowAdjacencies(&mesh, DIMENSION);
   }

   // Move the mesh boundary points to the curve/surface.
   if (move) {
      std::vector<std::size_t> bs;
      geom::determineBoundaryVertices(mesh, std::back_inserter(bs));
      if (areUsingClosestPoint) {
         // The functor for computing the closest point.
         geom::ISS_SD_ClosestPoint<Boundary> cp(signedDistance);
         geom::transform(&mesh, bs.begin(), bs.end(), cp);
      }
      else {
         // The functor for computing the closest point in the normal direction.
         geom::ISS_SD_ClosestPointDirection<Boundary> cp(signedDistance);
         geom::transform(&mesh, bs.begin(), bs.end(), cp);
      }
   }

   // Print quality measures for the output mesh.
   geom::printQualityStatistics(std::cout, mesh);

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
