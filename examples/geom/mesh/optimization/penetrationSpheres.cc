// -*- C++ -*-

/*!
  \file penetrationSpheres.cc
  \brief Push a solid mesh into an elastic surface.
*/

/*!
  \page examples_geom_mesh_penetrationCanister Penetration with an elastic surface

*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/penetration.h"
#include "stlib/geom/mesh/iss/laplacian.h"
#include "stlib/geom/mesh/simplicial/build.h"
#include "stlib/geom/mesh/simplicial/refine.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <string>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

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
         << programName << "\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   // The simplicial mesh.
   typedef geom::IndSimpSetIncAdj<3, 3> Mesh;
   typedef geom::IndSimpSetIncAdj<3, 2> Surface;
   typedef Mesh::Vertex Point;
   typedef std::tuple<std::size_t, std::size_t, Point> Record;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // If they did not specify the mesh, points, and output file.
   if (parser.getNumberOfArguments() != 0) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Check that there are no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Read the sphere.
   Mesh s1;
   readAscii("../../../../data/geom/mesh/33/sphereR1E0.3.txt", &s1);
   std::cout << "The sphere has " << s1.vertices.size()
             << " vertices and " << s1.indexedSimplices.size()
             << " simplices.\n";
   Mesh s2(s1);

   // Move the spheres.
   const Point offset = {{1, 1, 1}};
   for (Mesh::VertexIterator i = s1.vertices.begin();
        i != s1.vertices.end(); ++i) {
      *i += offset;
   }
   for (Mesh::VertexIterator i = s2.vertices.begin();
         i != s2.vertices.end(); ++i) {
      *i -= offset;
   }

   // Make the surface mesh.
   Surface surface;
   {
      const double vertexCoordinates[] = {0, 3, 3,
                                          0, -3, 3,
                                          0, -3, -3,
                                          0, 3, -3
      };
      const std::size_t numVertices = sizeof(vertexCoordinates) / 
         (sizeof(double) * 3);
      const std::size_t indexedSimplices[] = {0, 1, 2,
                                              2, 3, 0
      };
      const std::size_t numSimplices = sizeof(indexedSimplices) / 
         (sizeof(std::size_t) * 3);
      // The coarse mesh.
      geom::IndSimpSetIncAdj<3, 2> coarse;
      build(&coarse, numVertices, vertexCoordinates, numSimplices,
            indexedSimplices);

      // Refine so no edge is longer than 0.1.
      geom::SimpMeshRed<3, 2> fine(coarse);
      geom::refine(&fine, ads::constructUnaryConstant<Point, double>(0.1));
      geom::buildIndSimpSetFromSimpMeshRed(fine, &surface);
   }

   std::cout << "The surface mesh has " << surface.vertices.size()
             << " vertices and " << surface.indexedSimplices.size()
             << " simplices.\n";

   // Write the initial solid meshes.
   writeAscii("sphere1Initial.txt", s1);
   writeAscii("sphere2Initial.txt", s2);
   // Write the initial surface mesh.
   writeAscii("surfaceInitial.txt", surface);

   std::cout << "Calculating...\n";
   ads::Timer timer;
   timer.tic();

   std::vector<Record> penetrations;

   for (std::size_t n = 0; n != 100; ++n) {
      std::cout << n << ' ';
      std::cout.flush();
      // Push in the spheres.
      for (Mesh::VertexIterator i = s1.vertices.begin();
            i != s1.vertices.end(); ++i) {
         (*i)[0] -= 0.02;
      }
      for (Mesh::VertexIterator i = s2.vertices.begin();
            i != s2.vertices.end(); ++i) {
         (*i)[0] += 0.02;
      }
      for (std::size_t m = 0; m != 10; ++m) {
         // Laplacian smoothing.
         geom::applyLaplacian(&surface);
         // Determine the penetrations.
         penetrations.clear();
         geom::reportPenetrations(s1, surface.vertices.begin(),
                                  surface.vertices.end(),
                                  std::back_inserter(penetrations));
         geom::reportPenetrations(s2, surface.vertices.begin(),
                                  surface.vertices.end(),
                                  std::back_inserter(penetrations));
         // Remove the penetrations.
         for (std::vector<Record>::const_iterator i = penetrations.begin();
               i != penetrations.end(); ++i) {
            surface.vertices[std::get<0>(*i)] = std::get<2>(*i);
         }
      }
   }

   double elapsedTime = timer.toc();
   std::cout << "done.\n" << elapsedTime << " seconds.\n";

   // Write the solid meshes.
   writeAscii("sphere1Final.txt", s1);
   writeAscii("sphere2Final.txt", s2);

   // Write the final surface mesh.
   writeAscii("surfaceFinal.txt", surface);

   return 0;
}
