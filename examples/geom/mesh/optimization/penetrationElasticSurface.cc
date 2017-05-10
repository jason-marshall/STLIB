// -*- C++ -*-

/*!
  \file penetrationElasticSurface.cc
  \brief Push a solid mesh into an elastic surface.
*/

/*!
  \page examples_geom_mesh_penetrationElasticSurface Penetration with an elastic surface

*/

//#include "stlib/geom/mesh/simplicial/file_io.h"
#include "../iss_io.h"
//#include "../smr_io.h"

#include "stlib/geom/mesh/iss/penetration.h"
#include "stlib/geom/mesh/iss/laplacian.h"
#include "stlib/geom/mesh/simplicial/build.h"
#include "stlib/geom/mesh/simplicial/refine.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <string>

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
   //typedef geom::SimpMeshRed<3,2> Surface;
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

   // Read the input solid mesh.
   Mesh mesh;
   readAscii("../../../../data/geom/mesh/33/enterpriseL50.txt", &mesh);
   std::cout << "The solid mesh has " << mesh.vertices.size()
             << " vertices and " << mesh.indexedSimplices.size()
             << " simplices.\n";

   // Make the surface mesh.
   Surface surface;
   {
      const double x0 = -500;
      const double x1 = 500;
      const double y = -400;
      const double z0 = -500;
      const double z1 = 180;
      const double vertexCoordinates[] = {x0, y, z0,
                                          x1, y, z0,
                                          x1, y, z1,
                                          x0, y, z1
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

      // Refine so no edge is longer than 4.
      geom::SimpMeshRed<3, 2> fine(coarse);
      geom::refine(&fine, ads::constructUnaryConstant<Point, double>(4));
      geom::buildIndSimpSetFromSimpMeshRed(fine, &surface);
   }

   std::cout << "The surface mesh has " << surface.vertices.size()
             << " vertices and " << surface.indexedSimplices.size()
             << " simplices.\n";

   // Write the initial surface mesh.
   writeAscii("surfaceInitial.txt", surface);

   std::cout << "Calculating...\n";
   ads::Timer timer;
   timer.tic();

   std::vector<Record> penetrations;
   //std::vector<Point> vertices;

   for (std::size_t n = 0; n != 500; ++n) {
      std::cout << n << ' ';
      std::cout.flush();
      // Push in the -y direction.
      for (Mesh::VertexIterator i = mesh.vertices.begin();
            i != mesh.vertices.end(); ++i) {
         (*i)[1] -= 1;
      }
      for (std::size_t m = 0; m != 10; ++m) {
         // Laplacian smoothing.
         geom::applyLaplacian(&surface);
#if 0
         // Record vertices.
         vertices.clear();
         for (Surface::VertexIterator i = surface.vertices.begin();
               i != surface.vertices.end(); ++i) {
            vertices.push_back(*i);
         }
#endif
         // Determine the penetrations.
         penetrations.clear();
         geom::reportPenetrations(mesh, surface.vertices.begin(),
                                  surface.vertices.end(),
                                  std::back_inserter(penetrations));
         // Remove the penetrations.
         for (std::vector<Record>::const_iterator i = penetrations.begin();
               i != penetrations.end(); ++i) {
            surface.vertices[std::get<0>(*i)] = std::get<2>(*i);
         }
#if 0
         for (std::vector<Record>::const_iterator i = penetrations.begin();
               i != penetrations.end(); ++i) {
            vertices[std::get<0>(*i)] = std::get<2>(*i);
         }
         surface.setVertices(vertices.begin(), vertices.end());
#endif
      }
   }

   double elapsedTime = timer.toc();
   std::cout << "done.\n" << elapsedTime << " seconds.\n";

   // Write the solid mesh.
   writeAscii("solidFinal.txt", mesh);

   // Write the final surface mesh.
   writeAscii("surfaceFinal.txt", surface);

   return 0;
}
