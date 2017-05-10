// -*- C++ -*-

/*!
  \file subdivideSphericalTriangles.cc
  \brief Subdivide a mesh of spherical triangles.
*/

/*!
  \page mst_subdivideSphericalTriangles Subdivide a mesh of spherical triangles.

  CONTINUE.
*/

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include "stlib/geom/mesh/iss/file_io.h"
#include "stlib/geom/mesh/iss/quality.h"
#include "stlib/geom/mesh/iss/subdivide.h"

#include <fstream>

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_VECTOR_IO_OPERATORS;
using namespace stlib;

namespace {

//
// Global variables.
//

//! The program name.
std::string programName;

//
// Local functions.
//

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " inputMesh inputCenters outputMesh [outputCenters]\n"
         << "- inputMesh is the input mesh.\n"
         << "- inputCenters are the sphere centers.\n"
         << "- outputMesh is the subdivided mesh.\n"
         << "- outputCenters are the sphere centers for the subdivided mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSetIncAdj<3, 2> Mesh;
   typedef Mesh::Vertex Point;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // There should be no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the input mesh, the sphere centers, and
   // the output mesh.
   if (!(parser.getNumberOfArguments() == 3 ||
         parser.getNumberOfArguments() == 4)) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input mesh.
   Mesh mesh;
   {
      std::ifstream in(parser.getArgument().c_str());
      geom::readAscii(in, &mesh);
   }

   // Read the sphere centers.
   std::vector<Point> centers;
   {
      std::ifstream in(parser.getArgument().c_str());
      in >> centers;
   }

   if (centers.size() != mesh.indexedSimplices.size()) {
      std::cerr << "Error: The mesh has " << mesh.indexedSimplices.size()
                << " triangles, but there are " << centers.size()
                << " sphere centers.\n";
      exitOnError();
   }

   // Print quality measures for the input mesh.
   std::cout << "\nQuality of the input mesh:\n\n";
   geom::printQualityStatistics(std::cout, mesh);

   std::cout << "Subdividing the mesh...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Subdivide the mesh.
   Mesh subdividedMesh;
   geom::subdivide(mesh, &subdividedMesh);
   assert(mesh.indexedSimplices.size() * 4 == 
          subdividedMesh.indexedSimplices.size());

   // Move the vertices of the subdivided mesh onto the spheres.
   // Loop over the simplices of the initial mesh.
   double radius;
   std::size_t simplexIndex, vertexIndex;
   Point p;
   for (std::size_t n = 0; n != mesh.indexedSimplices.size(); ++n) {
     radius = stlib::ext::euclideanDistance(mesh.getSimplexVertex(n, 0),
                                            centers[n]);
      // The four subdivided triangles.
      for (std::size_t m = 0; m != 4; ++m) {
         simplexIndex = 4 * n + m;
         // The three vertices of the triangle.
         for (std::size_t i = 0; i != 3; ++i) {
            vertexIndex = subdividedMesh.indexedSimplices[simplexIndex][i];
            p = subdividedMesh.vertices[vertexIndex] - centers[n];
            stlib::ext::normalize(&p);
            p *= radius;
            p += centers[n];
            subdividedMesh.vertices[vertexIndex] = p;
         }
      }
   }

   double elapsedTime = timer.toc();
   std::cout << "done.\n"
             << "Subdivision took " << elapsedTime << " seconds.\n";

   // Print quality measures for the subdivided mesh.
   std::cout << "\nQuality of the subdivided mesh:\n";
   geom::printQualityStatistics(std::cout, subdividedMesh);

   // Write the output mesh.
   {
      std::ofstream out(parser.getArgument().c_str());
      geom::writeAscii(out, subdividedMesh);
   }

   // If they specified the output centers file.
   if (! parser.areArgumentsEmpty()) {
      std::ofstream out(parser.getArgument().c_str());
      // Each input triangle is split into 4 triangles.
      out << 4 * centers.size() << "\n";
      for (std::size_t n = 0; n != centers.size(); ++n) {
         // Each of these four triangles have the same sphere center.
         for (std::size_t i = 0; i != 4; ++i) {
            out << centers[n] << "\n";
         }
      }
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
