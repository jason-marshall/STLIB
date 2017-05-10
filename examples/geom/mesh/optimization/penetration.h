// -*- C++ -*-

/*!
  \file penetration.h
  \brief Report penetrations of a solid mesh.
*/

/*!
  \page examples_geom_mesh_penetration Penetrations

  \section examples_geom_mesh_penetration_introduction Introduction

  This program takes a simplicial mesh and a list of points and detects
  which points penetrate the mesh. For each of these points it reports the
  simplex that contains the point and the closest point on the surface of the
  mesh.

  \section examples_geom_mesh_penetration_usage Usage

  \verbatim
  penetrationN.exe mesh points penetrations\endverbatim
  - mesh is the file name for the input mesh.
  - points is the file name for the points.
  - penetrations is the file name for recording the penetrations.
  .
  The penetrations are reported in the format:
  [point index] [simplex index] [closest point on the boundary].

  \section examples_geom_mesh_penetration_methodology Methodology

  Reporting penetrations proceeds by first detecting the points that
  penetrate the mesh and then computing the closest points on the surface
  for the penetrating points. To accomplish the former we iterate over the
  simplices in the mesh and determine the points inside the simplex. To make
  this operation efficient, the points are stored in an
  \ref geom_orq "orthogonal range query"
  (ORQ) data structure. These data structures can report the points
  inside a bounding box (orthogonal range). There are many choices for ORQ
  data structures:
  \ref geom_orq_orq_kdtree "kd-trees",
  \ref geom_orq_orq_octree "quadtrees", etc.
  While tree data structures can
  robustly handle the greatest variety inputs,
  \ref geom_orq_orq_cell "bucketing strategies"
  offer better performance for most data sets.  We use the CellArray class,
  which is a dense array of buckets, with a judicious choice of bucket size.
  Another advantage of using dense cell arrays is that they can be
  constructed quickly; the complexity of constructing
  one is linear in the number of buckets and the number of points.

  To determine the points inside a simplex, we first perform an ORQ with the
  bounding box that contains it. We then determine which of the reported points
  are inside the simplex. For this one computes the signed distances to the
  supporting hyper-planes of the simplex faces. A point is inside if and
  only if all the distances are negative.

  After determining the penetrating points and the simplices that contain them,
  we compute the closest points on the mesh boundary. To accomplish this, we
  extract the boundary and store it in a bounding box tree (BBoxTree).
  A bounding box tree is a heirarchical data structure which is quite similar
  to a kd-tree. However instead of storing points, it stores the bounding
  boxes of objects, in this
  case the boundary faces of the mesh. The tree stores a heirarchy of bounding
  boxes and the indices of the objects. We use a lower/upper bound search to
  find the potential closest boundary faces. This search, as the name suggests,
  maintains lower and upper bounds on the distance as it traverses the
  tree toward the leaf nodes. The closest face is chosen from the candidates
  by computing the distance to each. Finally, we compute the closest point on
  that face.

  \section examples_geom_mesh_penetration_complexity Computational Complexity

  The computational complexity of performing orthogonal range queries is
  dependent upon the distribution of the points. Likewise the complexity of
  a lower/upper bound search in a bounding box tree depends upon the geometry
  of the object being searched. There are no methods that have good
  worst-case complexities. However, for reasonable inputs one can calculate
  the expected complexity.

  Let the D-dimensional mesh have \e M simplices and \e m boundary faces.
  Let there be \e N points, \e n of which penetrate the mesh. The expected
  computational complexity of reporting the penetrations is
  \f$\mathcal{O}(M + N + (m D + n) \log m)\f$. This is the sum of the following
  procedures:
  - \f$\mathcal{O}(N)\f$ for constructing the cell array.
  - \f$\mathcal{O}(M + N + n)\f$ for detecting the points which lie inside
  mesh simplices.
  - \f$\mathcal{O}(m D \log m)\f$ for constructing the bounding box tree.
  - \f$\mathcal{O}(n \log m)\f$ for \e n lower/upper bound searches.

  \section examples_geom_mesh_penetration_example 3-D Example

  There is an 3-D example in
  <tt>stlib/results/geom/mesh/3/penetration/</tt>. Here the solid mesh is
  a tesselation of the unit sphere and the test points are a lattice of
  one million points that lie in the plane <em>z = -0.99</em>.

  \image html SpherePlatePenetration.jpg "The solid mesh and the points."
  \image latex SpherePlatePenetration.jpg "The solid mesh and the points."

  \verbatim
  penetration3.exe mesh.txt points.txt penetration.txt\endverbatim

  There are 4068 penetrations. Detecting the penetrations takes 0.64 seconds.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/penetration.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <string>

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
         << programName << "\n"
         << " mesh points penetrations\n\n"
         << "- mesh is the file name for the input mesh.\n"
         << "- points is the file name for the points.\n"
         << "- penetrations is the file name for recording the penetrations.\n\n"
         << "The penetrations are reported in the format:\n"
         << "[point index] [simplex index] [closest point on the boundary]\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   static_assert(Dimension == 2 || Dimension == 3,
                 "The dimension must be 2 or 3.");

   // The simplicial mesh.
   typedef geom::IndSimpSetIncAdj<Dimension, Dimension> Mesh;
   typedef Mesh::Vertex Point;
   typedef std::tuple<std::size_t, std::size_t, Point> Record;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // If they did not specify the mesh, points, and output file.
   if (parser.getNumberOfArguments() != 3) {
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
   readAscii(parser.getArgument().c_str(), &mesh);
   std::cout << "The solid mesh has " << mesh.vertices.size()
             << " vertices and " << mesh.indexedSimplices.size()
             << " simplices.\n";

   // Read the points.
   std::vector<Point> points;
   {
      std::ifstream in(parser.getArgument().c_str());
      std::size_t size = 0;
      in >> size;
      points.resize(size);
      for (std::size_t i = 0; i != size; ++i) {
         in >> points[i];
      }
   }
   std::cout << "There are " << points.size() << " points.\n";

   std::vector<Record> penetrations;

   std::cout << "Reporting penetrations...\n";
   ads::Timer timer;
   timer.tic();

   geom::reportPenetrations(mesh, points.begin(), points.end(),
                            std::back_inserter(penetrations));

   double elapsedTime = timer.toc();
   std::cout << "done.\nReporting penetrations took " << elapsedTime
             << " seconds.\n"
             << "Number of penetrations = " << penetrations.size() << '\n';

   // Write the penetrations.
   {
      std::ofstream out(parser.getArgument().c_str());
      out << penetrations.size() << '\n';
      for (std::vector<Record>::const_iterator i = penetrations.begin();
            i != penetrations.end(); ++i) {
         out << std::get<0>(*i) << ' '
             << std::get<1>(*i) << ' '
             << std::get<2>(*i) << '\n';
      }
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
