// -*- C++ -*-

/*!
  \file pinballContact.cc
  \brief Report contacts with the pinball method.
*/

/*!
  \page examples_geom_mesh_pinball Pinball Contact

  \section examples_geom_mesh_pinball_introduction Introduction

  This program takes a simplicial mesh and detects contact using the pinball
  method.

  \section examples_geom_mesh_pinball_usage Usage

  \verbatim
  pinballContactN.exe mesh contacts\endverbatim
  - mesh is the file name for the input mesh.
  - contact is the file name for recording the contacts.
  .
  The contacts are reported in the format:
  \verbatim
  numberOfContacts
  identifier0 identifier1 penetrationX penetrationY penetrationZ
  ...\endverbatim

  The identifiers are the
  simplex indices. The penetration is a vector whose length is the
  penetration distance and whose direction is from the centroid of the
  latter simplex toward the centroid of the former simplex.

  \section examples_geom_mesh_pinball_methodology Methodology

  Contact detection is performed with a derivative of the pinball algorithm
  presented in \ref BelytschkoNeal "[1]".
  It proceeds by first computing the centroids and content (area,
  volume, etc.) of the simplices. Then balls are constructed with these
  centers and contents. Instead of detecting contact between the simplices,
  the pinball method detects contact between the balls. This approximates
  simplex contact. One could use a master/slave labelling and only search
  for contact between different components. However this approach becomes
  complicated when there are more than two components and furthermore it
  cannot detect self-contact within a component. Thus we calculate all
  pairwise penetrations. A contact is reported if two balls penetrate, but their
  associated simplices do not share a vertex in the mesh. (The latter condition
  ensures that contact is not reported between adjacent elements in the mesh.)

  To efficiently detect contact between balls, we store the centers in an
  \ref geom_orq "orthogonal range query"
  (ORQ) data structure. These data structures can report the points
  inside a bounding box (orthogonal range).
  We use the CellArrayStatic class,
  which is a dense array of cells (also commonly called buckets).
  An advantage of using dense cell arrays is that they can be
  constructed quickly; the complexity of constructing
  one is linear in the number of cells and the number of points.
  Because of direct accesss to the cells that contain the points,
  cell arrays typically offer better performance than tree data structures
  such as the kd-tree and octree.
  A potential drawback of using cell arrays is high memory overhead; most
  of the cells could be empty. We have resolved this issue by developing
  a packed data structure that uses only a single pointer per cell. We
  choose the total number of cells to be the number of points. Thus there
  are many cells, which improves the performance of the queries, yet there is
  a small memory overhead. Furthermore, because the data structure is packed
  into two large arrays, it makes efficient use of the cache.

  Note that there two ways of detecting any particular contact. If balls
  \e x and \e y are in contact, then one could either search in a neighborhood
  of \e x or in a neighborhood of \e y. Suppose we want to find the balls
  that penetrate \e x. We need to search a spherical region around the center
  of \e x whose radius is the sum of its radius and the maximum radius of
  any ball. We would start such a search by performing an orthogonal range
  query with a box that contains the spherical search region. Note that this
  approach could be very slow if the balls are not roughly the same size.
  If the maximum radius is much greater than the average radius, each
  ORQ will return many candidate contacts which must then be accepted or
  rejected.

  We solve this efficiency problem by exploiting the fact that a contact
  may be reported in either of two ways. For a ball \e x, we only only search
  for balls with smaller radii. (To be precise, in reporting a contact between
  \e x and \e y, either the radius of \e x is greater than \e y or the radii
  are equal and index of \e x is less than the index of \e y. In this way
  the radii and indices form a compound number which is used to order the
  balls.) Now instead of increasing the radius of \e x by the maximum radius,
  we just double the radius of \e x to form the spherical search domain.
  If a ball \e y with smaller radius penetrates \e x, then its center
  must lie in this search domain.

  \section examples_geom_mesh_pinball_example 3-D Example

  We test the contact on a meshing of the Enterprise and a wall.  To
  generate this test I used Cubit to mesh the Enterprise model in the
  data directory. I used an edge length of 10. Then I made walls with
  dimensions 900x20x900. I moved the first -400 in the y coordinate so
  that the Enterprise penetrates about half of the way through the
  wall. I left second wall centered so it would intersect a large
  portion of the Enterprise.  I also meshed these walls with an edge
  length of 10.

  The meshes were exported in binary Exodus format. I used ncdump to convert
  to ascii Exodus II format. I then used the exodus2iss33.py script to
  convert each mesh to an indexed simplex set. I merged the Enterprise and
  wall meshes with the merge.py script.

  Each mesh
  has 475,485 tetrahedra. For the small penetration there are 506 contacts.
  Contact detection takes 2.16 seconds, which is 4.5 microseconds per
  tetrahedron.
  For the large penetration there are 71,407 contacts.
  Contact detection takes 2.18 seconds, which is 4.6 microseconds per
  tetrahedron.

  To visualize the mesh I extracted the boundary with boundary33.exe.
  I converted the boundary mesh to VTK format with iss2vtk32.exe. Then
  I saved screenshots from ParaView.

  \image html pinballEnterpriseWall.jpg "The Enterprise and a wall. Small penetration."

  \image html pinballEnterpriseWallCenter.jpg "The Enterprise and a wall. Large penetration."

  \image html pinballEnterpriseEdges.jpg "Close-up showing edges."

  Next we consider a graded mesh in which there is a great difference in sizes
  of the tetrahedra.  The mesh shown below, which is used to model a spherical
  projectile impacting a plate, has 28,174 tetrahedra. There are no contacts.
  Contact detection takes 0.26 seconds, which is 5.8 microseconds per
  tetrahedron.

  \image html spherePlateFront.jpg "View from above."
  \image html spherePlateAngle.jpg "Close-up showing the projectile."

  \anchor BelytschkoNeal
  1. Ted Belytschko and Mark O. Neal. <em>Contact-Impact by the Pinball
  Algorithm with Penalty and Lagrangian Methods.</em> International Journal
  for Numerical Methods in Engineering, Vol. 31, 547-572 (1991).
*/

#ifndef DIMENSION
#error DIMENSION must be defined to compile this program.
#endif

#include "iss_io.h"

#include "stlib/geom/mesh/iss/pinballContact.h"
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
         << programName
         << " mesh contacts\n\n"
         << "- mesh is the file name for the input mesh.\n"
         << "- contacts is the file name for recording the contacts.\n\n"
         << "The contacts are reported in the format:\n"
         << "identifier0 identifier1 penetrationX penetrationY penetrationZ\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   static_assert(DIMENSION == 2 || DIMENSION == 3,
                 "The dimension must be 2 or 3.");

   // The simplicial mesh.
   typedef geom::IndSimpSetIncAdj<DIMENSION, DIMENSION> Mesh;
   typedef Mesh::Vertex Point;
   typedef std::tuple<std::size_t, std::size_t, Point, double> Record;

   ads::ParseOptionsArguments parser(argc, argv);
   programName = parser.getProgramName();

   // If they did not specify the mesh and output file.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input solid mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // Check that there are no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   std::vector<Point> velocities(mesh.getVerticesSize(),
                                 ext::filled_array<Point>(0));
   const double maximumRelativePenetration = 0.1;
   std::vector<Record> contacts;
   ads::Timer timer;
   timer.tic();
   const double stableTimeStep =
      geom::pinballContact(mesh, velocities, maximumRelativePenetration,
                           std::back_inserter(contacts));
   double elapsedTime = timer.toc();

   {
      // Write the contacts.
      std::ofstream out(parser.getArgument().c_str());
      out << contacts.size() << '\n';
      for (std::vector<Record>::const_iterator i = contacts.begin();
            i != contacts.end(); ++i) {
         out << std::get<0>(*i) << ' '
             << std::get<1>(*i) << ' '
             << std::get<2>(*i) << '\n';
      }
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   std::cout << "Contact detection took " << elapsedTime << " seconds.\n"
             << "Number of simplices = " << mesh.getSimplicesSize() << '\n'
             << "Stable time step = " << stableTimeStep << '\n'
             << "Number of contacts = " << contacts.size() << '\n'
             << "Time per simplex = "
             << elapsedTime / mesh.getSimplicesSize() * 1e6
             << " microseconds.\n";

   return 0;
}
