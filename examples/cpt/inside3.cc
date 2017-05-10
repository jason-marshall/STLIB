// -*- C++ -*-

/*!
  \file inside3.cc
  \brief Determine which grid points are inside the 3-D solid.
*/

/*!
  \page cpt_inside3 Determine which grid points are inside the 3-D solid.


  \section cpt_driver3_introduction  Introduction

  The file <tt>stlib/examples/cpt/inside3.cc</tt> shows how to call the
  functions that determine which grid points are inside the sollid.
  Below are the command line arguments.

  \verbatim
  ./inside3.exe geom brep out
  geom contains the domain and size of the grid.
  brep contains the b-rep, an indexed triangle face set.
  out is the name for the output file (boolean array).
  \endverbatim

  \section cpt_inside3_summary Summary

  This example reads grid information and a b-rep from files, determines
  which grid points are inside the solid, and then writes
  a boolean array to a file.

  \section cpt_input3_compiling Compiling.

  Use gnu "make" on "Makefile" with the command "make inside3.exe" or
  simply "make" to compile all of the examples.

  \section cpt_driver3_input_file_format Input File Format

  The geometry file contains information about the computational domain
  and the extents of the grid.

  \verbatim
  xmin   ymin   zmin   xmax   ymax   zmax
  grid_extent_x   grid_extent_y   grid_extent_z
  \endverbatim

  Description of fields:
  - \c xmin, etc. describe the Cartesian domain spanned by the grid.
  - \c grid_extent_* are the number of grid points in each direction.
  .

  The b-rep file contains a triangle mesh.  (Specifically, an indexed triangle
  face set.)  The first two fields specify the space dimension and the
  simplex dimension.  These must be 3 and 2, respectively.
  \c num_vertices gives the number of vertices
  This is followed by the coordinates of the vertices.
  \c num_faces specifies the number of triangle faces in the mesh.
  Next the faces are enumerated.  A face is specified by the indices of
  three vertices.  (These indices are in the range [0..num_vertices-1].)

  \verbatim
  space_dimension simplex_dimension
  num_vertices
  vertex_0_x    vertex_0_y	vertex_0_z
  vertex_1_x    vertex_1_y	vertex_1_z
  ...
  num_faces
  face_0_index_0	face_0_index_1		face_0_index_2
  face_1_index_0	face_1_index_1		face_1_index_2
  ...
  \endverbatim


  \section cpt_driver3_output_file_format Output File Format


  For the boolean array which records which grid points are inside the
  solid, the three grid extents are written first, followed by the
  grid elements.  The last index varies slowest.

  \verbatim
  x_size y_size z_size
  is_inside(0,0,0)
  is_inside(1,0,0)
  is_inside(2,0,0)
  ...
  is_inside(l,m,n)
  \endverbatim
*/

// We don't use performance measuring for this example.  Make sure that it is
// not defined.
#undef CPT_PERFORMANCE

// The cpt header.
#include "stlib/cpt/cpt.h"

// ADS package.
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <cassert>
#include <cmath>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

//
// Typedefs.
//

#ifdef CPT_SINGLE_PRECISION
//! The number type.
typedef float Number;
#else
//! The number type.
typedef double Number;
#endif


namespace {

//
// Global variables.
//

//! The program name.
std::string programName;

//
// Local functions.
//

//! Exit with an error.
void
exitOnError() {
   std::cerr
         << "Usage:\n"
         << programName << " geom brep out\n"
         << "geom contains the domain and size of the grid.\n"
         << "brep contains the b-rep, an indexed triangle face set.\n"
         << "out is the name for the output file (boolean array).\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   const std::size_t N = 3;
   typedef std::array<std::size_t, N> SizeList;

   std::cout.precision(16);

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // There should be no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   //
   // If they don't specify the three files, print usage information and exit.
   //
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   //
   // Get the Cartesian domain and the grid extents.
   //
   geom::BBox<Number, N> domain;
   SizeList extents;
   {
      // Open the geometry file.
      std::ifstream file(parser.getArgument().c_str());
      // Check that the file exists.
      if (! file) {
         std::cerr << "Could not read the geometry file.  Exiting...\n";
         exitOnError();
      }

      // Get the domain.
      file >> domain;
      // Check for degeneracies.
      assert(! isEmpty(domain));

      // Get the grid extents.
      file >> extents;
      // Check for degeneracies.
      assert(extents[0] > 0 && extents[1] > 0 && extents[2] > 0);
   }


   //
   // Read the b-rep.
   //
   geom::IndSimpSet < N, N - 1 > brep;
   {
      // Open the b-rep file.
      std::ifstream file(parser.getArgument().c_str());
      // Check that the file exists.
      if (! file) {
         std::cerr << "Could not read the b-rep file.  Exiting...\n";
         exitOnError();
      }

      geom::readAscii(file, &brep);
   }


   // Make the boolean array.
   container::MultiArray<bool, N> areInside(extents);

   // The data structure that holds the state.
   cpt::State<N, Number> state;

   // Set the b-rep.
   state.setBRepWithNoClipping(brep.vertices, brep.indexedSimplices);

   std::cout << "\nDetermining which points are inside..." << std::flush;
   ads::Timer timer;
   timer.tic();

   state.determinePointsInside(domain, &areInside);

   double time = timer.toc();
   std::cout << "Finished.\n" << time << " seconds.\n\n";

   //
   // Write the array.
   //
   {
      std::cout << "Writing the array..." << std::flush;
      std::ofstream file(parser.getArgument().c_str());
      assert(file);
      file << areInside;
      std::cout << "done.\n";
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
