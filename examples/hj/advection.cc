// -*- C++ -*-

/*!
  \file examples/hj/advection.cc
  \brief Driver for the constant advection.
*/

/*!
  \page hj_advection The Constant Advection Driver


  \section hj_advection_introduction  Introduction

  \verbatim
  ./advection [-b] [-o] geom distance
  \endverbatim
  - -b Read and write the grid files in binary format.
  - -o Write the advected field files.
  - geom contains geometry information.
  - distance is the the distance grid.


  \section hj_advection_summary Summary

  This driver reads the geometry information and the distance.
  It makes three fields which hold the x, y and z coordinates values.
  It sets the boundary condition for these fields in the ghost fluid region
  using \c cvadv::constant_advection().

  There is a matlab script "advection.m" to visualize the
  advected fields.  (The grid output files must be written in binary format.)


  \section hj_advection_compiling Compiling.

  Make \c advection with gnu "make".


  \section hj_advection_file_format File Format

  The geometry file must be in ascii format.
  The grid files may be in binary or ascii format.  This option is specified
  on the command line.  For ascii files, white space is ignored.  For the
  grids, the first index varies fastest and the last index varies slowest.

  Geometry:
  \verbatim
  x_min y_min z_min x_max y_max z_max
  grid_extent_x grid_extent_y grid_extent_z
  max_distance
  \endverbatim

  x_min etc. describes the Cartesian domain of the grid.  This is
  follewed by the grid extents.  max_distance is how far the distance
  has been computed and how far the boundary condition will be set.
  Anything in the geometry file following this is ignored.

  Fields like distance:
  \verbatim
  grid_extent_x grid_extent_y grid_extent_z
  distance(0,0,0)
  distance(1,0,0)
  distance(2,0,0)
  ...
  distance(l,m,n)
  \endverbatim
*/


#include "stlib/hj/hj.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer/Timer.h"

#include <iostream>
#include <fstream>
#include <string>

#include <cassert>
#include <cmath>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

namespace {

// Global variables.

//! The program name.
std::string programName;

// Local functions.

//! Exit with an error message and usage information.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " [-b] [-o] geom distance\n";
   exit(1);
}

}


//! See advection for documentation.
int
main(int argc, char* argv[]) {
   const std::size_t Dimension = 3;
   typedef std::array<double, Dimension> Point;
   typedef container::MultiArray<double, Dimension> MultiArray;
   typedef MultiArray::SizeList SizeList;

   //
   // Parse the program options and arguments.
   //

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the command line options.
   //

   const bool binary_io = parser.getOption("b");
   if (binary_io) {
      std::cout << "Using binary I/O for the grids.\n";
   }
   else {
      std::cout << "Using ascii I/O for the grids.\n";
   }

   const bool write_data = parser.getOption("o");
   if (write_data) {
      std::cout << "Will write the advected fields.\n";
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   //
   // Parse the program arguments.
   //

   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   //
   // Read the geometry file.
   //
   geom::BBox<double, Dimension> domain;
   SizeList grid_extents;
   double max_distance = 0;
   {
      std::cout << "Reading the geometry file ..." << std::flush;
      std::ifstream file(parser.getArgument().c_str());
      if (! file) {
         std::cerr << "Could not open geometry file.  Exiting...\n";
         exitOnError();
      }
      file >> domain;
      assert(! isEmpty(domain));
      file >> grid_extents;
      file >> max_distance;
      assert(max_distance > 0);
      std::cout << "done.\n";
      std::cout << "The domain is " << domain << ".\n"
                << "The grid extents are " << grid_extents << ".\n"
                << "The maximum distance is " << max_distance << ".\n";
   }

   //
   // Read the distance grid.
   //
   std::cout << "Opening the distance grid..." << std::flush;
   MultiArray distance;
   {
      std::ifstream file(parser.getArgument().c_str());
      assert(file);
      if (binary_io) {
         read(&distance, file);
      }
      else {
         file >> distance;
      }
      assert(distance.extents() == grid_extents);
   }
   std::cout << "done\n";

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   const geom::RegularGrid<Dimension, double> grid(grid_extents, domain);
   MultiArray fields[3];
   fields[0].rebuild(grid_extents);
   fields[1].rebuild(grid_extents);
   fields[2].rebuild(grid_extents);
   {
      const std::size_t i_end = grid_extents[0];
      const std::size_t j_end = grid_extents[1];
      const std::size_t k_end = grid_extents[2];
      Point location = {{0, 0, 0}};
      double value;

      for (std::size_t i = 0; i != i_end; ++i) {
         location[0] = i;
         grid.convertIndexToLocation(&location);
         value = location[0];
         for (std::size_t j = 0; j != j_end; ++j) {
            for (std::size_t k = 0; k != k_end; ++k) {
               fields[0](i, j, k) = value;
            }
         }
      }

      for (std::size_t j = 0; j != j_end; ++j) {
         location[1] = j;
         grid.convertIndexToLocation(&location);
         value = location[1];
         for (std::size_t i = 0; i != i_end; ++i) {
            for (std::size_t k = 0; k != k_end; ++k) {
               fields[1](i, j, k) = value;
            }
         }
      }

      for (std::size_t k = 0; k != k_end; ++k) {
         location[2] = k;
         grid.convertIndexToLocation(&location);
         value = location[2];
         for (std::size_t i = 0; i != i_end; ++i) {
            for (std::size_t j = 0; j != j_end; ++j) {
               fields[2](i, j, k) = value;
            }
         }
      }
   }

   //
   // Write the original fields.
   //
   if (write_data) {
      std::cout << "Writing the original fields..." << std::flush;
      if (binary_io) {
         {
            std::ofstream file("field_x.bin");
            assert(file);
            write(fields[0], file);
         }
         {
            std::ofstream file("field_y.bin");
            assert(file);
            write(fields[1], file);
         }
         {
            std::ofstream file("field_z.bin");
            assert(file);
            write(fields[2], file);
         }
      }
      else {
         {
            std::ofstream file("field_x.txt");
            assert(file);
            file.precision(16);
            file << fields[0];
         }
         {
            std::ofstream file("field_y.txt");
            assert(file);
            file.precision(16);
            file << fields[1];
         }
         {
            std::ofstream file("field_z.txt");
            assert(file);
            file.precision(16);
            file << fields[2];
         }
      }
      std::cout << "done.\n";
   }

   ads::Timer timer;
   double elapsed_time;

   //
   // Fast marching constant advection solve.
   //
   std::cout << "Starting the fast marching constant advection..."
             << std::flush;
   timer.tic();

   for (std::size_t i = 0; i != 3; ++i) {
      hj::advectConstantIntoNegativeDistance(fields[i], grid, distance,
                                             max_distance, 0.0);
   }

   elapsed_time = timer.toc();
   std::cout << "Finished.\n"
             << "Fast marching constant advection took " << elapsed_time
             << " seconds.\n";

   //
   // Write the advected fields.
   //
   if (write_data) {
      std::cout << "Writing the advected fields..." << std::flush;
      if (binary_io) {
         {
            std::ofstream file("advected_field_x.bin");
            assert(file);
            write(fields[0], file);
         }
         {
            std::ofstream file("advected_field_y.bin");
            assert(file);
            write(fields[1], file);
         }
         {
            std::ofstream file("advected_field_z.bin");
            assert(file);
            write(fields[2], file);
         }
      }
      else {
         {
            std::ofstream file("advected_field_x.txt");
            assert(file);
            file.precision(16);
            file << fields[0];
         }
         {
            std::ofstream file("advected_field_y.txt");
            assert(file);
            file.precision(16);
            file << fields[1];
         }
         {
            std::ofstream file("advected_field_z.txt");
            assert(file);
            file.precision(16);
            file << fields[2];
         }
      }
      std::cout << "done.\n";
   }

   return 0;
}
