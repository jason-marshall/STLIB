// -*- C++ -*-

/*!
  \file cpt2.cc
  \brief Driver for 2-D closest point transform.
*/

/*!
  \page cpt_driver2 The 2-D CPT Driver


  \section cpt_driver2_introduction  Introduction

  The file <tt>stlib/examples/cpt/cpt2.cc</tt> shows how to call the CPT
  functions using each of the three interfaces.  Below are the command
  line arguments.

  \verbatim
  ./cpt2.exe [-unsigned] [-binary] [noClipping] [gradient]
    [-point] [-face] [-flood] [-sign] [-bbox] [-brute]
    [-localClipping] [-globalClipping=g] [-decimation=d] geom brep [out]
  -unsigned: Compute unsigned distance.  (The default is signed distance.)
  -binary: Write the output files in binary format.
  -noClipping: Do not use the Cartesian domain to clip the b-rep.
  -gradient: Compute the gradient of the distance transform.
  -point: Compute the closest point transform.
  -face: Compute the closest face transform.
  -flood: Flood fill the distance transform.
  -sign: If all of the grid points are far away, find the correct sign of
     the distance.
  -bbox: Use the bounding box method.
  -brute: Use the brute force method.
  -localClipping: Use local clipping to reduce the size of the face polygons.
  -globalClippng: Use global clipping to reduce the size of the face polygons.
     For level 0 no global clippping is performed.  This is the default.
     For level 1 limited global clippping is performed.
     For level 2 full global clippping is performed.
  -decimation: Specify the decimation factor for global clipping.
     By default the factor is unity.  That is, all vertices are used.
  geom contains the domain and size of the grid.
  brep contains the b-rep, an indexed triangle face set.
  out specifies the base name for the output files.
  \endverbatim

  \section cpt_driver2_summary Summary

  This driver reads grid information and a b-rep from files, computes
  the closest point transform to that b-rep on the grid, and then writes
  the transforms to files.  The driver checks the grids to ensure that the CPT
  was computed correctly.  It will print warnings if there are errors.
  If you suspect that the CPT library is not computing the correct result
  for some b-rep, you can use the driver to help dignose the problem.
  There are example b-rep files in the <tt>data</tt>
  directory.  The matlab script "cpt.m" allows you to visualize the
  distance and the closest point transforms.
  (To use this script, the output files must be written in binary format.)


  \section cpt_driver2_compiling Compiling.

  Use gnu "make" on "Makefile" to compile the driver.
  This driver can be compiled into three
  executables which use the three interfaces to the cpt library.
  "cpt2" uses the standard interface.  "cpt2_c" and "cpt2_f" use
  the C and fortran interfaces, respectively.  By default, only
  "cpt2" is made.  Use "make all" to make all the executables.

  \section cpt_driver2_input_file_format Input File Format

  The geometry file contains information about the computational domain,
  the lattice, the grids, and options for the CPT.

  \verbatim
  max_distance
  xmin   ymin   xmax   ymax
  lattice_extent_x   lattice_extent_y
  num_grids
  extent_0_x extent_0_y base_0_x base_0_y
  extent_1_x extent_1_y base_1_x base_1_y
  ...
  \endverbatim

  Description of fields:
  - The signed distance and closest point will be correctly computed up to
    \c max_distance from the surface.  This number must be positive.
  - \c xmin, etc. describe the Cartesian domain spanned by the lattice.
  - \c lattice_extent_* are the number of grid points in each direction
    of the lattice.
  - \c num_grids is the number of grids.
  - \c extent_n_* and base_n_* describe the grid extents (sizes) and
  bases (starting indices).
  .

  The distance for any points farther away than \c max_distance is set
  to \c std::numeric_limits<Number>::max().  If the -l option is
  given, the far away distances are set to \f$ \pm \f$ \c max_distance.  Each
  component of the gradient of the distance and closest point for
  far away grid points are set to \c std::numeric_limits<Number>::max().
  The closest face for far away grid points is set to -1.

  The b-rep file contains a piecwise linear curve.  (Specifically, an indexed
  face set.)  The first two fields specify the space dimension and the
  simplex dimension.  These must be 2 and 1, respectively.
  \c num_vertices gives the number of vertices
  This is followed by the coordinates of the vertices.
  \c num_faces specifies the number of faces in the mesh.
  Next the faces are enumerated.  A face is specified by the indices of
  two vertices.  (These indices are in the range [0..num_vertices-1].)

  \verbatim
  space_dimension simplex_dimension
  num_vertices
  vertex_0_x    vertex_0_y
  vertex_1_x    vertex_1_y
  ...
  num_faces
  face_0_index_0	face_0_index_1
  face_1_index_0	face_1_index_1
  ...
  \endverbatim


  \section cpt_driver2_output_file_format Output File Formats

  For the scalar fields of distance and closest face, the two grid
  extents are written first, followed by the grid elements.  The last
  index varies slowest.

  Distance:
  \verbatim
  x_size y_size
  distance(0,0)
  distance(1,0)
  distance(2,0)
  ...
  distance(m,n)
  \endverbatim

  For the vector fields of gradient of distance and closest point, the
  two grid extents are written first, followed by the grid elements.
  The last index varies slowest.

  Closest Point:
  \verbatim
  x_size y_size
  closest_point_x(0,0) closest_point_y(0,0)
  closest_point_x(1,0) closest_point_y(1,0)
  closest_point_x(2,0) closest_point_y(2,0)
  ...
  closest_point_x(m,n) closest_point_y(m,n)
  \endverbatim

*/

// The cpt header.
#include "stlib/cpt/cpt.h"

// ADS package.
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

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

//! Get the numerical extensions.  Used in constructing file names.
/*!
  Convert a number to a zero-padded string that can be used a a file name
  extension.
*/
void
makeNumericExtension(const std::size_t n, std::size_t max_num,
                     std::string* ext) {
   std::ostringstream iss;
   std::size_t width = 1;
   while (max_num / 10 != 0) {
      ++width;
      max_num /= 10;
   }
   iss << std::setw(width) << n;
   *ext = iss.str();
}


//! Exit with an error.
void
exitOnError() {
   std::cerr
         << "Usage:\n"
         << programName << " [-unsigned] [-binary] [noClipping] [gradient]\n"
         << " [-point] [-face] [-flood] [-sign] [-bbox] [-brute]\n"
         << " [-localClipping] [-globalClipping=g] [-decimation=d] geom brep [out]\n"
         << "-unsigned: Compute unsigned distance.  (The default is signed distance.)\n"
         << "-binary: Write the output files in binary format.\n"
         << "-noClipping: Do not use the Cartesian domain to clip the b-rep.\n"
         << "-gradient: Compute the gradient of the distance transform.\n"
         << "-point: Compute the closest point transform.\n"
         << "-face: Compute the closest face transform.\n"
         << "-flood: Flood fill the distance transform.\n"
         << "-sign: If all of the grid points are far away, find the correct sign of\n"
         << "   the distance.\n"
         << "-bbox: Use the bounding box method.\n"
         << "-brute: Use the brute force method.\n"
         << "-localClipping: Use local clipping to reduce the size of the face polygons.\n"
         << "-globalClippng: Use global clipping to reduce the size of the face polygons.\n"
         << "  For level 0 no global clippping is performed.  This is the default.\n"
         << "  For level 1 limited global clippping is performed.\n"
         << "  For level 2 full global clippping is performed.\n"
         << "-decimation: Specify the decimation factor for global clipping.\n"
         << "  By default the factor is unity.  That is, all vertices are used.\n"
         << "geom contains the domain and size of the grid.\n"
         << "brep contains the b-rep, an indexed triangle face set.\n"
         << "out specifies the base name for the output files.\n";
   exit(1);
}

}



//! The main loop.
int
main(int argc, char* argv[]) {
   const std::size_t N = 2;
   typedef std::array<Number, N> Point;
   typedef std::array<std::size_t, N> SizeList;
   typedef std::ptrdiff_t Index;
   typedef std::array<Index, N> IndexList;

   std::cout.precision(std::numeric_limits<Number>::digits10);

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //
   std::cout << "Getting the options..." << std::flush;
   const bool areUsingUnsignedDistance = parser.getOption("unsigned");
   const bool areUsingBinaryOutput = parser.getOption("binary");
   const bool areClippingMesh = ! parser.getOption("noClipping");
   const bool areComputingGradientOfDistance = parser.getOption("gradient");
   const bool areComputingClosestPoint = parser.getOption("point");
   const bool areComputingClosestFace = parser.getOption("face");
   const bool areFloodFillingDistance = parser.getOption("flood");
   const bool areFindingSignFarAway = parser.getOption("sign");
   const bool areUsingBBoxMethod = parser.getOption("bbox");
   const bool areUsingBruteForceMethod = parser.getOption("brute");
   if (areUsingBBoxMethod && areUsingBruteForceMethod) {
      std::cerr << "Error.  Can't use both the bbox and brute methods.\n";
      exitOnError();
   }
   const bool areUsingLocalClipping = parser.getOption("localClipping");

   std::size_t globalClippingMethod = 0;
   parser.getOption("globalClipping", &globalClippingMethod);
   if (!(globalClippingMethod == 0 || globalClippingMethod == 1 ||
         globalClippingMethod == 2)) {
      std::cerr << "Bad global clipping level.  Exiting..." << "\n";
      exitOnError();
   }

   std::size_t decimationFactor = 1;
   parser.getOption("decimation", &decimationFactor);
   if (decimationFactor < 1) {
      std::cerr << "Bad decimation factor.  Exiting..." << "\n";
      exitOnError();
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the inputs and output.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   //
   // Open the geometry file.
   //
   std::ifstream geometryFile(parser.getArgument().c_str());
   if (! geometryFile) {
      std::cerr << "Could not read the geometry file.  Exiting...\n";
      exitOnError();
   }

   //
   // Get the distance around the curve on which to compute the closest
   // point transform.
   //
   Number maximumDistance;
   geometryFile >> maximumDistance;
   assert(0 < maximumDistance);

   //
   // Get the Cartesian domain of the lattice in the format:
   // xmin ymin xmax ymax
   //
   Point latticeDomainLower, latticeDomainUpper;
   geometryFile >> latticeDomainLower >> latticeDomainUpper;
   // Check for degeneracies.
   for (std::size_t n = 0; n != N; ++n) {
      assert(latticeDomainLower[n] < latticeDomainUpper[n]);
   }

   //
   // Get the lattice extents.
   //
   SizeList latticeExtents;
   geometryFile >> latticeExtents;
   // Check for degeneracies.
   assert(latticeExtents[0] > 0 && latticeExtents[1] > 0);

   //
   // Get the number of grids.
   //
   std::size_t numberOfGrids;
   geometryFile >> numberOfGrids;
   assert(numberOfGrids >= 1);

   //
   // Allocate arrays for the grid index ranges and sizes.
   //
   std::vector<SizeList> gridExtents(numberOfGrids);
   std::vector<IndexList> gridBases(numberOfGrids);

   // For each grid.
   for (std::size_t i = 0; i != numberOfGrids; ++i) {
      //
      // Get grid index ranges.
      //
      geometryFile >> gridExtents[i];
      geometryFile >> gridBases[i];
      // Check for degeneracies.
      for (std::size_t n = 0; n != N; ++n) {
         assert(gridExtents[i][n] > 0);
         assert(gridBases[i][n] >= 0 && gridExtents[i][n] <= latticeExtents[n]);
      }
   }

   //
   // Close the geometry file.
   //
   geometryFile.close();

   //
   // Open the b-rep file.
   //
   std::ifstream brepFile(parser.getArgument().c_str());
   if (! brepFile) {
      std::cerr << "Could not read the b-rep file.  Exiting...\n";
      exitOnError();
   }

   // Read the b-rep.
   geom::IndSimpSet < N, N - 1, Number > brep;
   geom::readAscii(brepFile, &brep);

   brepFile.close();

   //---------------The closest point transform calls-------------------------

   std::cout << "Initializing the grid and the b-rep..." << std::flush;

   cpt::State<N, Number> state;

   // The Cartesian domain that contains the grids.
   const cpt::State<N, Number>::BBox cartesianDomain = {latticeDomainLower,
                                                        latticeDomainUpper};
   // Set the parameters.
   state.setParameters(cartesianDomain, maximumDistance,
                       areUsingLocalClipping, globalClippingMethod,
                       decimationFactor);

   // Set the b-rep.
   if (areClippingMesh) {
      state.setBRep(brep.vertices, brep.indexedSimplices);
   }
   else {
      state.setBRepWithNoClipping(brep.vertices, brep.indexedSimplices);
   }

   //
   // Set the lattice.
   //
   state.setLattice(latticeExtents, cartesianDomain);

   // For each grid.
   for (std::size_t n = 0; n != numberOfGrids; ++n) {
      // Add the grid.
      state.insertGrid(gridExtents[n], gridBases[n],
                       areComputingGradientOfDistance,
                       areComputingClosestPoint,
                       areComputingClosestFace);
   } // End the loop over the grids.

   std::cout << "done.\n";

   //
   // Write state info.
   //
   state.displayInformation(std::cout);

   std::cout << "\nStarting the closest point transform..." << std::flush;
   ads::Timer timer;
   timer.tic();

   //
   // Compute the closest point transform.
   //
   if (areUsingUnsignedDistance) {
      if (areUsingBBoxMethod) {
         state.computeClosestPointTransformUnsignedUsingBBox();
      }
      else if (areUsingBruteForceMethod) {
         state.computeClosestPointTransformUnsignedUsingBruteForce();
      }
      else { // Standard method.
         state.computeClosestPointTransformUnsigned();
      }
   }
   else { // Signed distance.
      if (areUsingBBoxMethod) {
         state.computeClosestPointTransformUsingBBox();
      }
      else if (areUsingBruteForceMethod) {
         state.computeClosestPointTransformUsingBruteForce();
      }
      else { // Standard method.
         state.computeClosestPointTransform();
      }
   }

   double time = timer.toc();
   std::cout << "Finished.\n"
             << "Closest point transforms took " << time
             << " seconds.\n\n";

   //
   // Write state info.
   //
   state.displayInformation(std::cout);

   //
   // Check the data.
   //
   std::cout << "Checking the grids..." << std::flush;
   if (areUsingUnsignedDistance) {
      if (state.areGridsValidUnsigned()) {
         std::cout << "All the grids are valid.\n";
      }
      else {
         std::cout << "The grids are NOT VALID.\n";
      }
   }
   else {
      if (state.areGridsValid()) {
         std::cout << "All the grids are valid.\n";
      }
      else {
         std::cout << "The grids are NOT VALID.\n";
      }
   }


   //
   // The flood fill option.
   //
   if (areFloodFillingDistance) {
      std::cout << "Flood filling the distance..." << std::flush;

      if (areUsingUnsignedDistance) {
         state.floodFillUnsigned(maximumDistance);
         std::cout << "done.\n";
      }
      else {
         if (areFindingSignFarAway) {
            state.floodFillDetermineSign(maximumDistance);
         }
         else {
            state.floodFillAtBoundary(maximumDistance);
         }
         std::cout << "done.\n";
      }
   }

   //
   // If they specified at output base name.
   //
   if (! parser.areArgumentsEmpty()) {
      std::string outputBaseName = parser.getArgument();
      //
      // Write the distance.
      //
      std::cout << "Writing the distance grids..." << std::flush;

      std::string ext;
      // Loop over the grids.
      for (std::size_t n = 0; n != numberOfGrids; ++n) {
         // Open the file.
         makeNumericExtension(n, numberOfGrids, &ext);
         std::string name = outputBaseName + ext + ".dist";

         if (areUsingBinaryOutput) {
            std::ofstream file(name.c_str(), std::ios_base::binary);
            assert(file);
            write(state.getGrid(n).getDistance(), file);
         }
         else {
            std::ofstream file(name.c_str());
            assert(file);
            file << state.getGrid(n).getDistance();
         }
      } // End loop over the grids.

      std::cout << "done.\n";

      //
      // Write the gradient of the distance.
      //
      if (areComputingGradientOfDistance) {
         std::cout << "Writing the: gradient of the distance grids..."
                   << std::flush;

         std::string ext;
         // Loop over the grids.
         for (std::size_t n = 0; n != numberOfGrids; ++n) {
            // Open the file.
            makeNumericExtension(n, numberOfGrids, &ext);
            std::string name = outputBaseName + ext + ".grad";

            if (areUsingBinaryOutput) {
               std::ofstream file(name.c_str(), std::ios_base::binary);
               assert(file);
               write(state.getGrid(n).getGradientOfDistance(), file);
            }
            else {
               std::ofstream file(name.c_str());
               assert(file);
               file << state.getGrid(n).getGradientOfDistance();
            }
         } // End loop over the grids.

         std::cout << "done.\n";
      }

      //
      // Write the closest point.
      //
      if (areComputingClosestPoint) {
         std::cout << "Writing the: closest point grids..." << std::flush;

         std::string ext;
         // Loop over the grids.
         for (std::size_t n = 0; n != numberOfGrids; ++n) {
            // Open the file.
            makeNumericExtension(n, numberOfGrids, &ext);
            std::string name = outputBaseName + ext + ".cp";

            if (areUsingBinaryOutput) {
               std::ofstream file(name.c_str(), std::ios_base::binary);
               assert(file);
               write(state.getGrid(n).getClosestPoint(), file);
            }
            else {
               std::ofstream file(name.c_str());
               assert(file);
               file << state.getGrid(n).getClosestPoint();
            }
         } // End loop over the grids.

         std::cout << "done.\n";
      }

      //
      // Write the closest face.
      //
      if (areComputingClosestFace) {
         std::cout << "Writing the: closest face..." << std::flush;

         std::string ext;
         // Loop over the grids.
         for (std::size_t n = 0; n != numberOfGrids; ++n) {
            // Open the file.
            makeNumericExtension(n, numberOfGrids, &ext);
            std::string name = outputBaseName + ext + ".cf";

            if (areUsingBinaryOutput) {
               std::ofstream file(name.c_str(), std::ios_base::binary);
               assert(file);
               write(state.getGrid(n).getClosestFace(), file);
            }
            else {
               std::ofstream file(name.c_str());
               assert(file);
               file << state.getGrid(n).getClosestFace();
            }
         } // End loop over the grids.

         std::cout << "done.\n";
      }
   }
   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
