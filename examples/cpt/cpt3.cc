// -*- C++ -*-

/*!
  \file cpt3.cc
  \brief Driver for 3-D closest point transform.
*/

/*!
  \page cpt_driver3 The 3-D CPT Driver


  \section cpt_driver3_introduction  Introduction

  The file <tt>stlib/examples/cpt/cpt3.cc</tt> shows how to call the CPT
  functions using each of the four interfaces.  Below are the command
  line arguments.

  \verbatim
  ./cpt3.exe [-unsigned] [-ascii] [-binary] [-noClipping] [-gradient]
    [-point] [-face] [-flood] [-sign] [-bbox] [-brute] geom brep [out]
  -unsigned: Compute unsigned distance.  (The default is signed distance.)
  -ascii: Write the output files in plain ascii format.
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
  geom contains the domain and size of the grid.
  brep contains the b-rep, an indexed triangle face set.
  out specifies the base name for the output files.  The default output format
  is VTK XML.
  \endverbatim

  \section cpt_driver3_summary Summary

  This driver reads grid information and a triangle mesh from files, computes
  the closest point transform to that triangle mesh on the grid, and then writes
  the transforms to files.  The driver checks the grids to ensure that the CPT
  was computed correctly.  It will print warnings if there are errors.
  If you suspect that the CPT library is not computing the correct result
  for some mesh, you can use the driver to help dignose the problem.
  There are example mesh files in <tt>stlib/data/cpt/3</tt>.

  \section cpt_driver3_compiling Compiling.

  Use gnu "make" on "Makefile".  Edit "Makefile" to suit
  your preferences.  This driver can be compiled into three executables
  which use the three interfaces to the cpt library.  "cpt3" uses the
  standard interface.  "cpt3_c" and "cpt3_f" use the C and fortran
  interfaces, respectively.
  By default, only "cpt3" is made.  Use "make all" to make all the
  executables.

  \section cpt_driver3_input_file_format Input File Format

  The geometry file contains information about the computational domain,
  the lattice, the grids, and options for the CPT.

  \verbatim
  maximumDistance
  xmin   ymin   zmin   xmax   ymax   zmax
  lattice_extent_x   lattice_extent_y   lattice_extent_z
  num_grids
  extent_0_x extent_0_y extent_0_z base_0_x base_0_y base_0_z
  extent_1_x extent_1_y extent_1_z base_1_x base_1_y base_1_z
  ...
  \endverbatim

  Description of fields:
  - The signed distance and closest point will be correctly computed up to
    \c maximumDistance from the surface.  This number must be positive.
  - \c xmin, etc. describe the Cartesian domain spanned by the lattice.
  - \c lattice_extent_* are the number of grid points in each direction
    of the lattice.
  - \c num_grids is the number of grids.
  - \c extent_n_* and base_n_* describe the grid extents (sizes) and
  bases (starting indices).
  .

  The distance for any points farther away than \c maximumDistance is set
  to \c std::numeric_limits<number_type>::max().  If the -l option is
  given, the far away distances are set to \f$ \pm \f$ \c maximumDistance.  Each
  component of the gradient of the distance and closest point for
  far away grid points are set to \c std::numeric_limits<number_type>::max().
  The closest face for far away grid points is set to -1.

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


  \section cpt_driver3_output_file_format Output File Formats

  By default the output is written to a <a href="http://www.vtk.org/">VTK</a>
  XML file.  You can visualize the distance and
  other fields using <a href="http://www.paraview.org/">ParaView</a>.

  You can select a simple text output with the -ascii option.
  For the scalar fields of distance and closest face, the three grid
  extents are written first, followed by the grid elements.  The last
  index varies slowest.

  Distance:
  \verbatim
  x_size y_size z_size
  distance(0,0,0)
  distance(1,0,0)
  distance(2,0,0)
  ...
  distance(l,m,n)
  \endverbatim

  For the vector fields of gradient of distance and closest point, the
  three grid extents are written first, followed by the grid elements.
  The last index varies slowest.

  Closest Point:
  \verbatim
  x_size y_size z_size
  closest_point_x(0,0,0) closest_point_y(0,0,0) closest_point_z(0,0,0)
  closest_point_x(1,0,0) closest_point_y(1,0,0) closest_point_z(1,0,0)
  closest_point_x(2,0,0) closest_point_y(2,0,0) closest_point_z(2,0,0)
  ...
  closest_point_x(l,m,n) closest_point_y(l,m,n) closest_point_z(l,m,n)
  \endverbatim

  \section cpt_driver3_example Example

  You can execute the following example in the directory
  <tt>stlib/examples/cpt</tt>. We copy the geometry and mesh files for a
  canister and generate a VTK unstructred grid file for the triangle mesh.

  \verbatim
  cp ../../data/cpt/3/canister.geom .
  cp ../../data/cpt/3/canister.brep .
  ../geom/mesh/utility/iss2vtk32.exe canister.brep canister.vtu
  \endverbatim

  Below is a visualization of the triangle mesh in ParaView. Open the file
  <tt>canister.vtu</tt>. Use the "extract edges" filter to get the triangle
  edges. Color the edges black and increase the edge thickness to 2 so they
  stand out from the triangle faces.

  \image html CanisterMesh.jpg "The triangle mesh."

  The geometry file specifies that we will compute the distance up to 0.045
  away from the surface on a single 100 x 100 x 100 grid. We compute the signed
  distance and flood fill it to aid in the visualization.

  \verbatim
  ./cpt3.exe -flood canister.geom canister.brep canister
  \endverbatim

  Open canister.vtr, a VTK rectilinear grid file, in ParaView. Hit the Contour
  button and set the iso-surface value to 0. Below we see that the zero
  iso-surface approximates the original triangle mesh.

  \image html CanisterIsoSurface.jpg "The zero iso-surface."

  Select the canister.vtu item in ParaView and color it by the distance
  field. You can hit the Clip button to visualize the volume data.

  \image html CanisterClip.jpg "The signed distance."
*/

// The cpt header.
#include "stlib/cpt/cpt.h"

#ifdef CPT_PERFORMANCE
#include "stlib/cpt/performance.h"
#endif

// ADS package.
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <iomanip>
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

//! Get the numerical extensions.  Used in constructing file names.
/*!
  Convert a number to a zero-padded string that can be used a a file name
  extension.
*/
void
makeNumericExtension(const std::size_t n, std::size_t maximumNumber,
                     std::string* ext) {
   std::ostringstream iss;
   std::size_t width = 1;
   while (maximumNumber / 10 != 0) {
      ++width;
      maximumNumber /= 10;
   }
   iss << std::setw(width) << n;
   *ext = iss.str();
}


//! Exit with an error.
void
exitOnError() {
   std::cerr
         << "Usage:\n"
         << programName << " [-unsigned] [-ascii] [-binary] [-noClipping] [-gradient]\n"
         << " [-point] [-face] [-flood] [-sign] [-bbox] [-brute] geom brep [out]\n"
         << "-unsigned: Compute unsigned distance.  (The default is signed distance.)\n"
         << "-ascii: Write the output files in plain ascii format.\n"
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
         << "geom contains the domain and size of the grid.\n"
         << "brep contains the b-rep, an indexed triangle face set.\n"
         << "out specifies the base name for the output files.  The default\n"
         << "output format is VTK XML.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   const std::size_t N = 3;
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
   const bool areUsingAsciiOutput = parser.getOption("ascii");
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
      std::cerr << "Error. Can't use both the bbox and brute methods.\n";
      exitOnError();
   }
   if (areUsingAsciiOutput && areUsingBinaryOutput) {
      std::cerr << "Error. You can't specify both ascii and binary output.\n";
      exitOnError();
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the inputs and output.
   if (!(parser.getNumberOfArguments() == 2 ||
         parser.getNumberOfArguments() == 3)) {
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
   // Get the distance around the surface on which to compute the closest
   // point transform.
   //
   Number maximumDistance;
   geometryFile >> maximumDistance;
   assert(0 < maximumDistance);

   //
   // Get the Cartesian domain of the lattice in the format:
   // xmin ymin zmin xmax ymax zmax
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
   assert(latticeExtents[0] > 0 && latticeExtents[1] > 0 &&
          latticeExtents[2] > 0);

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

   std::cout << "Initializing the b-rep..." << std::flush;

   ads::Timer timer;
   timer.tic();

   cpt::State<N, Number> state;

   // Determine the Cartesian domain that contains the grids.
   const cpt::State<N, Number>::BBox cartesianDomain = {latticeDomainLower,
                                                        latticeDomainUpper};

   // Set the parameters.
   state.setParameters(cartesianDomain, maximumDistance);

   // Set the b-rep.
   if (areClippingMesh) {
      state.setBRep(brep.vertices, brep.indexedSimplices);
   }
   else {
      state.setBRepWithNoClipping(brep.vertices, brep.indexedSimplices);
   }

   double time = timer.toc();
   std::cout << "Finished.\n"
             << time << " seconds.\n\n";

   //
   // Set the lattice.
   //
   std::cout << "Setting the lattice and adding the grids..." << std::flush;
   timer.tic();

   state.setLattice(latticeExtents, cartesianDomain);

   // For each grid.
   for (std::size_t n = 0; n != numberOfGrids; ++n) {
      // Add the grid.
      state.insertGrid(gridExtents[n], gridBases[n],
                       areComputingGradientOfDistance,
                       areComputingClosestPoint,
                       areComputingClosestFace);
   } // End the loop over the grids.

   time = timer.toc();
   std::cout << "Finished.\n"
             << time << " seconds.\n\n";

   //
   // Write state info.
   //
   state.displayInformation(std::cout);

   std::cout << "\nStarting the closest point transforms..." << std::flush;
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

   time = timer.toc();
   std::cout << "Finished.\n" << time << " seconds.\n\n";

   //
   // Write performance info.
   //
#ifdef CPT_PERFORMANCE
   cpt::performance::print(std::cout);
#endif

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
      timer.tic();

      if (areUsingUnsignedDistance) {
         state.floodFillUnsigned(maximumDistance);
         std::cout << " done.\n";
      }
      else { // Signed distance.
         if (areFindingSignFarAway) {
            state.floodFillDetermineSign(maximumDistance);
         }
         else { // Don't find the sign of the distance if it is unknown.
            state.floodFillAtBoundary(maximumDistance);
         }
      } // End: Signed distance.
      time = timer.toc();
      std::cout << "Finished.\n" << time << " seconds.\n\n";
   } // End: if flood filling is done.


   //
   // If they specified an output base name.
   //
   if (! parser.areArgumentsEmpty()) {
      std::string outputBaseName = parser.getArgument();

      //
      // VTK output is the default.
      //
      if (!(areUsingAsciiOutput || areUsingBinaryOutput)) {
         // Write the distance.
         std::cout << "Writing the VTK grids..." << std::flush;

         // Open the file.
         std::string name = outputBaseName + ".vtr";
         std::ofstream file(name.c_str());
         assert(file);
         file << "<?xml version=\"1.0\"?>\n"
              << "<VTKFile type=\"RectilinearGrid\">\n"
              << "<RectilinearGrid WholeExtent=\"0 "
              << latticeExtents[0] - 1
              << " 0 " << latticeExtents[1] - 1
              << " 0 " << latticeExtents[2] - 1
              << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";
         // Loop over the grids.
         for (std::size_t n = 0; n != numberOfGrids; ++n) {
            file << "<Piece Extent=\""
                 << gridBases[n][0] << " "
                 << gridBases[n][0] + gridExtents[n][0] - 1 << " "
                 << gridBases[n][1] << " "
                 << gridBases[n][1] + gridExtents[n][1] - 1 << " "
                 << gridBases[n][2] << " "
                 << gridBases[n][2] + gridExtents[n][2] - 1 << "\">\n"
                 << "<PointData>\n";

            // Distance.
            file << "<DataArray type=\"Float64\" Name=\"Distance\" format=\"ascii\">\n";
            const Number* iter = state.getGrid(n).getDistance().data();
            const Number* iter_end = iter + stlib::ext::product(gridExtents[n]);
            for (; iter != iter_end; ++iter) {
               file << *iter << "\n";
            }
            file << "</DataArray>\n";

            // The gradient of the distance
            if (areComputingGradientOfDistance) {
               file << "<DataArray type=\"Float64\" Name=\"Gradient of Distance\" NumberOfComponents=\"3\" format=\"ascii\">\n";
               const Point* iter = state.getGrid(n).getGradientOfDistance().data();
               const Point* iter_end = iter +
                 stlib::ext::product(gridExtents[n]);
               for (; iter != iter_end; ++iter) {
                  file << *iter << '\n';
               }
               file << "</DataArray>\n";
            }

            // Closest point.
            if (areComputingClosestPoint) {
               file << "<DataArray type=\"Float64\" Name=\"Closest Point\" NumberOfComponents=\"3\" format=\"ascii\">\n";
               const Point* iter = state.getGrid(n).getClosestPoint().data();
               const Point* iter_end = iter +
                 stlib::ext::product(gridExtents[n]);
               for (; iter != iter_end; ++iter) {
                  file << *iter << '\n';
               }
               file << "</DataArray>\n";
            }

            // Closest face.
            if (areComputingClosestFace) {
               file << "<DataArray type=\"Int32\" Name=\"Closest Face\" format=\"ascii\">\n";
               const std::size_t* iter = state.getGrid(n).getClosestFace().data();
               const std::size_t* iter_end = iter +
                 stlib::ext::product(gridExtents[n]);
               for (; iter != iter_end; ++iter) {
                  file << *iter << "\n";
               }
               file << "</DataArray>\n";
            }

            file << "</PointData>\n";
            file << "<Coordinates>" << "\n";
            for (std::size_t d = 0; d != N; ++d) {
               file << "<DataArray type=\"Float64\">" << "\n";
               const Number lower = latticeDomainLower[d];
               const Number factor = (latticeDomainUpper[d] - lower) /
                                     (latticeExtents[d] - 1);
               for (Index i = gridBases[n][d];
                     i != Index(gridBases[n][d] + gridExtents[n][d]); ++i) {
                  file << lower + factor* i << "\n";
               }
               file << "</DataArray>" << "\n";
            }
            file << "</Coordinates>" << "\n";
            file << "</Piece>\n";
         }
         file << "</RectilinearGrid>\n"
              << "</VTKFile>\n";
         std::cout << "done.\n";
      }
      //
      // Plain ascii or binary output.
      //
      else {
         //
         // Write the distance.
         //
         std::cout << "Writing the distance grids..." << std::flush;

         std::string ext;
         // Loop over the grids.
         for (std::size_t n = 0; n != numberOfGrids; ++n) {
            // Open the file.
            makeNumericExtension(n, numberOfGrids, &ext);
            std::string name = outputBaseName + ext + "dist";

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
         // Write the gradient of the distance
         //
         if (areComputingGradientOfDistance) {
            std::cout << "Writing the: gradient of the distance..." << std::flush;

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
            std::cout << "Writing the: closest point..." << std::flush;

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
   }
   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
