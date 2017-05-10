// -*- C++ -*-

/*!
  \file mst.cc
  \brief Driver for computing a triangulation of a protein surface.
*/

/*!
\page mst_driver The MST Driver

\section mstDriverIntroduction Introduction

This driver reads a molecule in xyzr format, triangulates the visible
surface, and then write the triangle surface mesh as an indexed simplex set.

\section mst_driverUsage Usage

\verbatim
mst.exe [-test] [-error] [-bot] [-cut] [-rubber] [-noMerge] [-spherical]
  [-level=refinement] [-length=a,b] [-radius=r] [-stretch=s] [-area=a]
  [-centers=file] [-individual=file] input.xyzr output.txt
\endverbatim

- If the test option is specified, each atom will be erased and
  re-inserted.  After each such operation, the triangle surface is
  updated.  This tests the dynamic capability.
- If the error option is specified, radius deviation and penetration
  statistics will be computed.
- The bot option specifies that cut clipping (instead of hybrid
  clipping) will be done with a bucket of triangles.
- The cut option specifies that cut clipping (instead of hybrid
  clipping) will be done.
- The rubber option specifies that rubber clipping (instead of
  hybrid clipping) will be done.
- The noMerge option specifies that duplicate vertices will not be
  merged in the output mesh.
- With the spherical option, the edges will be interpreted as spherical
  arcs when performing rubber clipping.  Otherwise, they are interpreted
  as line segments.
- The level option allows you to choose a refinement level for the initial
  triangulation of each atom.
- length specifies the maximum allowed edge length.  The length is
  a r + b where r is the atomic radius.
- radius specifies the probe radius.  By default it is 0.
- stretch specifies the maximum allowed stretch factor in performing
  rubber clipping.  This parameter must be between 0 and 1
  inclusize.  A value of 0 allows infinite stretching.  A value of
  1 does not allow any stretching.  For hybrid clipping, the default value
  is 0.5. For rubber clipping, the default value is 0.
- area specifies the minimum allowed area for a triangle as a fraction of
  the target area.  (The target triangle area is derived from the target
  edge length).  Triangles with smaller area are discarded.
- epsilon is used to determine if a vertex is considered to be on
  the clipping surface when performing cut clipping.  The default
  value is 0.01.
- With the centers option, you specify a file to which the atom
  centers for each triangle in the mesh will be written.
- For the individual option, the meshes of each atom will we output
  in separate files.  The argument is the base name of the files.
- input.xyzr is the input file containing the atom's centers and radii.
- output.txt will contain the surface triangulation.
.
One must specify either the refinement level or the maximum edge length,
but not both.
The input file simply contains an enumeration of the centers and radii
of the atoms in the molecule.  For example, a molecule with a single atom
of radius 1 Angstrom and centered at the origin would have the file:
\verbatim
0.0 0.0 0.0 1.0
\endverbatim

\section mst_driverExample Example

We generate the surface for a "generic" molecule. (It has blobs where the
side chains would be.)  We specify a target
edge length of 2 Angstroms and test the dynamic capability.

\verbatim
cp ../../data/mst/gen.xyzr .
./mst.exe -test -length=0,2 gen.xyzr genL2.txt
\endverbatim

The output is shown below.

\verbatim
Using the function: 0 * radius + 2.
Reading the input file...
The protein has 785 atoms.
Triangulating the surface...
Done.  Time to triangulate = 0.43
Testing the dynamic capability...
For erasing and re-inserting each atom:
  Total time = 5.62
  Time per atom = 0.00715924
  Total number of modified triangles = 177836
  Average number of modified triangles per atom = 226.543
The surface triangulation has 13139 triangles.
Writing the surface file...
Done.
\endverbatim

The triangle mesh has 13,139 triangles.  It takes 0.43 seconds to initially
generate the mesh.  The average time for erasing an atom, re-inserting it,
and updating the mesh is about 7 milliseconds.  On average, this operation
modifies about 227 triangles in the mesh.

We can assess the quality of the surface mesh with the
\ref examples_geom_mesh_utility_quality program.

\verbatim
../geom/mesh/utility/quality32.exe genL2.txt
\endverbatim

This produces the following output.

\verbatim
Space dimension = 3
Simplex dimension = 2
Bounding box = -0.837 -24.639 -12.498 32.714 10.588 32.066
Number of vertices = 12727
Number of simplices = 13139
Number of simplices with positive volume = 13139
content = 7594.8 min = 8.88274e-05 max = 4.54829 mean = 0.578035
determinant: min = 0.000205138 max = 10.5038 mean = 1.33491
mod mean ratio: min = 0.00886307 max = 1 mean = 0.638357
mod cond num: min = 0.00886307 max = 1 mean = 0.638357
edge lengths: min = 0.00219449 max = 4.35144
\endverbatim

We can visualize the mesh by generating a VTK file and viewing it in ParaView.

\verbatim
../geom/mesh/utility/iss2vtk32.exe genL2.txt genL2.vtu
\endverbatim

I open the VTK file with "File->Open Data".  I then extract the edges with
"Filter->Extract Edges".  From the "Display" tab of the edges, I change
the actor color to black and the line width to 2.  The following figure
is a JPEG file generated with "File->Save View Image" with both the surface
mesh and the edges visible.

\image html mstGenL2.jpg "The surface and edges of the generic molecule."



Next we generate a triangulation with a target edge length of 1 Angstrom
and and probe radius of 1 Angstrom.

\verbatim
./mst.exe -test -length=0,1 -radius=1 gen.xyzr genL1R1.txt
\endverbatim

This produces the following output.

\verbatim
Reading the input file...
The protein has 785 atoms.
Triangulating the surface...
Done.  Time to triangulate = 1.172
Testing the dynamic capability...
For erasing and re-inserting each atom:
  Total time = 15.672
  Time per atom = 0.0199643
  Total number of modified triangles = 469427
  Average number of modified triangles per atom = 597.996
The surface triangulation has 37361 triangles.
Writing the surface file...
Done.
\endverbatim

The triangle mesh has 37,361 triangles.  Because it is larger (due to the
probe radius) and finer (due to the smaller edge length)
than before, it takes longer to generate and update the mesh.
It takes 1.172 seconds to initially
generate the mesh.  The average time for erasing an atom, re-inserting it,
and updating the mesh is now about 20 milliseconds.  On average,
this operation modifies about 598 triangles.

To visualize the mesh, we generate the modified condition number and content
of the cells using the \ref examples_geom_mesh_utility_cellAttributes
program.  Then we make a VTK file and view it in ParaView.

\verbatim
../../examples/geom/mesh/utility/cellAttributes32.exe -mcn genL1R1.txt mcn.txt
../../examples/geom/mesh/utility/cellAttributes32.exe -c genL1R1.txt c.txt
../../examples/geom/mesh/utility/iss2vtk32.exe -cellData=mcn.txt,c.txt genL1R1.txt genL1R1.vtu
\endverbatim

\image html mstGenL1R1.jpg "The modified condition number of the triangles."

We see that the triangles have lower quality where the atomic surfaces
intersect.  Away from the interecting curves, the triangles are nearly
equilateral.

*/

#include "stlib/mst/readXyzr.h"
#include "stlib/mst/MolecularSurface.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/utility/string.h"
#include "stlib/geom/mesh/iss/file_io.h"
#include "stlib/geom/mesh/iss/buildFromSimplices.h"

#include <vector>
#include <sstream>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
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
         << programName << "\n"
         << " [-test] [-error] [-bot] [-cut] [-rubber] [-noMerge] [-spherical]\n"
         << " [-level=refinement] [-length=a,b] [-radius=r] [-stretch=s] [-area=a]\n"
         << " [-centers=file] [-individual=file] input.xyzr output.txt\n"
         << "- If the test option is specified, each atom will be erased and\n"
         << "  re-inserted.  After each such operation, the triangle surface is\n"
         << "  updated.  This tests the dynamic capability.\n"
         << "- If the error option is specified, radius deviation and penetration\n"
         << "  statistics will be computed.\n"
         << "- The bot option specifies that cut clipping (instead of hybrid \n"
         << "  clipping) will be done with a bucket of triangles.\n"
         << "- The cut option specifies that cut clipping (instead of hybrid\n"
         << "  clipping) will be done.\n"
         << "- The rubber option specifies that rubber clipping (instead of\n"
         << "  hybrid clipping) will be done.\n"
         << "- The noMerge option specifies that duplicate vertices will not be\n"
         << "  merged in the output mesh.\n"
         << "- With the spherical option, the edges will be interpreted as spherical\n"
         << "  arcs when performing rubber clipping.  Otherwise, they are interpreted\n"
         << "  as line segments.\n"
         << "- The level option allows you to choose a refinement level for the initial\n"
         << "  triangulation of each atom.\n"
         << "- length specifies the maximum allowed edge length.  The length is\n"
         << "  a r + b where r is the atomic radius.\n"
         << "- radius specifies the probe radius.  By default it is 0.\n"
         << "- area specifies the minimum allowed area for a triangle as a fraction of\n"
         << "  the target area.  (The target triangle area is derived from the target\n"
         << "  edge length).  Triangles with smaller area are discarded.\n"
         << "- stretch specifies the maximum allowed stretch factor in performing\n"
         << "  rubber clipping.  This parameter must be between 0 and 1\n"
         << "  inclusize.  A value of 0 allows infinite stretching.  A value of\n"
         << "  1 does not allow any stretching.  For hybrid clipping, the default value\n"
         << "  is 0.5. For rubber clipping, the default value is 0.\n"
         << "- epsilon is used to determine if a vertex is considered to be on\n"
         << "  the clipping surface when performing cut clipping.  The default\n"
         << "  value is 0.01.\n"
         << "- With the centers option, you specify a file to which the atom\n"
         << "  centers for each triangle in the mesh will be written.\n"
         << "- For the individual option, the meshes of each atom will we output\n"
         << "  in separate files.  The argument is the base name of the files.\n"
         << "- input.xyzr is the input file containing the atom's centers and radii.\n"
         << "- output.txt will contain the surface triangulation.\n\n"
         << "One must specify either the refinement level or the maximum edge length,\n"
         << "but not both.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef double Number;
   typedef mst::MolecularSurface<Number> MolecularSurface;
   typedef MolecularSurface::AtomType AtomType;
   typedef MolecularSurface::Point Point;
   typedef MolecularSurface::Triangle Triangle;
   typedef MolecularSurface::IdentifierConstIterator IdentifierConstIterator;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // If they did not specify the input and output files.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Maximum edge length.
   Number edgeLengthSlope = 0, edgeLengthOffset = 0;
   std::string lengthString;
   const bool areUsingEdgeLength = parser.getOption("length", &lengthString);
   if (areUsingEdgeLength) {
      // Get the coefficient strings.
      std::vector<std::string> coefficients;
      const std::size_t size =
         ads::split(lengthString, ",", std::back_inserter(coefficients));
      if (size != 2) {
         std::cerr
               << "Wrong number of coefficients for the edge length function.\n";
         exitOnError();
      }
      // Set the coefficients.
      {
         std::istringstream iss(coefficients[0]);
         iss >> edgeLengthSlope;
      }
      {
         std::istringstream iss(coefficients[1]);
         iss >> edgeLengthOffset;
      }
      std::cout << "Using the function: " << edgeLengthSlope
                << " * radius + " << edgeLengthOffset << ".\n";
   }

   // Using a refinement level.
   int refinementLevel = -1;
   const bool areUsingRefinementLevel =
      parser.getOption("level", &refinementLevel);
   if (areUsingRefinementLevel) {
      std::cout << "Using uniform refinement, level " << refinementLevel
                << ".\n";
   }

   if (areUsingEdgeLength && areUsingRefinementLevel) {
      std::cerr << "Error: You cannot specify both an edge length function and a refinement level.\n";
      exitOnError();
   }
   else if (! areUsingEdgeLength && ! areUsingRefinementLevel) {
      std::cerr << "Error: You must specify either an edge length function or a refinement level.\n";
      exitOnError();
   }

   if (areUsingRefinementLevel && refinementLevel < 0) {
      std::cerr << "Error: Bad refinement level.\n";
      exitOnError();
   }

   // The data structure for the molecular surface.
   MolecularSurface molecularSurface(edgeLengthSlope, edgeLengthOffset,
                                     refinementLevel);

   if (parser.getOption("spherical")) {
      molecularSurface.setAreUsingCircularEdges(true);
   }

   // Probe radius.
   Number probeRadius = 0;
   parser.getOption("radius", &probeRadius);

   // Minimum allowed triangle area.
   {
      Number minimumAllowedArea = -1;
      if (parser.getOption("area", &minimumAllowedArea)) {
         // Check for bad values.
         if (minimumAllowedArea < 0) {
            std::cerr << "Error: Bad value for the minimum allowed area.\n";
            exitOnError();
         }
         // Set the minimum allowed area.
         molecularSurface.setMinimumAllowedArea(minimumAllowedArea);
      }
   }

   // Epsilon for cut clipping.
   Number epsilonForCutClipping = 0.01;
   parser.getOption("epsilon", &epsilonForCutClipping);
   if (epsilonForCutClipping < 0) {
      std::cerr << "Error: Bad value for epsilon.  It must be non-negative.\n";
      exitOnError();
   }
   molecularSurface.setEpsilonForCutClipping(epsilonForCutClipping);

   // If we are using bot clipping.
   bool areUsingBotClipping = false;
   if (parser.getOption("bot")) {
      areUsingBotClipping = true;
   }

   // If we are using cut clipping.
   bool areUsingCutClipping = false;
   if (parser.getOption("cut")) {
      areUsingCutClipping = true;
   }

   // If we are using rubber clipping.
   bool areUsingRubberClipping = false;
   if (parser.getOption("rubber")) {
      areUsingRubberClipping = true;
   }

   if (int(areUsingBotClipping) + int(areUsingCutClipping) +
         int(areUsingRubberClipping) > 1) {
      std::cerr << "Error: you may not specify more than one of the bot, cut,\n"
                << "and rubber options.\n";
      exitOnError();
   }

   // Stretch factor for rubber clipping.
   Number maximumStretchFactor;
   // If we are only using rubber clipping.
   if (areUsingRubberClipping) {
      // Allow infinite stretching by default.
      maximumStretchFactor = 0;
   }
   else {
      // Othwise, allow limited stretching by default.
      maximumStretchFactor = 0.5;
   }
   parser.getOption("stretch", &maximumStretchFactor);
   if (maximumStretchFactor < 0 || maximumStretchFactor > 1) {
      std::cerr << "Error: Bad value for the stretch factor.\n"
                << "It must be between 0 and 1 inclusive.\n";
      exitOnError();
   }
   molecularSurface.setMaximumStretchFactor(maximumStretchFactor);

   // The atoms.
   std::vector<AtomType> atoms;

   // Read the input file.
   std::cout << "Reading the input file...\n" << std::flush;
   mst::readXyzr<Number>(parser.getArgument().c_str(),
                         std::back_inserter(atoms));

   // Offset by the probe radius.
   if (probeRadius != 0) {
      for (std::vector<AtomType>::iterator i = atoms.begin(); i != atoms.end();
            ++i) {
         // Make sure we don't have an atom with negative radius.
         if (i->radius + probeRadius <= 0) {
            std::cerr << "Bad value for the probe radius.\n";
            exitOnError();
         }
         i->radius = i->radius + probeRadius;
      }
   }

   std::cout << "The protein has " << atoms.size() << " atoms.\n";

   // Triangulate the visible surface.
   std::cout << "Triangulating the surface...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Insert each atom.
   for (std::size_t i = 0; i != atoms.size(); ++i) {
      molecularSurface.insert(i, atoms[i]);
   }
   // Update the surface.
   if (areUsingBotClipping) {
      // Use cut clipping with a bucket of triangles.
      molecularSurface.updateSurfaceUsingBot();
   }
   else if (areUsingCutClipping) {
      // Use cut clipping.
      molecularSurface.updateSurfaceWithCutClipping();
   }
   else if (areUsingRubberClipping) {
      // Use rubber clipping.
      molecularSurface.updateSurfaceWithRubberClipping();
   }
   else {
      // Use hybrid clipping.
      molecularSurface.updateSurface();
   }

   double elapsedTime = timer.toc();
   std::cout << "Done.  Time to triangulate = " << elapsedTime
             << "\n" << std::flush;


   //
   // If we are testing the dynamic capability, erase and insert each atom.
   //
   if (parser.getOption("test")) {
      std::cout << "Testing the dynamic capability...\n" << std::flush;
      timer.tic();

      std::size_t modifiedCount = 0;
      std::vector<int> modifiedIndices;
      std::vector<Triangle> modifiedTriangles;
      // For each atom.
      for (std::size_t i = 0; i != atoms.size(); ++i) {
         // Erase the atom.
         molecularSurface.erase(i);
         // Re-insert the atom.
         molecularSurface.insert(i, atoms[i]);
         // Update the surface.
         if (areUsingBotClipping) {
            // Use cut clipping with a bucket of triangles.
            molecularSurface.updateSurfaceUsingBot();
         }
         else if (areUsingCutClipping) {
            // Use cut clipping.
            molecularSurface.updateSurfaceWithCutClipping();
         }
         else if (areUsingRubberClipping) {
            // Use rubber clipping.
            molecularSurface.updateSurfaceWithRubberClipping();
         }
         else {
            // Use hybrid clipping.
            molecularSurface.updateSurface();
         }
         // Get the modified triangles.
         molecularSurface.getModifiedTriangleIndices
         (std::back_inserter(modifiedIndices));
         for (std::vector<int>::const_iterator j = modifiedIndices.begin();
               j != modifiedIndices.end(); ++j) {
            modifiedTriangles.push_back(molecularSurface.getTriangle(*j));
         }
         modifiedCount += modifiedIndices.size();
         modifiedIndices.clear();
         modifiedTriangles.clear();
      }

      elapsedTime = timer.toc();
      std::cout << "For erasing and re-inserting each atom:\n"
                << "  Total time = " << elapsedTime << "\n"
                << "  Time per atom = " << (elapsedTime / atoms.size()) << "\n"
                << "  Total number of modified triangles = "
                << modifiedCount << "\n"
                << "  Average number of modified triangles per atom = "
                << Number(modifiedCount) / atoms.size() << "\n";
   }

   // Get the triangles.
   std::vector<Point> vertices;
   molecularSurface.getTriangleVertices(std::back_inserter(vertices));

   std::cout << "The surface triangulation has "
             << vertices.size() / 3
             << " triangles.\n" << std::flush;

   // Print information about the number of triangles for each atom.
   //molecularSurface.printInformation(std::cout);

   if (parser.getOption("error")) {
      Number minimumRadiusDeviation,
             maximumRadiusDeviation,
             meanRadiusDeviation,
             minimumPenetration,
             maximumPenetration,
             meanPenetration;
      std::cout << "Computing error statistics.\n" << std::flush;
      molecularSurface.computeErrorStatistics(&minimumRadiusDeviation,
                                              &maximumRadiusDeviation,
                                              &meanRadiusDeviation,
                                              &minimumPenetration,
                                              &maximumPenetration,
                                              &meanPenetration);
      std::cout << "Radius deviation: min = " << minimumRadiusDeviation
                << ", max = " << maximumRadiusDeviation
                << ", mean = " << meanRadiusDeviation << "\n"
                << "Penetration: min = " << minimumPenetration
                << ", max = " << maximumPenetration
                << ", mean = " << meanPenetration << "\n" << std::flush;
   }

   // Build an indexed simplex set.
   geom::IndSimpSetIncAdj<3, 2, Number> surfaceTriangulation;
   if (parser.getOption("noMerge")) {
      // Build the ISS without merging duplicate vertices.
      typedef geom::IndSimpSetIncAdj<3, 2, Number>::IndexedSimplex
      IndexedSimplex;
      std::vector<IndexedSimplex> indexedSimplices(vertices.size() / 3);
      for (std::size_t i = 0; i != indexedSimplices.size(); ++i) {
         indexedSimplices[i][0] = 3 * i;
         indexedSimplices[i][1] = 3 * i + 1;
         indexedSimplices[i][2] = 3 * i + 2;
      }
      build(&surfaceTriangulation, vertices.size(), &vertices[0],
            indexedSimplices.size(), &indexedSimplices[0]);
   }
   else {
      // Build the ISS by merging duplicate vertices.
      geom::buildFromSimplices(vertices.begin(), vertices.end(),
                               &surfaceTriangulation);
   }

   // Write the tesselation of the surface.
   std::cout << "Writing the surface file...\n" << std::flush;
   {
      std::ofstream out(parser.getArgument().c_str());
      geom::writeAscii(out, surfaceTriangulation);
   }
   std::cout << "Done.\n" << std::flush;

   // The atom centers option.
   std::string centersName;
   if (parser.getOption("centers", &centersName)) {
      // Build an array of the atom centers.
      std::vector<Point> centers;
      for (std::size_t n = 0; n != molecularSurface.getTrianglesSize(); ++n) {
         if (molecularSurface.isTriangleNonNull(n)) {
            centers.push_back(molecularSurface.getAtomForTriangle(n).center);
         }
      }
      // The number of atom centers must match the number of triangles.
      assert(centers.size() == surfaceTriangulation.indexedSimplices.size());
      // Open the output stream.
      std::ofstream out(centersName.c_str());
      // Write the number of points.
      out << int(centers.size()) << "\n";
      // Write the center points.
      for (std::vector<Point>::const_iterator i = centers.begin();
            i != centers.end(); ++i) {
         out << *i << "\n";
      }
   }

   // The individual option.
   std::string baseName;
   if (parser.getOption("individual", &baseName)) {
      geom::IndSimpSetIncAdj<3, 2, Number> mesh;
      // For each atom.
      for (IdentifierConstIterator
            i = molecularSurface.getIdentifiersBeginning();
            i != molecularSurface.getIdentifiersEnd(); ++i) {
         // The atom identifier.
         const std::size_t identifier = *i;
         // Get the mesh.
         molecularSurface.buildMeshUsingRubberClipping(identifier, &mesh);

         // Make the file name.
         std::ostringstream name;
         name << baseName << identifier << ".txt";
         // Open the file.
         std::ofstream out(name.str().c_str());
         // Write the mesh.
         geom::writeAscii(out, mesh);
      }
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
