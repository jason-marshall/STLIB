// -*- C++ -*-

/*!
  \file coarsenCells.h
  \brief Remove the specified cells by collapsing edges.
*/

/*!
  \page examples_geom_mesh_coarsenCells Remove the specified cells by collapsing edges.

*/

#include "../smr_io.h"

#include "stlib/geom/mesh/simplicial/coarsen.h"
#include "stlib/geom/mesh/simplicial/quality.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <string>
#include <sstream>

#include <cassert>

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
         << programName << " cells input output\n"
         << "cells contains the indices of the cells to be removed.\n"
         << "input is the input mesh.\n"
         << "output is the output mesh.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   static_assert((SpaceDimension == 2 && SimplexDimension == 2) ||
                 (SpaceDimension == 3 && SimplexDimension == 2) ||
                 (SpaceDimension == 3 && SimplexDimension == 3),
                 "Those dimensions are not supported.");

   typedef geom::SimpMeshRed<SpaceDimension, SimplexDimension> Mesh;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // If they did not specify the input mesh, the cells and output mesh.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the cell indices.
   std::vector<std::size_t> cells;
   {
      std::ifstream file(parser.getArgument().c_str());
      if (! file) {
         std::cerr << "Bad input file.  Exiting...\n";
         exitOnError();
      }

      std::size_t size;
      file >> size;
      cells.resize(size);
      for (std::size_t n = 0; n != size; ++n) {
         cells[n] = -1;
         file >> cells[n];
      }
   }

   // Read the input mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // Print quality measures for the input mesh.
   std::cout << "Quality of the input mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);
   std::cout << "Number of edges = " << mesh.computeEdgesSize() << "\n";
   geom::printEdgeLengthStatistics(std::cout, mesh);

   std::cout << "\nCoarsening the mesh...\n" << std::flush;
   ads::Timer timer;
   timer.tic();

   // Coarsen the mesh.
   const std::size_t count = geom::coarsen(&mesh, cells.begin(), cells.end());

   double elapsedTime = timer.toc();
   std::cout << "done.\n"
             << "Number of edges collapsed = " << count << "\n"
             << "Coarsening took " << elapsedTime << " seconds.\n";

   // Print quality measures for the coarsened mesh.
   std::cout << "\nQuality of the coarsened mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);
   std::cout << "Number of edges = " << mesh.computeEdgesSize() << "\n";
   geom::printEdgeLengthStatistics(std::cout, mesh);

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
