// -*- C++ -*-

/*!
  \file cellAttributes.h
  \brief Computes cell attributes for simplicial meshes.
*/

/*!
  \page examples_geom_mesh_utility_cellAttributes Cell Attributes


  \section cellAttributesIntroduction Introduction

  This program computes the cell attributes of a simplicial mesh.
  It reads meshes in ascii format.

  \section cellAttributesUsage Usage

  \verbatim
  cellAttributesNM.exe [-mr|meanRatio] [-mmr|modifiedMeanRatio]
    [-cn|conditionNumber] [-mcn|modifiedConditionNumber]
    [-c|content] inputMesh outputArray
  \endverbatim

  Here N is the space dimension and M is the simplex dimension.
  See \ref geom_mesh_simplex for a description of the mean ratio and
  condition number quality metrics.  "Content" is the dimension-independent
  name for length, area, volume, etc.

  The file format for the output cell data follows.
  \verbatim
  attributeName
  numberOfCells
  attribute_0
  attribute_1
  ...
  \endverbatim
  The first line contains the name of the attribute.
  The second line gives the number of cells.
  Next the attributes for each cell are enumerated.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/cellAttributes.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <cassert>

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
         << "  [-mr|meanRatio] [-mmr|modifiedMeanRatio]\n"
         << "  [-cn|conditionNumber] [-mcn|modifiedConditionNumber]\n"
         << "  [-c|content]\n"
         << "  inputMesh outputArray\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSet<SpaceDimension, SimplexDimension> ISS;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // If they did not specify the input mesh and output array.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the input mesh.
   ISS mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // Open the output array file.
   std::ofstream output(parser.getArgument().c_str());
   if (! output) {
      std::cerr << "Bad output file.  Exiting...\n";
      exitOnError();
   }

   // Make the cell attributes array.  Initialize it with a bad value.
   std::vector<double> attributes(mesh.indexedSimplices.size(),
                                  std::numeric_limits<double>::max());

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   std::string attributeName;
   if (parser.getOption("mr") || parser.getOption("meanRatio")) {
      attributeName = "Mean Ratio";
      geom::computeMeanRatio(mesh, attributes.begin());
   }
   else if (parser.getOption("mmr") || parser.getOption("modifiedMeanRatio")) {
      attributeName = "Modified Mean Ratio";
      geom::computeModifiedMeanRatio(mesh, attributes.begin());
   }
   else if (parser.getOption("cn") || parser.getOption("conditionNumber")) {
      attributeName = "Condition Number";
      geom::computeConditionNumber(mesh, attributes.begin());
   }
   else if (parser.getOption("mcn") ||
            parser.getOption("modifiedConditionNumber")) {
      attributeName = "Modified Condition Number";
      geom::computeModifiedConditionNumber(mesh, attributes.begin());
   }
   else if (parser.getOption("c") ||
            parser.getOption("content")) {
      attributeName = "Content";
      geom::computeContent(mesh, attributes.begin());
   }
   else {
      std::cerr << "Bad option.\n";
      exitOnError();
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Print the attribute name.
   output << attributeName << "\n";
   // Print the attributes array.
   output.precision(std::numeric_limits<double>::digits10);
   output << attributes;

   return 0;
}
