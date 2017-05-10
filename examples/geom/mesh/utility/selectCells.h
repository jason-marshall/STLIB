// -*- C++ -*-

/*!
  \file selectCells.h
  \brief Select cells that have an attribute in a specified range.
*/

/*!
  \page examples_geom_mesh_selectCells Select cells that have an attribute in a specified range.

  \section selectCellsIntroduction Introduction

  This program selects cells of a simplicial mesh with attributes in a
  specified range.

  \section selectCellsUsage Usage

  \verbatim
  selectCells.exe [-lower=l] [-upper=u] inputAttributes outputIndices
  \endverbatim

  The file format for the input cell data follows.
  \verbatim
  attributeName
  numberOfCells
  attribute_0
  attribute_1
  ...
  \endverbatim
  The first line contains the name of the attribute.
  The second line gives the number of cells.
  In the rest of the file, the attributes for each cell are enumerated.

  The output format for the indices is:
  \verbatim
  numberOfIndices
  index_0
  index_1
  ...
  \endverbatim
*/


#include "../iss_io.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ext/vector.h"

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
         << programName << " [-lower=l] [-upper=u] inputAttributes outputIndices\n"
         << "  inputAttributes contains the attribute name and an array of values.\n"
         << "  outputIndices is the output array of indices.\n";

   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the options.
   //

   double lower = - std::numeric_limits<double>::max();
   parser.getOption("lower", &lower);
   double upper = std::numeric_limits<double>::max();
   parser.getOption("upper", &upper);

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // If they did not specify the input attributes and the output indices.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the attributes.
   std::vector<double> attributes;
   {
      std::ifstream file(parser.getArgument().c_str());
      if (! file) {
         std::cerr << "Bad attribute data file.\n";
         exitOnError();
      }
      // Read the attribute name
      std::string attributeName;
      std::getline(file, attributeName);
      // Read the attribute data.
      file >> attributes;
   }

   //
   // Get the cells.
   //

   std::vector<std::size_t> cells;
   // Loop over the cells.
   for (std::size_t n = 0; n != attributes.size(); ++n) {
      if (lower <= attributes[n] && attributes[n] <= upper) {
         cells.push_back(n);
      }
   }

   // Write the indices of the cells.
   {
      std::ofstream file(parser.getArgument().c_str());
      file << int(cells.size()) << "\n";
      for (std::vector<std::size_t>::const_iterator i = cells.begin();
            i != cells.end(); ++i) {
         file << *i << "\n";
      }
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
