// -*- C++ -*-

/*!
  \file simplexQuality.h
  \brief Computes the quality metrics for a simplex.
*/

/*!
  \page examples_geom_mesh_simplexQuality Simplex Quality

  This program asseses the quality of a single simplex.
*/

#include "stlib/geom/mesh/simplex/SimplexModMeanRatio.h"
#include "stlib/geom/mesh/simplex/SimplexModCondNum.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <fstream>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
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
         << programName << " simplex\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef std::array<double, Dimension> Vertex;
   typedef std::array < Vertex, Dimension + 1 > Simplex;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the simplex.
   Simplex s;
   {
      std::ifstream file(parser.getArgument().c_str());
      if (! file) {
         std::cerr << "Bad input file.  Exiting...\n";
         exitOnError();
      }
      file >> s;
   }

   // Print quality measures for the input mesh.
   geom::SimplexModMeanRatio<Dimension> smmr;
   smmr.setFunction(s);
   geom::SimplexModCondNum<Dimension> smcn;
   smcn.setFunction(s);
   std::cout << "Space dimension = " << Dimension << "\n"
             << "content = " << smmr.computeContent() << "\n"
             << "determinant = " << smmr.getDeterminant() << "\n"
             << "mod mean ratio = " << 1.0 / smmr() << "\n"
             << "mod cond num = " << 1.0 / smcn() << "\n";

   return 0;
}
