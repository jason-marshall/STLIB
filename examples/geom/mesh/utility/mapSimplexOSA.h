// -*- C++ -*-

/*!
  \file mapSimplexOSA.h
  \brief Map a simplex using the orientation, skew and aspect ratio matrices.
*/

/*!
  \page examples_geom_mesh_mapSimplexOSA Map a Simplex using the Orientation, Skew and Aspect Ratio Matrices.
*/

#include "stlib/ads/tensor/SquareMatrix.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ext/array.h"

#include <iostream>
#include <fstream>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
namespace std {USING_STLIB_EXT_ARRAY_IO_OPERATORS;}
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
         << programName << " inputSimplex matrices outputSimplex\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef std::array<double, Dimension> Vertex;
   typedef std::array<Vertex, Dimension + 1> Simplex;
   typedef ads::SquareMatrix<Dimension> Matrix;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   if (parser.getNumberOfArguments() != 3) {
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

   // Read the matrices that comprise the Jacobian.
   Matrix jacobian;
   {
      Matrix m;
      std::ifstream file(parser.getArgument().c_str());
      if (! file) {
         std::cerr << "Bad Jacobian file.  Exiting...\n";
         exitOnError();
      }
      // Orientation.
      file >> jacobian;
      // Skew.
      file >> m;
      jacobian *= m;
      // Aspect ratio.
      file >> m;
      jacobian *= m;
   }


   Vertex v;
   for (std::size_t n = 0; n != Dimension + 1; ++n) {
      ads::computeProduct(jacobian, s[n], &v);
      s[n] = v;
   }

   {
      // Write the simplex.
      std::ofstream file(parser.getArgument().c_str());
      if (! file) {
         std::cerr << "Bad output file.  Exiting...\n";
         exitOnError();
      }
      file << s;
   }

   return 0;
}
