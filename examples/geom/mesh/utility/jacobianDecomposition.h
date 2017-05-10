// -*- C++ -*-

/*!
  \file jacobianDecomposition.h
  \brief Computes the decomposition of the Jacobian given a simplex.
*/

/*!
  \page examples_geom_mesh_jacobianDecomposition Jacobian Decomposition

  This program calculates the decomposition of the Jacobian of mapping
  from the ideal simplex to the physical simplex.
*/

#include "stlib/geom/mesh/simplex/SimplexJac.h"
#include "stlib/geom/mesh/simplex/decomposition.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <fstream>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

//! The main loop.
int
main(int argc, char* argv[]) {
   typedef std::array<double, Dimension> Vertex;
   typedef std::array<Vertex, Dimension + 1> Simplex;
   typedef geom::SimplexJac<Dimension> SimplexJac;
   typedef SimplexJac::Matrix Matrix;

   ads::ParseOptionsArguments parser(argc, argv);

   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Bad arguments.  Usage:\n"
                << parser.getProgramName() << " simplex\n";
      exit(1);
   }

   // Read the simplex.
   Simplex s;
   {
      std::ifstream file(parser.getArgument().c_str());
      if (! file) {
         std::cerr << "Bad input file.  Exiting...\n";
         exit(1);
      }
      file >> s;
   }

   SimplexJac sj(s);
   Matrix orientation, skew, aspectRatio;
   geom::decompose(sj.getMatrix(), &orientation, &skew, &aspectRatio);

   // CONTINUE: remove the product.
   std::cout << "Jacobian:\n" << sj.getMatrix()
             << "\nOrientation\n" << orientation
             << "\nSkew\n" << skew
             << "\nAspect Ratio\n" << aspectRatio
             << "\nProduct\n" << orientation* skew* aspectRatio;

   return 0;
}
