// -*- C++ -*-

/*!
  \file randomize.h
  \brief Randomly move the vertex locations for simplicial meshes.
*/

/*!
  \page examples_geom_mesh_randomize Randomly Move Vertices in Simplicial Meshes


  \section mesh_randomize_introduction Introduction

  This program reads a mesh, randomly moves the positions of the interior
  vertices and then writes the distorted mesh.  This is useful for producing
  distorted meshes to test the mesh optimization functionality.


  \section mesh_randomize_compiling Compiling.

  The makefile defines the Dimension macro to compile this code
  into \c randomize2, which distorts triangle meshes in 2-D, and into
  \c randomize3, which distorts tetrahedral meshes in 3-D.


  \section mesh_randomize_usage Usage

  Command line options:
  \verbatim
  randomizeN distance mesh_in mesh_out
  \endverbatim
  Here N is either 2 or 3.
  Each interior vertex will be moved up to distance in each direction.
  mesh_in is the input mesh.  mesh_out is the file name for
  the distorted mesh.

  The programs read and write meshes in ascii format.
*/

#include "../iss_io.h"

#include "stlib/geom/mesh/iss/quality.h"
#include "stlib/geom/mesh/iss/set.h"
#include "stlib/geom/mesh/iss/transform.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <string>
#include <sstream>

#include <cassert>
#include <cstdlib>

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
         << programName << " distance input output\n"
         << "Each interior vertex will be moved up to distance in each direction.\n";
   exit(1);
}


//! Add a random offset to a point.
template<std::size_t N>
class Randomize :
   public std::unary_function < const std::array<double, N>&,
      const std::array<double, N>& > {
private:
   //! The base class.
   typedef std::unary_function < const std::array<double, N>&,
           const std::array<double, N>& >
           Base;

   //! Maximum amount to offset in each dimension.
   double _offset;
   mutable std::array<double, N> _result;

   //! Default constructor not implemented.
   Randomize();

   //! Assignment operator constructor not implemented.
   Randomize&
   operator=(const Randomize&);

public:
   //! Argument type.
   typedef typename Base::argument_type argument_type;
   //! Result type.
   typedef typename Base::result_type result_type;

   //! Construct from an offset distance.
   Randomize(const double offset) :
      _offset(offset) {}

   //! Copy constructor.
   Randomize(const Randomize& other) :
      _offset(other._offset) {}

   //! Functor.  Add a random offset.
   result_type
   operator()(argument_type x) const {
      _result = x;
      for (std::size_t i = 0; i != _result.size(); ++i) {
         _result[i] += _offset * (2. * rand() * (1. / RAND_MAX) - 1.);
      }
      return _result;
   }
};

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef geom::IndSimpSetIncAdj<Dimension, Dimension> Mesh;

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   // If they did not specify the distance, the input mesh, and the output mesh.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // The distance.
   double distance = 0;
   {
      std::istringstream iss(parser.getArgument().c_str());
      iss >> distance;
      if (distance <= 0) {
         std::cerr << "Bad distance value.\n";
         exitOnError();
      }
   }

   // Read the input mesh.
   Mesh mesh;
   readAscii(parser.getArgument().c_str(), &mesh);

   // Print quality measures for the input mesh.
   std::cout << "Quality of the input mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);

   std::cout << "\nDistorting the mesh..." << std::flush;

   std::vector<std::size_t> interior;
   geom::determineInteriorVertices(mesh, std::back_inserter(interior));
   Randomize<Dimension> f(distance);
   geom::transform(&mesh, interior.begin(), interior.end(), f);

   std::cout << "done.\n\n";

   // Print quality measures for the output mesh.
   std::cout << "Quality of the distorted mesh:\n";
   geom::printQualityStatistics(std::cout, mesh);

   // Write the output mesh.
   writeAscii(parser.getArgument().c_str(), mesh);

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
