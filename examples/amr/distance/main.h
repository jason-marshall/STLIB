// -*- C++ -*-

/*!
  \file examples/amr/distance/main.h
  \brief Distance the ghost cells in an array.
*/

/*!
  \page examples_amr_distance Example: Capture Zero Iso-surface of a Ball.

  These examples compute the distance to a ball with radius \f$\sqrt{D} / 2\f$,
  where \e D is the space dimension, and center at the origin. The AMR mesh
  spans the domain [0..1]<sup>D</sup>. The refinement criterion is being 
  close to the zero iso-surface.

  \par
  Below we compile the code and then run the 3-D example. (One may generate
  output to visualize the mesh only for the 3-D code. For other dimensions,
  the examples simply report the number of patches.)
  \verbatim
  cd stlib/examples
  scons amr/distance
  cd release/amr/distance
  ./distance3 out\endverbatim

  \par
  The example generates a VTK image file with a .vti extension for each 
  of the patches. It also generate a Paraview file that list the collection
  of impage files.
  Open the file \c out.pvd in ParaView. In the Style section choose the
  "Surface With Edges" representation. Then in the Color section, color by
  "Distance." Finally, click the "Edit Color Map..." button and select
  "Show Color Legend" in the Color Legend tab. A screenshot of the result
  is shown below. We can see patches at levels 2, 3, and 4, which is the 
  maximum allowed level for the 3-D example.

  \image html BallDistance.png "Distance to the surface of a ball."
  \image latex BallDistance.png "Distance to the surface of a ball."
*/

#ifndef __examples_amr_distance_main_h__
#error This is an implementation detail.
#endif

#include "stlib/amr/Orthtree.h"
#include "stlib/amr/PatchAdjacent.h"
#include "stlib/amr/CellData.h"
#include "stlib/amr/Traits.h"
#include "stlib/amr/LocationCellCentered.h"
#include "stlib/amr/writers.h"

#include "stlib/ads/timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>
#include <fstream>

#include <cassert>
#include <cmath>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

using namespace stlib;

namespace {

//
// Types.
//
typedef amr::Traits<DIMENSION, MaximumLevel> Traits;
typedef Traits::Number Number;
typedef Traits::Point Point;
typedef Traits::SpatialIndex SpatialIndex;
typedef Traits::SizeList SizeList;

typedef amr::CellData < Traits, 1 /*Depth*/, 1 /*GhostWidth*/ > CellData;
typedef CellData::FieldTuple FieldTuple;
typedef CellData::Array Array;
typedef CellData::ArrayView ArrayView;

typedef amr::PatchAdjacent<CellData, Traits> Patch;
typedef amr::Orthtree<Patch, Traits> Orthtree;
typedef amr::PatchDescriptor<Traits> PatchDescriptor;
typedef amr::LocationCellCentered<Traits> LocationCellCentered;
typedef Orthtree::value_type Node;
typedef Orthtree::iterator iterator;

//
// Constants.
//

// The array extents for a patch.
const std::size_t ArrayExtent = 4;
const SizeList ArrayExtents = ext::filled_array<SizeList>(ArrayExtent);
// The grid spacing at level zero.
const Number Dx0 = 1./ArrayExtent;
// The distance between diagonal elements at level zero.
const Number Diag0 = Dx0 * std::sqrt(DIMENSION);

//
// Functions.
//

void
computeDistance(const Orthtree& orthtree, iterator node) {
   // The functor for computing locations. The third argument holds the array
   // extents. For each coordinate, this is the number of nodes times the
   // array extents for the node.
   LocationCellCentered f(orthtree.getLowerCorner(),
                          orthtree.getExtents(),
                          (std::size_t(1) << node->first.getLevel()) *
                          node->second.getPatchData().getInteriorExtents());
   const Number radius = 0.5 * std::sqrt(DIMENSION);
   node->second.getPatchData().getArray().
      fill(ext::filled_array<FieldTuple>(std::numeric_limits<Number>::max()));
   ArrayView array = node->second.getPatchData().getInteriorArray();
   Point x;
   for (ArrayView::iterator i = array.begin(); i != array.end(); ++i) {
      // Compute the Cartesian position from the multi-index.
      f(i.indexList(), &x);
      // Compute the signed distance from a ball with center at the origin and
      // radius defined above.
      (*i)[0] = std::sqrt(stlib::ext::dot(x, x)) - radius;
   }
}

bool
shouldRefine(iterator node) {
   if (! node->first.canBeRefined()) {
      return false;
   }
   // Compute the minimum in absolute value of the distances.
   Number minAbs = std::numeric_limits<Number>::max();
   Number value;
   ArrayView array = node->second.getPatchData().getInteriorArray();
   // For each element.
   for (ArrayView::const_iterator i = array.begin(); i != array.end(); ++i) {
      value = std::abs((*i)[0]);
      if (value < minAbs) {
         minAbs = value;
      }
   }
   // The diagonal grid spacing on the array for this node.
   const Number Diag = Diag0 / (1 << node->first.getLevel());
   // Return true if the block is close to the zero iso-surface.
   return minAbs <= Diag;
}

//
// Global variables.
//

//! The program name.
static std::string programName;

//
// Error message.
//

//! Exit with an error message.
void
exitOnError() {
#if DIMENSION == 3
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " [output]\n";
#else
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << '\n';
#endif
   exit(1);
}
}


//! The main loop.
int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

#if DIMENSION == 3
   const std::size_t MaxArguments = 1;
#else
   const std::size_t MaxArguments = 0;
#endif
   if (parser.getNumberOfArguments() > MaxArguments) {
      std::cerr << "Bad number of arguments.\n";
      exitOnError();
   }

#ifdef _OPENMP
   int threads;
   if (parser.getOption("threads", &threads)) {
      omp_set_num_threads(threads);
   }
   std::cout << "Number of processors = " << omp_get_num_procs() << "\n"
             << "Number of threads = " << omp_get_max_threads() << "\n";
#endif

   // There should be no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Construct the orthtree.
   Orthtree orthtree(ext::filled_array<Point>(0.),
                     ext::filled_array<Point>(1.));

   ads::Timer timer;
   timer.tic();

   // Insert a node.
   {
      SpatialIndex key;
      orthtree.insert(key, Patch(key, ArrayExtents));
   }
   // Compute distance on the level 0 node.
   computeDistance(orthtree, orthtree.begin());

   //
   // Refinement.
   //
   std::vector<iterator> nodes, refined;
   refined.push_back(orthtree.begin());
   do {
      nodes.swap(refined);
      refined.clear();
      for (std::vector<iterator>::const_iterator node = nodes.begin();
            node != nodes.end(); ++node) {
         if (shouldRefine(*node)) {
            orthtree.split(*node, std::back_inserter(refined));
         }
      }
      // Compute distance.
#pragma omp parallel for
      for (int n = 0; n < int(refined.size()); ++n) {
         computeDistance(orthtree, refined[n]);
      }
   }
   while (! refined.empty());

   //
   // Balancing.
   //
   nodes.clear();
   orthtree.balance(std::back_inserter(nodes));
   // Compute distance.
#pragma omp parallel for
   for (int n = 0; n < int(nodes.size()); ++n) {
      computeDistance(orthtree, nodes[n]);
   }

   // Synchronize.
   orthtree.synchronizeBalanced();

   double elapsedTime = timer.toc();

   std::cout << "Done.  Time = " << elapsedTime << " seconds.\n"
             << "The orthtree has " << orthtree.size() << " nodes.\n";

#if DIMENSION == 3
   // If they specified an output file.
   if (! parser.areArgumentsEmpty()) {
      PatchDescriptor patchDescriptor(ArrayExtents, 1,
                                      amr::FieldDescriptor(1, "Distance"));
      // Write the output.
      amr::writeCellDataParaview(parser.getArgument(), orthtree,
                                 patchDescriptor);
   }
#endif

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   return 0;
}
