// -*- C++ -*-

/*!
  \file examples/amr/interface.ipp
  \brief Track a boundary described by a level set.
*/

#ifndef __examples_amr_interface_h__
#error This is an implementation detail.
#endif

#include "stlib/amr/Orthtree.h"
#include "stlib/amr/PatchAdjacent.h"
#include "stlib/amr/CellData.h"
#include "stlib/amr/Traits.h"
#include "stlib/amr/LocationCellCentered.h"
#include "stlib/amr/writers.h"

#include "stlib/geom/kernel/content.h"
#include "stlib/geom/kernel/Ball.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/utility/string.h"

#include <iostream>
//#include <functional>
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

typedef amr::CellData < Traits, 1 /*Depth*/, 0 /*GhostWidth*/ > CellData;
typedef CellData::FieldTuple FieldTuple;
typedef CellData::Array Array;
typedef Array::Range Range;

typedef amr::PatchAdjacent<CellData, Traits> Patch;
typedef amr::Orthtree<Patch, Traits> Orthtree;
typedef amr::PatchDescriptor<Traits> PatchDescriptor;
typedef amr::LocationCellCentered<Traits> LocationCellCentered;
typedef Orthtree::value_type Node;
typedef Orthtree::iterator iterator;

typedef geom::Ball<Number, DIMENSION> Ball;

//
// Constants.
//

const std::size_t Depth = Orthtree::Patch::PatchData::Depth;
const std::size_t GhostWidth = Orthtree::Patch::PatchData::GhostWidth;
// The array extents for a patch.
const std::size_t ArrayExtent = 2;
const SizeList ArrayExtents = ext::filled_array<SizeList>(ArrayExtent);
const Point Lower = ext::filled_array<Point>(-1);
const Point Upper = ext::filled_array<Point>(2);
// The grid spacing at level zero.
const Number Dx0 = (Upper[0] - Lower[0]) / ArrayExtent;
// The distance between diagonal elements at level zero.
const Number Diag0 = Dx0 * std::sqrt(DIMENSION);

//
// Global variables.
//

//! The program name.
static std::string programName;
static Ball ball;


//
// Functions.
//

void
computeDistance(iterator node) {
   typedef container::MultiIndexRangeIterator<DIMENSION> Iterator;
   // Since there are no ghost cells, we directly access the whole array.
   static_assert(GhostWidth == 0, "The ghost width must be zero.");
   // The functor for computing locations. The third argument holds the array
   // extents. For each coordinate, this is the number of nodes times the
   // array extents for the node.
   LocationCellCentered f(Lower, Upper - Lower,
                          (std::size_t(1) << node->first.getLevel()) *
                          ArrayExtents);
   Array& array = node->second.getPatchData().getArray();
   Point x;
   const Iterator end = Iterator::end(array.range());
   for (Iterator i = Iterator::begin(array.range()); i != end; ++i) {
      // Compute the Cartesian position from the multi-index.
      f(*i, &x);
      // Compute the signed distance from the ball.
      array(*i)[0] = distance(ball, x);
   }
}


bool
_shouldRefine(const SpatialIndex& spatialIndex, const Range& range) {
   typedef container::MultiIndexRangeIterator<DIMENSION> Iterator;
   // Since there are no ghost cells, we directly access the whole array.
   static_assert(GhostWidth == 0, "The ghost width must be zero.");

   if (! spatialIndex.canBeRefined()) {
      return false;
   }

   // The diagonal grid spacing on the array for this node.
   const Number Diag = Diag0 / (1 << spatialIndex.getLevel());
   // The functor for computing locations. The third argument holds the array
   // extents. For each coordinate, this is the number of nodes times the
   // array extents for the node.
   LocationCellCentered f(Lower, Upper - Lower,
                          (std::size_t(1) << spatialIndex.getLevel()) *
                          ArrayExtents);
   Point x;
   const Iterator end = Iterator::end(range);
   for (Iterator i = Iterator::begin(range); i != end; ++i) {
      // Compute the Cartesian position from the multi-index.
      f(*i, &x);
      // Refine if the point is close to the interface.
      if (std::abs(distance(ball, x)) < Diag) {
         return true;
      }
   }

   return false;
}


// Wrapper function that extracts the spatial index and the array range for
// the patch.
bool
shouldRefine(iterator node) {
   return _shouldRefine(node->first,
                        node->second.getPatchData().getArray().range());
}


bool
shouldCoarsen(iterator node) {
   // Compute the spatial index for the parent.
   SpatialIndex parentKey = node->first;
   parentKey.transformToParent();
   // Compute the array range for the parent.
   const Range childRange = node->second.getPatchData().getArray().range();
   const Range parentRange(childRange.extents(),
                           childRange.bases() / std::ptrdiff_t(2));
   // The group should be coarsened if the parent should not be refined.
   return ! _shouldRefine(parentKey, parentRange);
}


//
// Error message.
//

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " [mesh.vtu]" << "\n";
   exit(1);
}

#if 0
//! Return the minimum of two numbers.
struct Minimum {
   Number
   operator()(const Number x, const Number y) const {
      return std::min(x, y);
   }
};
#endif

#if 0
//! Return the maximum of two numbers.
struct Maximum {
   Number
   operator()(const Number x, const Number y) const {
      return std::max(x, y);
   }
};
#endif

#if 0
//! Count the balanced neighbors of a node.
struct CountBalancedNeighbors {
   typedef int result_type;

   result_type
   operator()(const Orthtree& orthtree, const const_iterator node) const {
      int count = 0;
      ads::TrivialOutputIteratorCount output(count);
      orthtree.getBalancedNeighbors(node, output);
      return count;
   }
};
#endif

}


//! The main loop.
int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   if (parser.getNumberOfArguments() > 1) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   std::string baseName;
   if (parser.getNumberOfArguments() != 0) {
      baseName = parser.getArgument();
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   // CONTINUE
   //std::size_t numberOfSteps = 50;
   std::size_t numberOfSteps = 5;
   if (parser.getOption("steps", &numberOfSteps)) {
      if (numberOfSteps < 1) {
         std::cerr << "Bad number of steps = " << numberOfSteps << ".\n";
         exitOnError();
      }
   }

   // There should be no more options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // The interface.
   ball.center = ext::filled_array<Point>(-0.5);
   ball.radius = 0.01;
   Point centerOffset;
   for (std::size_t d = 0; d != Orthtree::Dimension; ++d) {
      centerOffset[d] = 1.5 / (d + 1);
   }
   for (std::size_t i = 0; i != centerOffset.size(); ++i) {
     centerOffset[i] /= numberOfSteps;
   }
   const Number radiusOffset = 0.25 / numberOfSteps;

   // Construct the orthtree.
   Orthtree orthtree(Lower, Upper);

   // Insert a node.
   orthtree.insert(SpatialIndex(), Patch(SpatialIndex(), ArrayExtents));
   // Compute distance on the level 0 node.
   computeDistance(orthtree.begin());

   std::cout << "Tracking the interface...\n";

   ads::Timer timer;
   timer.tic();

   std::vector<iterator> nodes;
   std::size_t countDistance = 0, countCoarsen = 0, countRefine = 0;
   for (std::size_t step = 0; step != numberOfSteps; ++step) {
      // Move the ball.
      ball.center += centerOffset;
      ball.radius += radiusOffset;
      // Refine.
      countRefine += refine(&orthtree, shouldRefine);
      // Balance.
      if (AutomaticBalancing) {
         countRefine += orthtree.balance();
      }
      // Coarsen.
      countCoarsen += coarsen(&orthtree, shouldCoarsen, AutomaticBalancing);
      // Compute the distance for the new ball position.
      for (iterator i = orthtree.begin(); i != orthtree.end(); ++i) {
         computeDistance(i);
      }
      countDistance += orthtree.size();

#if DIMENSION == 3
   // If they specified an output file.
      if (! baseName.empty()) {
         std::string name = baseName;
         std::string extension;
         ads::makeZeroPaddedExtension(step, numberOfSteps - 1, &extension);
         name += extension;
         PatchDescriptor patchDescriptor(ArrayExtents, GhostWidth,
                                         amr::FieldDescriptor(Depth,
                                                              "Distance"));
         // Write the output.
         amr::writeCellDataParaview(name, orthtree, patchDescriptor);
      }
#endif
#if 0
      if (! baseName.empty()) {
         std::string name = baseName;
         std::string extension;
         ads::makeZeroPaddedExtension(step, numberOfSteps - 1, &extension);
         name += extension;
         //name += ".vtu";
         //std::ofstream out(name.c_str());
         //geom::printVtkUnstructuredGrid(out, orthtree);
         writeCellDataParaview(name, orthtree
      }
#endif
   }

   double elapsedTime = timer.toc();

   std::cout << "Done.  Time = " << elapsedTime << " seconds.\n"
             << "The orthtree has " << orthtree.size() << " nodes.\n"
             << "Average size = " << double(countDistance) / numberOfSteps
             << ".\n"
             << "Performed " << countDistance << " distance calculations.\n"
             << "Performed " << countCoarsen << " coarsening operations.\n"
             << "Performed " << countRefine << " refinement operations.\n";



   // CONTINUE HERE
#if 0
   // For some reason using ptr_fun doesn't work.
   //std::ptr_fun(std::min<double>)
   Minimum minimumFunctor;
   const Number minimumDistance =
      geom::accumulate(orthtree, std::numeric_limits<Number>::max(),
                       minimumFunctor);
   Maximum maximumFunctor;
   const Number maximumDistance =
      geom::accumulate(orthtree, -std::numeric_limits<Number>::max(),
                       maximumFunctor);
   const Number meanDistance =
      geom::accumulate(orthtree, Number(0)) / orthtree.size();
   std::cout << "Distance: min = " << minimumDistance
             << ", max = " << maximumDistance
             << ", mean = " << meanDistance << "\n";

   // CONTINUE Count neighbors in an unbalanced tree.
   if (Orthtree::AutomaticBalancing) {
      CountBalancedNeighbors countBalancedNeighbors;
      const int neighbors =
         geom::accumulateFunction(orthtree, int(0), countBalancedNeighbors);
      std::cout << "Total number of adjacent neighbors = " << neighbors << "\n"
                << "Average neighbors per node = "
                << double(neighbors) / orthtree.size() << "\n";
   }
#endif

   return 0;
}
