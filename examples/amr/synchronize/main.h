// -*- C++ -*-

/*!
  \file examples/amr/synchronize/main.ipp
  \brief Synchronize the ghost cells in an array.
*/

/*!
  \page examples_amr_synchronize Example: Synchronize Patches.

  \par
  In these examples, each patch in the AMR mesh stores a multi-array
  of cell-centered data, using the amr::CellData class. Each patch stores 
  a multi-array with 4<sup>D</sup> regular elements. In addition, there is
  one layer of ghost cells. Because we 
  synchronize the patches, we use the amr::PatchAdjacent class, 
  which stores links to adjacent patches.
  These examples refine the patches in the mesh to a uniform level. Next the 
  patches are linked (adjacent neighbors are found for each patch).
  Then the patches are synchronized (data is copied from the adjacent 
  neighbors into the ghost cells).

  \par
  Below we compile the code and go to the build directory.
  \verbatim
  cd stlib/examples
  scons amr/synchronize
  cd release/amr/synchronize\endverbatim
  We run the 3-D example with refinement level 4. The program reports the 
  time to link the patches and to synchronize the patches.
  \verbatim
  $ ./synchronize3 -level=4
  Linking took 0.00665785 seconds.
  Synchronization took 0.0122836 seconds.
  The orthtree has 4096 nodes.\endverbatim
*/

#ifndef __examples_amr_synchronize_main_h__
#error This is an implementation detail.
#endif

#include "stlib/amr/Orthtree.h"
#include "stlib/amr/PatchAdjacent.h"
#include "stlib/amr/CellData.h"
#include "stlib/amr/Traits.h"

#include "stlib/ads/timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>

#include <cassert>
#include <cmath>

using namespace stlib;

namespace {

//
// Classes.
//

//! Refine until the specifed level is reached.
class RefinementCriterion {
   // Data.
private:
   std::size_t _level;

   // Not implemented.
private:
   //! Default constructor not implemented.
   RefinementCriterion();

   // Constructors, etc.
public:
   //! Construct from the desired level.
   RefinementCriterion(const std::size_t level) :
      _level(level) {}

   // Functor.
   //! Return true if the level is less than the desired level.
   template<typename NodeConstIterator>
   bool
   operator()(NodeConstIterator node) {
      return node->first.getLevel() < _level;
   }
};

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
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << "\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   //
   // Types.
   //

   typedef amr::Traits<Dimension, MaximumLevel> Traits;
   typedef Traits::Point Point;
   typedef Traits::SpatialIndex SpatialIndex;
   typedef Traits::SizeList SizeList;

   // In each patch, store a multi-array of cell-centered data. The data depth
   // is unity, so a single number is stored in each cell. There is one 
   // layer of ghost cells for the multi-array.
   typedef amr::CellData < Traits, 1 /*Depth*/, 1 /*GhostWidth*/ > CellData;

   typedef amr::PatchAdjacent<CellData, Traits> Patch;
   typedef amr::Orthtree<Patch, Traits> Orthtree;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // There should be no arguments.
   assert(parser.areArgumentsEmpty());

   std::size_t level = 2;
   parser.getOption("level", &level);

   // There should be no more options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Construct the orthtree.
   Orthtree orthtree(ext::filled_array<Point>(0.),
                     ext::filled_array<Point>(1.));
   // Insert a node.
   {
      SpatialIndex key;
      // Make a patch with 4^D cells.
      orthtree.insert(key, Patch(key, ext::filled_array<SizeList>(4U)));
   }
   // Refine to the specified level.
   amr::refine(&orthtree, RefinementCriterion(level));

   //
   // Linking.
   //
   ads::Timer timer;
   timer.tic();
   orthtree.linkBalanced();
   double elapsedTime = timer.toc();
   std::cout << "Linking took " << elapsedTime << " seconds.\n";

   //
   // Synchronize.
   //
   timer.tic();
   orthtree.synchronizeBalanced();
   elapsedTime = timer.toc();
   std::cout << "Synchronization took " << elapsedTime << " seconds.\n"
             << "The orthtree has " << orthtree.size() << " nodes.\n";

   return 0;
}
