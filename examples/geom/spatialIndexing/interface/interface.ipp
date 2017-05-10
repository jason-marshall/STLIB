// -*- C++ -*-

/*!
  \file examples/geom/spatialIndexing/interface.ipp
  \brief Track a boundary described by a level set.
*/

#ifndef __examples_geom_spatialIndexing_interface_ipp__
#error This is an implementation detail.
#endif

#include "geom/spatialIndexing/OrthtreeMap.h"
#include "geom/kernel/content.h"

#include "stlib/ads/array/Array.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/utility/string.h"

#include <iostream>
#include <functional>
#include <fstream>

#include <cassert>
#include <cmath>

namespace {

//
// Types.
//

//! The number type.
typedef double Number;
//! A Cartesian point.
typedef ads::FixedArray<Dimension, Number> Point;

//
// Functor that compute the distance from the surface of an N-sphere.
//

//! The distance from an N-sphere.
struct Sphere {
   Point center;
   Number radius;

   Number
   operator()(const Point& x) const {
      return geom::computeDistance(x, center) - radius;
   }
};

//! Return the distance from the interface at the center of the leaf.
template<typename _UnaryFunction>
class Distance {
   //
   // Public types.
   //
public:

   //! The result type.
   typedef Number result_type;

   //
   // Member data.
   //
private:

   const _UnaryFunction& _distance;

   //
   // Not implemented.
   //
private:

   Distance();
   Distance&
   operator=(const Distance&);

   //
   // Public functions.
   //
public:

   Distance(const _UnaryFunction& distance) :
      _distance(distance) {
   }

   result_type
   operator()(const Point& center) const {
      return _distance(center);
   }
};

//
// Functors to determine if we should refine or coarsen.
//

//! Return true if the element should be refined.
struct Refine {
   //! The result type.
   typedef bool result_type;

   //! Compare the squared distance with the squared half length of the diagonal.
   template<typename _Orthtree>
   result_type
   operator()(const _Orthtree& orthtree,
              const typename _Orthtree::const_iterator i) const {
      const Point& extents = orthtree.getExtents(i->first);
      const Number halfLength = 0.5 * ads::computeMaximum(extents);
      return i->second * i->second <
             halfLength * halfLength * _Orthtree::Dimension;
   }
};

//! Return true if the element should be coarsened.
struct Coarsen {
   //! The result type.
   typedef bool result_type;

   //! Compare the squared distance with the squared half length of the diagonal for each node in the group.
   template<typename _Orthtree>
   result_type
   operator()(const _Orthtree& orthtree,
              typename _Orthtree::const_iterator i) const {
      const Point& extents = orthtree.getExtents(i->first);
      const Number halfLength = 0.5 * ads::computeMaximum(extents);
      const Number squaredHalfDiagonal =
         halfLength * halfLength * _Orthtree::Dimension;
      // Loop over the nodes in the group.
      for (int n = 0; n != _Orthtree::NumberOfOrthants; ++n, ++i) {
         // If the interface passes through this element.
         if (i->second * i->second < squaredHalfDiagonal) {
            // We should not coarsen the group.
            return false;
         }
      }
      return true;
   }
};

//
// Orthtree types.
//

typedef geom::OrthtreeMap < Dimension, MaximumLevel, Number, AutomaticBalancing,
        Distance<Sphere>,
        Distance<Sphere>,
        Refine,
        Coarsen,
        Distance<Sphere> >
        Orthtree;
typedef Orthtree::Key Key;
typedef Orthtree::Element Element;
typedef Orthtree::iterator iterator;
typedef Orthtree::const_iterator const_iterator;

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
         << programName << " [mesh.vtu]" << "\n";
   exit(1);
}

//! Return the minimum of two numbers.
struct Minimum {
   Number
   operator()(const Number x, const Number y) const {
      return std::min(x, y);
   }
};

//! Return the maximum of two numbers.
struct Maximum {
   Number
   operator()(const Number x, const Number y) const {
      return std::max(x, y);
   }
};

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

   int numberOfSteps = 50;
   if (parser.getOption("steps", &numberOfSteps)) {
      if (numberOfSteps < 1) {
         std::cerr << "Bad number of steps = " << numberOfSteps << ".\n";
         exitOnError();
      }
   }

   // There should be no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // The interface.
   Sphere sphere;
   sphere.center = Point(-0.5);
   sphere.radius = 0.01;
   Point centerOffset;
   for (int d = 0; d != Orthtree::Dimension; ++d) {
      centerOffset[d] = 1.5 / (d + 1);
   }
   centerOffset /= numberOfSteps;
   const Number radiusOffset = 0.25 / numberOfSteps;
   Distance<Sphere> distance(sphere);

   // Construct the orthtree.
   Refine refine;
   Coarsen coarsen;
   Orthtree orthtree(Point(-1.), Point(2.), distance, distance, refine,
                     coarsen, distance);
   // Insert a single leaf.
   orthtree.insert(Key());

   std::cout << "Tracking the interface...\n";

   ads::Timer timer;
   timer.tic();

   int countDistance = 0, countCoarsen = 0, countRefine = 0;
   for (int step = 0; step != numberOfSteps; ++step) {
      sphere.center += centerOffset;
      sphere.radius += radiusOffset;

      // Update the distance.
      //std::cerr << "apply\n";
      orthtree.apply();
      countDistance += orthtree.size();
      //std::cerr << "After applying, size = " << orthtree.size() << "\n";

      // Coarsen.
      //std::cerr << "coarsen\n";
      countCoarsen += orthtree.coarsen();
      //std::cerr << "After coarsening, size = " << orthtree.size() << "\n";

      // Refine.
      //std::cerr << "refine\n";
      countRefine += orthtree.refine();
      //std::cerr << "After refining, size = " << orthtree.size() << "\n";

#ifdef DEBUG_geom
      if (AutomaticBalancing) {
         assert(orthtree.isBalanced());
      }
#endif

      if (! baseName.empty()) {
         std::string name = baseName;
         std::string extension;
         ads::makeZeroPaddedExtension(step, numberOfSteps - 1, &extension);
         name += extension;
         name += ".vtu";
         std::ofstream out(name.c_str());
         geom::printVtkUnstructuredGrid(out, orthtree);
      }
   }

   double elapsedTime = timer.toc();

   std::cout << "Done.  Time = " << elapsedTime << " seconds.\n"
             << "The orthtree has " << orthtree.size() << " nodes.\n"
             << "Average size = " << double(countDistance) / numberOfSteps
             << ".\n"
             << "Performed " << countDistance << " distance calculations.\n"
             << "Performed " << countCoarsen << " coarsening operations.\n"
             << "Performed " << countRefine << " refinement operations.\n";

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

   return 0;
}
