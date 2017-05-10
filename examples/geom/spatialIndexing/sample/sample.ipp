// -*- C++ -*-

/*!
  \file examples/geom/spatialIndexing/sample.ipp
  \brief Sample an function.
*/

#ifndef __examples_geom_spatialIndexing_sample_ipp__
#error This is an implementation detail.
#endif

#include "geom/spatialIndexing/OrthtreeMap.h"
#include "geom/kernel/Point.h"

#include "stlib/ads/algorithm/statistics.h"
#include "stlib/ads/array/Array.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/utility/string.h"

#include "numerical/constants.h"

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
// Element.
//

//! The element type stores function values at its corners.
struct Element {
   enum {NumberOfOrthants = numerical::Exponentiation<2, Dimension>::Result};
   typedef Number VtkOutput;
   ads::FixedArray<NumberOfOrthants, Number> values;
};

//! Write the average value.
std::ostream&
operator<<(std::ostream& out, const Element& element) {
   return out << ads::computeMean(element.values.begin(),
                                  element.values.end());
}

//
// Functors.
//

//! The sampled function.
struct SampledFunction {
   Number
   operator()(const Point& x) const {
      const Number y = geom::computeMagnitude(x);
      return std::exp(- 2 * y) *
             std::cos(4 * numerical::Constants<Number>::Pi() * y);
   }
};

//! Copy some values from the parent.  Sample at the other points.
class Split {
private:

   SampledFunction _f;

public:

   void
   operator()(const Element& parent, const int orthant,
              const Point& lowerCorner, const Point& extents,
              Element* element) const {
      Point x;
      for (int n = 0; n != Element::NumberOfOrthants; ++n) {
         // If we can copy the value from the parent.
         if (n == orthant) {
            element->values[n] = parent.values[n];
         }
         // Otherwise evaluate the function.
         else {
            // Determine the location of the corner using the lower corner
            // and the extents.
            x = lowerCorner;
            int m = n;
            for (int i = 0; i != Dimension; ++i) {
               x[i] += (m % 2) * extents[i];
               m /= 2;
            }
            // Evaluate the function.
            element->values[n] = _f(x);
         }
      }
   }
};

//! Return true if the element should be refined.
class Refine {
public:
   //! The result type.
   typedef bool result_type;

private:
   Number _maximumAllowedVariation;

public:

   void
   set(const Number maximumAllowedVariation) {
      _maximumAllowedVariation = maximumAllowedVariation;
   }

   result_type
   operator()(const Element& element) const {
      Number minimum = element.values[0];
      Number maximum = element.values[0];
      for (int n = 1; n != Element::NumberOfOrthants; ++n) {
         if (element.values[n] < minimum) {
            minimum = element.values[n];
         }
         else if (element.values[n] > maximum) {
            maximum = element.values[n];
         }
      }
      return maximum - minimum > _maximumAllowedVariation;
   }
};

//! Evaluate the function at the corners.
class Evaluate {
private:

   SampledFunction _f;

public:

   void
   operator()(const Point& lowerCorner, const Point& extents,
              Element* element) const {
      Point x;
      for (int n = 0; n != Element::NumberOfOrthants; ++n) {
         // Determine the location of the corner using the lower corner
         // and the extents.
         x = lowerCorner;
         int m = n;
         for (int i = 0; i != Dimension; ++i) {
            x[i] += (m % 2) * extents[i];
            m /= 2;
         }
         // Evaluate the function.
         element->values[n] = _f(x);
      }
   }
};

//
// Orthtree types.
//

typedef geom::OrthtreeMap < Dimension, MaximumLevel, Element,
        AutomaticBalancing,
        Split,
        geom::MergeNull,
        Refine,
        ads::GeneratorConstant<bool>,
        Evaluate >

        Orthtree;
typedef Orthtree::Key Key;
//typedef Orthtree::Element Element;
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
         << programName << " variation [output]" << "\n";
   exit(1);
}

}

namespace geom {
//! Define the VTK output type.
template<>
struct ElementVtkOutput<Element> {
   //! The output type is a floating point number.
   typedef Number Type;
};
}


//! The main loop.
int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   if (!(parser.getNumberOfArguments() == 1 ||
         parser.getNumberOfArguments() == 2)) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   Number variation = 0;
   parser.getArgument(&variation);

   std::string outputName;
   if (parser.getNumberOfArguments() != 0) {
      outputName = parser.getArgument();
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   // There should be no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Construct the orthtree.
   Orthtree orthtree(Point(-1.), Point(2.));
   // Set the maximum allowed variation.
   orthtree.getRefine().set(variation);
   // Insert a single leaf.
   orthtree.insert(Key());
   // Evaluate the function on the top level node.
   orthtree.apply();
   // Split the top level node.  (Otherwise the variation will be zero due
   // to symmetry.)
   orthtree.split(orthtree.begin());

   ads::Timer timer;
   timer.tic();

   // Refine the sample the function.
   int count = orthtree.refine();

   double elapsedTime = timer.toc();

   std::cout << "Done.  Time = " << elapsedTime << " seconds.\n"
             << "The orthtree has " << orthtree.size() << " nodes.\n"
             << "Performed " << count << " splitting operations.\n";

   std::cout << "Writing the orthtree...\n";
   if (! outputName.empty()) {
      std::ofstream out(outputName.c_str());
      geom::printVtkUnstructuredGrid(out, orthtree);
   }
   std::cout << "Done.\n";

   return 0;
}
