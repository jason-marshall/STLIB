// -*- C++ -*-

#if !defined(__performance_numerical_interpolation_PolynomialInterpolationSampler_ipp__)
#error This file is an implementation detail of PolynomialInterpolationSampler.
#endif

#include "stlib/numerical/interpolation/PolynomialInterpolationUsingCoefficients.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <algorithm>
#include <iostream>
#include <vector>

using namespace stlib;

namespace {

//
// Global variables.
//

//! The program name.
static std::string programName;

//
// Local functions.
//

//! Exit with an error message.
void
exitOnError() {
   std::cerr
      << "Bad arguments.  Usage:\n"
      << programName << " order degree inputCells outputCells\n\n";
   exit(1);
}

template<std::size_t _Order>
void
sample(const std::size_t degree, const std::size_t numInputCells,
       const std::size_t numOutputCells) {
   const double idx = (Upper - Lower) / numInputCells;
   const double odx = (Upper - Lower) / numOutputCells;

   // Print the array of function arguments.
   for (std::size_t i = 0; i != numOutputCells; ++i) {
      std::cout << odx * i << ' ';
   }
   std::cout << '\n';

   // Sample the function.
   F f;
   DF df;
   DDF ddf;
   std::vector<double> g(numInputCells + 1);
   std::vector<double> dg(numInputCells + 1);
   std::vector<double> ddg(numInputCells + 1);
   for (std::size_t i = 0; i != g.size(); ++i) {
      const double x = idx * i;
      g[i] = f(x);
      dg[i] = df(x);
      ddg[i] = ddf(x);
   }

   // The interpolating functor.
   numerical::PolynomialInterpolationUsingCoefficients<double, _Order> 
      y(g.begin(), g.size(), Lower, Upper);
   if (degree == 0) {
      y.setGridValues(g.begin());
   }
   else if (degree == 1) {
      y.setGridValues(g.begin(), dg.begin());
   }
   else if (degree == 2) {
      y.setGridValues(g.begin(), dg.begin(), ddg.begin());
   }
   else {
      assert(false);
   }
   // Function.
   for (std::size_t i = 0; i != numOutputCells; ++i) {
      std::cout << y(odx * i) << ' ';
   }
   std::cout << '\n';
   // Difference.
   for (std::size_t i = 0; i != numOutputCells; ++i) {
      std::cout << y(odx * i) - f(odx * i) << ' ';
   }
   std::cout << '\n';
}

} // end namespace

//! The main loop.
int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();
   
   // There should be four arguments.
   if (parser.getNumberOfArguments() != 4) {
      std::cerr << "Bad arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // The interpolation order.
   std::size_t order;
   if(! parser.getArgument(&order)) {
      std::cerr << "Could not parse the argument for the interpolation "\
         "order.\n";
      exitOnError();
   }

   // The derivative degree.
   std::size_t degree;
   if (! parser.getArgument(&degree)) {
      std::cerr << "Could not parse the argument for the derivative degree.\n";
      exitOnError();
   }
   if (degree > 2) {
      std::cerr << "The derivative degree must be no greater than 2.\n";
      exitOnError();
   }

   // The number of input cells.
   std::size_t numInputCells;
   if (! parser.getArgument(&numInputCells)) {
      std::cerr << "Could not parse the argument for the number of input "\
         "cells.\n";
      exitOnError();
   }
   if (numInputCells < 1) {
      std::cerr << "The number of input cells must be at least 1.\n";
      exitOnError();
   }

   // The number of output cells.
   std::size_t numOutputCells;
   if (! parser.getArgument(&numOutputCells)) {
      std::cerr << "Could not parse the argument for the number of output "\
         "cells.\n";
      exitOnError();
   }
   if (numOutputCells < 1) {
      std::cerr << "The number of output cells must be at least 1.\n";
      exitOnError();
   }

   // There should be no options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Print the function arguments and values.
   if (order == 1) {
      sample<1>(degree, numInputCells, numOutputCells);
   }
   else if (order == 3) {
      sample<3>(degree, numInputCells, numOutputCells);
   }
   else if (order == 5) {
      sample<5>(degree, numInputCells, numOutputCells);
   }
   else {
      assert(false);
   }

   return 0;
}
