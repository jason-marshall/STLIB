// -*- C++ -*-

#include "stlib/ads/array/Array.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"

#include <iostream>

using namespace stlib;

namespace
{

// Global variables.

//! The program name.
std::string programName;

// Local functions.

//! Exit with an error message.
void
exitOnError()
{
  std::cerr
      << "Bad arguments.  Usage:\n"
      << programName << " size iterations\n"
      << "- size: The size of the vector.\n"
      << "- iterations: The number of iterations to run.\n";
  exit(1);
}

}


int
main(int argc, char* argv[])
{
  ads::ParseOptionsArguments parser(argc, argv);

  programName = parser.getProgramName();

  // There should be no options.
  if (! parser.areOptionsEmpty()) {
    std::cerr << "Error.  Unmatched options:\n";
    parser.printOptions(std::cerr);
    exitOnError();
  }

  // If they did not specify the size and number of iterations.
  if (parser.getNumberOfArguments() != 2) {
    std::cerr << "Bad number of required arguments.\n"
              << "You gave the arguments:\n";
    parser.printArguments(std::cerr);
    exitOnError();
  }

  int size = -1;
  parser.getArgument(&size);
  if (size < 1) {
    std::cerr << "Bad vector size.\n";
    exitOnError();
  }

  int numberOfIterations = -1;
  parser.getArgument(&numberOfIterations);
  if (numberOfIterations < 0) {
    std::cerr << "Bad number of iterations.\n";
    exitOnError();
  }

  // There should be no more arguments.
  assert(parser.areArgumentsEmpty());

  // Make the vectors.
  ads::Array<1> x(size, 1.0), y(size, 1.0);

  volatile double result = 0;
  ads::Timer timer;

  // Warm up.
  double sum;
  for (int n = 0; n != 100; ++n) {
    sum = 0;
    // There is no cost for calling size().
    for (int i = 0; i != x.size(); ++i) {
      sum += x[i] * y[i];
    }
    result += sum;
  }

  // Time the dot product.
  timer.tic();
  for (int n = 0; n != numberOfIterations; ++n) {
    sum = 0;
    // There is no cost for calling size().
    for (int i = 0; i != x.size(); ++i) {
      sum += x[i] * y[i];
    }
    result += sum;
  }
  double elapsedTime = timer.toc();

  std::cout
      << "Meaningless result = " << result << "\n"
      << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
      << "size = " << size << "\n"
      << "iterations = " << numberOfIterations << "\n"
      << "elapsed time = " << elapsedTime << "\n"
      << "Time per dot product = " << elapsedTime / numberOfIterations * 1e9
      << " nanoseconds.\n";

  return 0;
}
