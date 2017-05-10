/* -*- C++ -*- */

#include "stlib/levelSet/vanDerWaalsCuda.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer.h"
#include "stlib/cuda/check.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>


using namespace stlib;

//
// Constants and types.
//
const std::size_t Dimension = 3;
typedef float Number;

std::string programName;
Number targetGridSpacing = 0.1;
bool verbose = false;


void
exitOnError()
{
  std::cerr
      << "Bad arguments.  Usage:\n"
      << programName << " [-v] [-s=gridSpacing] input.xyzr \n"
      << "Distance is measured in Angstroms.\n"
      << "In verbose mode (-v), timing and progress information will be printed.\n"
      << "The default grid spacing is 0.1 Angstroms.\n"
      << "The input file is a sequence of x, y, z coordinates and radius.\n"
      << "\nExiting...\n";
  exit(1);
}


void
parseNameAndOptions(ads::ParseOptionsArguments* parser)
{
  programName = parser->getProgramName();

  // Verbose
  verbose = parser->getOption("v");

  // Grid spacing.
  parser->getOption("s", &targetGridSpacing);
  if (targetGridSpacing <= 0) {
    std::cerr << "The target grid spacing must be positive.";
    exitOnError();
  }

  // Check that we parsed all of the options.
  if (! parser->areOptionsEmpty()) {
    std::cerr << "Error.  Unmatched options:\n";
    parser->printOptions(std::cerr);
    exitOnError();
  }

  if (parser->getNumberOfArguments() != 1) {
    std::cerr << "Bad number of required arguments.\n"
              << "You gave the arguments:\n";
    parser->printArguments(std::cerr);
    exitOnError();
  }
}


void
readBalls(ads::ParseOptionsArguments* parser,
          std::vector<geom::Ball<Number, Dimension> >* balls)
{
  if (verbose) {
    std::cout << "Reading the xyzr file..." << std::endl;
  }
  ads::Timer timer;
  timer.tic();
  {
    std::ifstream file(parser->getArgument().c_str());
    if (! file.good()) {
      std::cerr << "Could not open the input file.";
      exitOnError();
    }
    readElements(file, balls);
    if (balls->empty()) {
      std::cerr << "The set of balls is empty.\n";
      exitOnError();
    }
  }
  if (verbose) {
    std::cout << "  Time = " << timer.toc() << std::endl;
  }
}


int
main(int argc, char* argv[])
{
  // Parse the program name and options.
  ads::ParseOptionsArguments parser(argc, argv);
  parseNameAndOptions(&parser);

  // Read the set of balls.
  std::vector<geom::Ball<Number, Dimension> > balls;
  readBalls(&parser, &balls);

  //
  // Calculate the volume using the GPU.
  //
  if (verbose) {
    std::cout << "Calculate the volume..." << std::endl;
  }
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));

  const Number volume =
    levelSet::vanDerWaalsCuda(balls, targetGridSpacing);

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsedTime;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
  if (verbose) {
    std::cout << "Elapsed time for CUDA kernel = " << elapsedTime << " ms.\n";
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  if (verbose) {
    std::cout << "Volume:\n";
  }
  std::cout << volume << '\n';

  return 0;
}
