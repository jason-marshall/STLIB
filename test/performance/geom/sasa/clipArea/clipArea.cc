/* -*- C++ -*- */

#include "clipAreaGpu.h"
// Get the number of threads per block.
#include "clipAreaKernel.h"
#include "clipAreaCpu.h"

#include "stlib/ads/timer.h"
#include "stlib/ads/functor/Dereference.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/geom/orq/CellArrayStatic.h"

#include <iostream>

#include <cassert>
#include <cmath>

using namespace stlib;

std::string programName;

void
exitOnError();

void
generateRandomPointOnUnitSphere(float3* p);

void
initializeRandomBalls(std::vector<float3>* balls);

void
calculateClippingBalls(const std::vector<float3>& centers,
                       std::vector<Ball>* clippingBalls);

int
main(int argc, char* argv[])
{
  typedef ads::Dereference<std::vector<std::array<float, 3> >::iterator>
  Location;
  typedef geom::CellArrayStatic<3, Location> Orq;

  // Parse the command line arguments and options.
  ads::ParseOptionsArguments parser(argc, argv);
  // There should be no arguments.
  if (! parser.areArgumentsEmpty()) {
    exitOnError();
  }
  // Get the number of balls that comprise the object. The default is 256.
  std::size_t numberOfBalls = 256;
  parser.getOption('n', &numberOfBalls);
  // Get the number of points for the reference mesh. The default is 1024.
  std::size_t numberOfMeshPoints = 1024;
  parser.getOption('m', &numberOfMeshPoints);
  if (numberOfMeshPoints % ThreadsPerBlock != 0) {
    std::cerr << "The number of mesh points must be a multiple of the " \
              "number of threads per block " << ThreadsPerBlock << '\n';
    exitOnError();
  }
  // There should be no more options.
  if (! parser.areOptionsEmpty()) {
    std::cerr << "Error: unmatched options:\n";
    parser.printOptions(std::cerr);
    exitOnError();
  }

  // The reference mesh is a set of random points on the unit sphere.
  std::vector<float3> referenceMesh(numberOfMeshPoints);
  for (std::size_t i = 0; i != referenceMesh.size(); ++i) {
    generateRandomPointOnUnitSphere(&referenceMesh[i]);
  }
  // The balls that comprise the object.
  std::vector<float3> centers(numberOfBalls);
  initializeRandomBalls(&centers);

  // Determine the clipping balls.
  std::vector<Orq::Point> points(centers.size());
  for (std::size_t i = 0; i != points.size(); ++i) {
    points[i][0] = centers[i].x;
    points[i][1] = centers[i].y;
    points[i][2] = centers[i].z;
  }
  Orq orq(points.begin(), points.end());
  std::vector<Orq::Record> records;
  std::vector<std::size_t> clippingIndices;
  std::vector<std::size_t> clippingSizes;
  Orq::BBox window;
  for (std::size_t i = 0; i != points.size(); ++i) {
    window.lower = points[i];
    window.lower -= 2.;
    window.upper = points[i];
    window.upper += 2.;
    records.clear();
    orq.computeWindowQuery(std::back_inserter(records), window);
    std::size_t n = 0;
    for (std::size_t j = 0; j != records.size(); ++j) {
      const std::size_t index = records[j] - points.begin();
      if (index == i) {
        continue;
      }
      if (squaredDistance(points[index], points[i]) < 4) {
        clippingIndices.push_back(index);
        ++n;
      }
    }
    clippingSizes.push_back(n);
  }

  std::cout << "Number of points in the mesh = " << referenceMesh.size()
            << ".\n"
            << "Total number of points = "
            << centers.size() * referenceMesh.size() << ".\n"
            << "Maximum number of clipping balls for a mesh = "
            << *std::max_element(clippingSizes.begin(), clippingSizes.end())
            << '\n';

  //
  // Calculate the area on the GPU.
  //
  // Warm-up.
  calculateAreaGpu(referenceMesh, centers, clippingSizes, clippingIndices);
  ads::Timer timer;
  timer.tic();
  std::size_t activeCount = calculateAreaGpu(referenceMesh, centers,
                            clippingSizes, clippingIndices);
  double elapsedTime = timer.toc();
  std::cout << "Time for area calculation on the GPU = " << elapsedTime
            << ".\n"
            << "Number of active points = " << activeCount << ".\n";

  //
  // Calculate the area on the CPU.
  //
  timer.tic();
  activeCount = calculateAreaCpu(referenceMesh, centers, clippingSizes,
                                 clippingIndices);
  elapsedTime = timer.toc();
  std::cout << "Time for area calculation on the CPU = " << elapsedTime
            << ".\n"
            << "Number of active points = " << activeCount << ".\n";

  return 0;
}

void
exitOnError()
{
  std::cerr << "Usage:\n"
            << programName << " [-n numberOfBalls]\n";
  exit(1);
}

void
generateRandomPointOnUnitSphere(float3* p)
{
  const float InverseRandMax = 1. / RAND_MAX;
  float scaling;
  do {
    // Start with a point in the cube [-1..1]^3.
    p->x = 2 * (rand() * InverseRandMax) - 1;
    p->y = 2 * (rand() * InverseRandMax) - 1;
    p->z = 2 * (rand() * InverseRandMax) - 1;
    // Scale the point to lie on the unit sphere.
    scaling = 1. / sqrt((p->x * p->x + p->y * p->y + p->z * p->z));
    p->x *= scaling;
    p->y *= scaling;
    p->z *= scaling;
    // Accept if the point was inside the sphere before scaling.
  }
  while (scaling <= 1);
}

void
initializeRandomBalls(std::vector<float3>* centers)
{
  const float InverseRandMax = 1. / RAND_MAX;
  assert(! centers->empty());
  // The volume of the unit ball.
  const float volume = centers->size() * 4. / 3. * 3.14;
  const float length = std::pow(volume, float(1. / 3.));
  for (std::size_t i = 0; i != centers->size(); ++i) {
    (*centers)[i].x = length * rand() * InverseRandMax;
    (*centers)[i].y = length * rand() * InverseRandMax;
    (*centers)[i].z = length * rand() * InverseRandMax;
  }
}

void
calculateClippingBalls(const std::vector<float3>& centers,
                       std::vector<Ball>* clippingBalls)
{
  clippingBalls->resize(centers.size());
  for (std::size_t i = 0; i != clippingBalls->size(); ++i) {
    (*clippingBalls)[i].center = centers[i];
    (*clippingBalls)[i].squaredRadius =
      1. - 10 * std::numeric_limits<float>::epsilon();
  }
}
