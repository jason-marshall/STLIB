// -*- C++ -*-

#define STLIB_PERFORMANCE
#include "stlib/sfc/gatherRelevant.h"

#include "stlib/performance/PerformanceMpi.h"

#include <random>

int
main(int argc, char* argv[])
{
  std::size_t const Dimension = 3;
  typedef double Float;
  typedef std::array<Float, Dimension> Point;
  typedef stlib::geom::BBox<Float, Dimension> BBox;
  typedef stlib::sfc::Traits<Dimension, Float> Traits;
  typedef stlib::sfc::AdaptiveCells<Traits, void, true> AdaptiveCells;
  typedef AdaptiveCells::Grid Grid;
  using stlib::performance::record;

  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;
  int const commSize = stlib::mpi::commSize();
  int const commRank = stlib::mpi::commRank();

  // Generate a sequence of random points as the objects.
  std::size_t const numDistributedObjects = 1024;
  std::vector<Point> objects;
  // On process 0, fill the unit cube with random points.
  if (commRank == 0) {
    std::default_random_engine engine;
    std::uniform_real_distribution<Float> distribution(0, 1);
    objects.resize(numDistributedObjects);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j) {
        objects[i][j] = distribution(engine);
      }
    }
  }
  // Partition and distribute the query points.
  stlib::sfc::distribute<Float, Dimension>(&objects, comm);
  record("Number of local objects", objects.size());

  // Build local and global cells for the objects.
  AdaptiveCells localCells;
  AdaptiveCells distributedCells;
  {
    Grid const grid(BBox{{{0, 0, 0}}, {{1, 1, 1}}}, 0);
    stlib::sfc::Partition<Traits> codePartition(commSize);
    distributedCells = adaptiveCells(grid, &objects, 32, &localCells,
                                       &codePartition, comm);
  }

  // Declare that all the distributed cells are relevant.
  std::vector<std::size_t> relevantCells(distributedCells.size());
  for (std::size_t i = 0; i != relevantCells.size(); ++i) {
    relevantCells[i] = i;
  }
  
  // Gather the relevant objects with the ring pattern.
  {
    std::vector<Point> relevantObjects =
      stlib::sfc::gatherRelevantRing(objects, distributedCells, relevantCells,
                                     comm);
    record("Number of relevant objects with ring", relevantObjects.size());
    // Check the number of objects.
    assert(relevantObjects.size() == numDistributedObjects);
  }

  // Gather the relevant objects with the point-to-point pattern.
  {
    std::vector<Point> relevantObjects =
      stlib::sfc::gatherRelevantPointToPoint(objects, distributedCells,
                                             relevantCells, comm);
    record("Number of relevant objects with point-to-point",
           relevantObjects.size());
    // Check the number of objects.
    assert(relevantObjects.size() == numDistributedObjects);
  }

  stlib::performance::print();

  MPI_Finalize();
  return 0;
}
