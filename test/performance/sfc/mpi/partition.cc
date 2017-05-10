// -*- C++ -*-

#define STLIB_PERFORMANCE

#include "stlib/sfc/UniformCellsMpi.h"
#include "stlib/sfc/AdaptiveCellsMpi.h"

#include "stlib/performance/PerformanceMpi.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/mpi/BBox.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

using namespace stlib;

// The program name.
std::string programName;

const std::size_t DefaultNumObjects = 1024;

// Exit with a usage message.
void
helpMessage()
{
  if (stlib::mpi::commRank() == 0) {
    std::cout
      << "Usage:\n"
      << programName
      << " [-c=C] [-o=O] [-h]\n"
      << "-c: The kind of cell. 'u' for uniform 'm' for multi-level.\n"
      << "-o: The number of objects. The default is " << DefaultNumObjects
      << ".\n";
  }
  MPI_Finalize();
  exit(0);
}

int
main(int argc, char* argv[])
{
  const std::size_t Dimension = 3;
  typedef sfc::Traits<Dimension> Traits;
  typedef sfc::UniformCells<Traits, void, true> UniformCells;
  typedef Traits::Point Point;
  typedef Traits::Float Float;
  typedef Traits::BBox BBox;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;
  using stlib::performance::Scope;
  using stlib::performance::start;
  using stlib::performance::stop;
  using stlib::performance::record;

  // MPI initialization.
  MPI_Init(&argc, &argv);
  int commSize;
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  int commRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

  // Parse the options.
  ads::ParseOptionsArguments parser(argc, argv);
  programName = parser.getProgramName();
  if (parser.getOption('h')) {
    helpMessage();
  }
  bool useMultiLevel = true;
  {
    char cellType;
    if (parser.getOption('c', &cellType)) {
      if (cellType == 'u') {
        useMultiLevel = false;
      }
      else if (cellType == 'm') {
        useMultiLevel = true;
      }
      else {
        helpMessage();
      }
    }
  }
  std::size_t numObjects = DefaultNumObjects;
  parser.getOption('o', &numObjects);

  // Uniformly-distributed random points. We scale the first coordinate so
  // that the points for each process lie in thin sheets.
  std::vector<Point> objects(numObjects);
  {
    ContinuousUniformGenerator::DiscreteUniformGenerator generator;
    ContinuousUniformGenerator random(&generator);
    const Float offset = Float(commRank) / commSize;
    const Float scaling = Float(1) / commSize;
    for (std::size_t i = 0; i != objects.size(); ++i) {
      objects[i][0] = offset + scaling * random();
      for (std::size_t j = 1; j != Dimension; ++j) {
        objects[i][j] = random();
      }
    }
  }

  record("MPI processes", commSize);

  if (useMultiLevel) {
    Scope _("Distribute objects with multi-level cells");
    // Because of the thin sheets, we need to adjust the accuracy goal.
    stlib::sfc::distribute<Float, Dimension>
      (&objects, MPI_COMM_WORLD, 0.01 / std::sqrt(double(commSize)));
  }
  else {
    Scope _("Distribute objects with uniform cells");
    start("Build the local cells");

    // Determine the bounding box for the global set of objects.
    BBox const domain =
      mpi::allReduce(geom::specificBBox<BBox>(objects.begin(), objects.end()),
                     MPI_COMM_WORLD);

    // Build the local cells. Use all the available levels of refinement.
    UniformCells cells(domain, 0);
    cells.buildCells(&objects);

    stop();
    start("Partition the cells");
    
    // Note that there is a convenience function for performing the
    // partitioning and distributing together. Here I call them separately so
    // that I can time them.

    // Determine a fair partitioning of the cells.
    sfc::Partition<Traits> partition(commSize);
    sfc::_partitionCoarsen(&cells, &partition, MPI_COMM_WORLD);

    stop();
    start("Distribute the cells and objects");

    // Distribute the cells and objects.
    sfc::distribute(&cells, &objects, partition, MPI_COMM_WORLD);

    stop();
    record("Number of objects", objects.size());
    record("Number of levels", cells.numLevels());
    record("Number of cells", cells.size());
  }

  stlib::performance::print();

  MPI_Finalize();
  return 0;
}
