// -*- C++ -*-

#include "stlib/sfc/UniformCellsMpi.h"

#include "stlib/mpi/statistics.h"

#include <random>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

void
partitionCoarsenCube()
{
  typedef sfc::Traits<> Traits;
  std::size_t const Dimension = Traits::Dimension;
  static_assert(Dimension == 3, "Must be 3-D.");
  typedef Traits::Float Float;
  typedef Traits::Point Point;
  typedef Traits::BBox BBox;
  typedef sfc::UniformCells<Traits, void, true> UniformCells;

  BBox const Domain = {{{0, 0, 0}}, {{1, 1, 1}}};
  int const commSize = mpi::commSize(MPI_COMM_WORLD);
  int const commRank = mpi::commRank(MPI_COMM_WORLD);

  {
    // On all processes, fill the unit cube with uniformly-distributed random 
    // points.
    std::default_random_engine generator;
    std::uniform_real_distribution<Float> distribution(0, 1);
    std::vector<Point> points(1 << 10);
    for (std::size_t i = 0; i != points.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j) {
        points[i][j] = distribution(generator);
      }
    }
    std::size_t const NumGlobalPoints = mpi::reduce(points.size(), MPI_SUM);

    UniformCells cells(Domain, 0);
    cells.buildCells(&points);
    sfc::Partition<Traits> codePartition(commSize);
    sfc::partitionCoarsen(&cells, &points, &codePartition, MPI_COMM_WORLD);

    std::size_t const n = mpi::reduce(points.size(), MPI_SUM);
    if (commRank == 0) {
      assert(n == NumGlobalPoints);
    }

    if (commRank == 0) {
      std::cout << "\npartitionCoarsenCube():\n"
                << "commSize = " << commSize << '\n'
                << "Partitioned cells:\n"
                << "  lowerCorner() = " << cells.lowerCorner() << '\n'
                << "  lengths() = " << cells.lengths() << '\n'
                << "  numLevels() = " << cells.numLevels() << '\n'
                << "  size() = " << cells.size() << '\n';
    }
    mpi::printStatistics(std::cout, "Number of points", points.size(),
                         MPI_COMM_WORLD);
  }

  {
    // On the root process, fill the unit cube with uniformly-distributed 
    // random points.
    std::default_random_engine generator;
    std::uniform_real_distribution<Float> distribution(0, 1);
    std::vector<Point> points;
    if (commRank == 0) {
      points.resize(1 << 10);
      for (std::size_t i = 0; i != points.size(); ++i) {
        for (std::size_t j = 0; j != Dimension; ++j) {
          points[i][j] = distribution(generator);
        }
      }
    }
    std::size_t const NumGlobalPoints = mpi::reduce(points.size(), MPI_SUM);

    UniformCells cells(Domain, 0);
    cells.buildCells(&points);
    sfc::Partition<Traits> codePartition(commSize);
    sfc::partitionCoarsen(&cells, &points, &codePartition, MPI_COMM_WORLD);

    std::size_t const n = mpi::reduce(points.size(), MPI_SUM);
    if (commRank == 0) {
      assert(n == NumGlobalPoints);
    }

    if (commRank == 0) {
      std::cout << "\npartitionCoarsenCube():\n"
                << "commSize = " << commSize << '\n'
                << "Partitioned cells:\n"
                << "  lowerCorner() = " << cells.lowerCorner() << '\n'
                << "  lengths() = " << cells.lengths() << '\n'
                << "  numLevels() = " << cells.numLevels() << '\n'
                << "  size() = " << cells.size() << '\n';
    }
    mpi::printStatistics(std::cout, "Number of points", points.size(),
                         MPI_COMM_WORLD);
  }
}


void
partitionCoarsenCubeBottom()
{
  typedef sfc::Traits<> Traits;
  std::size_t const Dimension = Traits::Dimension;
  static_assert(Dimension == 3, "Must be 3-D.");
  typedef Traits::Float Float;
  typedef Traits::Point Point;
  typedef Traits::BBox BBox;
  typedef sfc::UniformCells<Traits, void, true> UniformCells;

  BBox const Domain = {{{0, 0, 0}}, {{1, 1, 1}}};
  int const commSize = mpi::commSize(MPI_COMM_WORLD);
  int const commRank = mpi::commRank(MPI_COMM_WORLD);

  // Uniformly-distributed random points on the bottom face.
  std::default_random_engine generator;
  std::uniform_real_distribution<Float> distribution(0, 1);
  std::vector<Point> points(1 << 10);
  for (std::size_t i = 0; i != points.size(); ++i) {
    for (std::size_t j = 0; j != Dimension - 1; ++j) {
      points[i][j] = distribution(generator);
    }
    points[i][Dimension - 1] = 0;
  }
  std::size_t const NumGlobalPoints = mpi::reduce(points.size(), MPI_SUM);

  UniformCells cells(Domain, 0);
  cells.buildCells(&points);
  sfc::Partition<Traits> codePartition(commSize);
  sfc::partitionCoarsen(&cells, &points, &codePartition, MPI_COMM_WORLD);

  std::size_t const n = mpi::reduce(points.size(), MPI_SUM);
  if (commRank == 0) {
    assert(n == NumGlobalPoints);
  }

  if (commRank == 0) {
    std::cout << "\npartitionCoarsenCubeBottom():\n"
              << "commSize = " << commSize << '\n'
              << "Partitioned cells:\n"
              << "  lowerCorner() = " << cells.lowerCorner() << '\n'
              << "  lengths() = " << cells.lengths() << '\n'
              << "  numLevels() = " << cells.numLevels() << '\n'
              << "  size() = " << cells.size() << '\n';
  }
  mpi::printStatistics(std::cout, "Number of points", points.size(),
                       MPI_COMM_WORLD);
}


int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  int commSize;
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);
  assert(commSize >= 2);
  int commRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

  // Send and receive.
  {
    const std::size_t NumLevels = 20;
    typedef sfc::Traits<1> Traits;
    typedef geom::BBox<Traits::Float, Traits::Dimension> Cell;
    typedef sfc::UniformCells<Traits, Cell, false> UniformCells;
    typedef UniformCells::Float Float;
    typedef UniformCells::Point Point;
    const Point LowerCorner = Point{{0}};
    const Point Lengths = Point{{1}};

    // Uniformly-spaced points.
    std::vector<Point> objects(512);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      objects[i][0] = Float(i) / objects.size();
    }

    UniformCells cells(LowerCorner, Lengths, NumLevels);
    cells.buildCells(&objects);
    if (commRank == 0) {
      send(cells, 1, 0, MPI_COMM_WORLD);
    }
    else if (commRank == 1) {
      UniformCells copy(LowerCorner, Lengths, NumLevels);
      recv(&copy, 0, 0, MPI_COMM_WORLD);
      assert(copy == cells);
    }
  }

  // Merge. Same objects.
  {
    const std::size_t NumLevels = 20;
    typedef sfc::Traits<1> Traits;
    typedef sfc::UniformCells<Traits, void, true> UniformCells;
    typedef UniformCells::Float Float;
    typedef UniformCells::Point Point;
    const Point LowerCorner = Point{{0}};
    const Point Lengths = Point{{1}};

    // Uniformly-spaced points.
    std::vector<Point> objects(512);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      objects[i][0] = Float(i) / objects.size();
    }

    UniformCells cells(LowerCorner, Lengths, NumLevels);
    cells.buildCells(&objects);

    // Merge.
    {
      UniformCells merged(LowerCorner, Lengths, NumLevels);
      merge(cells, &merged, MPI_COMM_WORLD);
      if (commRank == 0) {
        assert(merged.size() == cells.size());
        for (std::size_t i = 0; i != merged.size(); ++i) {
          // There must be one object from each process.
          assert(merged.delimiter(i + 1) - merged.delimiter(i) ==
                 std::size_t(commSize));
        }
      }
      else {
        assert(merged.size() == 0);
      }
    }

    // Merge and coarsen.
    {
      // No coarsening is necessary.
      UniformCells merged(LowerCorner, Lengths, NumLevels);
      mergeCoarsen(cells, cells.size(), &merged, MPI_COMM_WORLD);
      if (commRank == 0) {
        assert(merged.size() == cells.size());
        for (std::size_t i = 0; i != merged.size(); ++i) {
          // There must be one object from each process.
          assert(merged.delimiter(i + 1) - merged.delimiter(i) ==
                 std::size_t(commSize));
        }
      }
      else {
        assert(merged.size() == 0);
      }
    }
    {
      // Decrease cells by a factor of two.
      UniformCells merged(LowerCorner, Lengths, NumLevels);
      mergeCoarsen(cells, cells.size() / 2, &merged, MPI_COMM_WORLD);
      if (commRank == 0) {
        assert(merged.size() == cells.size() / 2);
        for (std::size_t i = 0; i != merged.size(); ++i) {
          // There must be one object from each process.
          assert(merged.delimiter(i + 1) - merged.delimiter(i) ==
                 2 * std::size_t(commSize));
        }
      }
      else {
        assert(merged.size() == 0);
      }
    }
    {
      // Decrease to a single cell.
      UniformCells merged(LowerCorner, Lengths, NumLevels);
      mergeCoarsen(cells, 1, &merged, MPI_COMM_WORLD);
      if (commRank == 0) {
        assert(merged.size() == 1);
        assert(merged.delimiter(1) - merged.delimiter(0) ==
               objects.size() * std::size_t(commSize));
      }
      else {
        assert(merged.size() == 0);
      }
    }

    // mergeCoarsenDisjoint()
    // Note that we are violating the assumption that the distributed cells 
    // are roughly disjoint.
    {
      // No coarsening is necessary.
      UniformCells merged(LowerCorner, Lengths, NumLevels);
      mergeCoarsenDisjoint(cells, commSize * cells.size(), &merged,
                           MPI_COMM_WORLD);
      if (commRank == 0) {
        assert(merged.size() == cells.size());
        for (std::size_t i = 0; i != merged.size(); ++i) {
          // There must be one object from each process.
          assert(merged.delimiter(i + 1) - merged.delimiter(i) ==
                 std::size_t(commSize));
        }
      }
      else {
        assert(merged.size() == 0);
      }
    }
    {
      // Decrease the number of cells.
      UniformCells merged(LowerCorner, Lengths, NumLevels);
      mergeCoarsenDisjoint(cells, cells.size(), &merged, MPI_COMM_WORLD);
      if (commRank == 0) {
        std::size_t const multiplicity = cells.size() / merged.size();
        for (std::size_t i = 0; i != merged.size(); ++i) {
          // There must be one object from each process.
          assert(merged.delimiter(i + 1) - merged.delimiter(i) ==
                 multiplicity * commSize);
        }
      }
      else {
        assert(merged.size() == 0);
      }
    }
    {
      // Decrease to a single cell.
      UniformCells merged(LowerCorner, Lengths, NumLevels);
      mergeCoarsenDisjoint(cells, 1, &merged, MPI_COMM_WORLD);
      if (commRank == 0) {
        assert(merged.size() == 1);
        assert(merged.delimiter(1) - merged.delimiter(0) ==
               objects.size() * std::size_t(commSize));
      }
      else {
        assert(merged.size() == 0);
      }
    }
  }

  // Merge. Different objects.
  {
    const std::size_t NumLevels = 20;
    typedef sfc::Traits<1> Traits;
    typedef sfc::UniformCells<Traits, void, true> UniformCells;
    typedef UniformCells::Float Float;
    typedef UniformCells::Point Point;
    const Point LowerCorner = Point{{0}};
    const Point Lengths = Point{{Float(commSize)}};

    // Uniformly-spaced points.
    std::vector<Point> objects(512);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      objects[i][0] = commRank + Float(i) / objects.size();
    }

    UniformCells cells(LowerCorner, Lengths, NumLevels);
    cells.buildCells(&objects);

    // Merge.
    {
      UniformCells merged(LowerCorner, Lengths, NumLevels);
      merge(cells, &merged, MPI_COMM_WORLD);
      if (commRank == 0) {
        assert(merged.size() == commSize * cells.size());
        for (std::size_t i = 0; i != merged.size(); ++i) {
          assert(merged.delimiter(i + 1) - merged.delimiter(i) == 1);
        }
      }
      else {
        assert(merged.size() == 0);
      }
    }

    // Merge and coarsen
    {
      UniformCells merged(LowerCorner, Lengths, NumLevels);
      mergeCoarsen(cells, cells.size(), &merged, MPI_COMM_WORLD);
      if (commRank == 0) {
        assert(merged.size() <= cells.size());
        assert(merged.delimiter(merged.size()) -
               merged.delimiter(0) == commSize * objects.size());
      }
      else {
        assert(merged.size() == 0);
      }
    }

    // mergeCoarsenDisjoint()
    {
      UniformCells merged(LowerCorner, Lengths, NumLevels);
      mergeCoarsenDisjoint(cells, cells.size(), &merged, MPI_COMM_WORLD);
      if (commRank == 0) {
        assert(merged.size() <= cells.size());
        assert(merged.delimiter(merged.size()) -
               merged.delimiter(0) == commSize * objects.size());
      }
      else {
        assert(merged.size() == 0);
      }
    }
  }

  // partitionCoarsen();
  partitionCoarsenCube();
  partitionCoarsenCubeBottom();

  MPI_Finalize();
  return 0;
}
