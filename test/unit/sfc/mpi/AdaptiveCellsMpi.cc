// -*- C++ -*-

#include "stlib/sfc/AdaptiveCellsMpi.h"


int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int const commSize = stlib::mpi::commSize();
  int const commRank = stlib::mpi::commRank();

  typedef stlib::sfc::Traits<1>::Code Code;
  typedef std::pair<Code, std::size_t> Pair;

  // buildAdaptiveBlocksFromDistributedObjects()
  // 1-D
  {
    typedef stlib::sfc::Traits<1> Traits;
    typedef Traits::Point Point;
    typedef stlib::sfc::AdaptiveCells<Traits, void, true> AdaptiveCells;
    typedef AdaptiveCells::Grid Grid;
    Code const Guard = Traits::GuardCode;
    
    // 0 levels
    {
      Grid const grid(Point{{0}}, Point{{1}}, 0);
      AdaptiveCells check(grid);
      {
        std::vector<Point> objects;
        AdaptiveCells output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
      }
      {
        std::vector<Point> objects = {{{0}}};
        AdaptiveCells output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{0, commSize}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
      }
    }

    // 1 level
    {
      Grid const grid(Point{{0}}, Point{{1}}, 1);
      AdaptiveCells check(grid);
      {
        std::vector<Point> objects;
        AdaptiveCells output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
      }
      {
        std::vector<Point> objects;
        if (stlib::mpi::commRank() == 0) {
          objects.push_back(Point{{0}});
        }
        AdaptiveCells output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{0, 1}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }

        if (stlib::mpi::commRank() == 1 % stlib::mpi::commSize()) {
          objects.push_back(Point{{0.5}});
        }

        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          if (commSize >= 2) {
            check.buildCells(std::vector<Pair>{{0, 2}, {Guard, 0}});
          }
          else {
            check.buildCells(std::vector<Pair>{{1, 1}, {3, 1}, {Guard, 0}});
          }
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
        
        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 2, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{0, 2}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }

        if (stlib::mpi::commRank() == 0) {
          objects.push_back(Point{{0}});
        }

        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{1, 2}, {3, 1}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }

        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 2, MPI_COMM_WORLD);
        if (commRank == 0) {
          if (commSize >= 2) {
            check.buildCells(std::vector<Pair>{{0, 3}, {Guard, 0}});
          }
          else {
            check.buildCells(std::vector<Pair>{{1, 2}, {3, 1}, {Guard, 0}});
          }
          assert(output == check);
        }
        else {
          assert(output.empty());
        }

        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 3, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{0, 3}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
      }
    }

    // 2 levels
    {
      Grid const grid(Point{{0}}, Point{{1}}, 2);
      AdaptiveCells check(grid);
      {
        std::vector<Point> objects;
        AdaptiveCells output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
      }
      {
        std::vector<Point> objects;
        if (stlib::mpi::commRank() == 0) {
          objects.push_back(Point{{0}});
        }
        AdaptiveCells output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{0, 1}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }

        if (stlib::mpi::commRank() == 0) {
          objects.push_back(Point{{0}});
        }
        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{0, 2}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
      }
      {
        std::vector<Point> objects;
        if (stlib::mpi::commRank() == 0) {
          objects.push_back(Point{{0}});
          objects.push_back(Point{{0.5}});
        }
        AdaptiveCells output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{1, 1}, {9, 1}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 2, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{0, 2}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
      }
      {
        std::vector<Point> objects;
        if (stlib::mpi::commRank() == 0) {
          objects.push_back(Point{{0}});
          objects.push_back(Point{{0.25}});
          objects.push_back(Point{{0.5}});
          objects.push_back(Point{{0.75}});
        }
        AdaptiveCells output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{2, 1}, {6, 1}, {10, 1}, {14, 1},
                                                                 {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 2, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{1, 2}, {9, 2}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 3, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{1, 2}, {9, 2}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 4, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{0, 4}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
      }
    }

    // 3 levels
    {
      Grid const grid(Point{{0}}, Point{{1}}, 3);
      AdaptiveCells check(grid);
      {
        std::vector<Point> objects;
        AdaptiveCells output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
      }
      {
        std::vector<Point> objects = {Point{{0}}};
        AdaptiveCells output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{0, commSize}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
      }
      {
        std::vector<Point> objects = {Point{{0}}, Point{{0.5}}};

        AdaptiveCells output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{1, commSize}, {17, commSize},
                                                       {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }

        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 2, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{0, 2 * commSize}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
      }
      {
        std::vector<Point> objects = {Point{{0}}, Point{{0.25}}, Point{{0.5}},
                                      Point{{0.75}}};
        AdaptiveCells output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{2, commSize}, {10, commSize},
                                        {18, commSize}, {26, commSize},
                                        {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }

        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 2, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{1, 2 * commSize},
              {17, 2 * commSize}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }

        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 4, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{0, 4 * commSize}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
      }
      {
        std::vector<Point> objects =
          {Point{{0}}, Point{{0.125}}, Point{{0.25}}, Point{{0.375}},
           Point{{0.5}}, Point{{0.625}}, Point{{0.75}}, Point{{0.875}}};
        AdaptiveCells output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 1, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>
            {{3, commSize}, {7, commSize}, {11, commSize}, {15, commSize},
             {19, commSize}, {23, commSize}, {27, commSize}, {31, commSize},
             {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }

        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 2, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>
                           {{2, 2 * commSize}, {10, 2 * commSize},
                                               {18, 2 * commSize},
                                               {26, 2 * commSize}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }

        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 4, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>
                           {{1, 4 * commSize}, {17, 4 * commSize}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }

        output =
          stlib::sfc::buildAdaptiveBlocksFromDistributedObjects<Traits, true>
          (grid, objects, 8, MPI_COMM_WORLD);
        if (commRank == 0) {
          check.buildCells(std::vector<Pair>{{0, 8 * commSize}, {Guard, 0}});
          assert(output == check);
        }
        else {
          assert(output.empty());
        }
      }
    }
  }

  // adaptiveCells()
  // 1-D
  {
    typedef stlib::sfc::Traits<1> Traits;
    typedef Traits::Point Point;
    typedef Traits::BBox BBox;
    typedef stlib::sfc::AdaptiveCells<Traits, BBox, true> AdaptiveCells;
    typedef AdaptiveCells::Grid Grid;
    typedef stlib::sfc::Partition<Traits> Partition;
    Code const Guard = Traits::GuardCode;
    using stlib::sfc::adaptiveCells;

    MPI_Comm const comm = MPI_COMM_WORLD;
    Partition partition(commSize);

    // 0 levels
    {
      Grid const grid(Point{{0}}, Point{{1}}, 0);
      {
        AdaptiveCells localCells;
        std::vector<Point> objects;
        AdaptiveCells const globalCells =
          adaptiveCells(grid, &objects, 1, &localCells, &partition, comm);
        assert(objects.empty());
        assert(localCells.size() == 0);
        assert(globalCells.size() == 0);
        assert(partition.delimiters.front() == 0);
        assert(partition.delimiters.back() == Guard);
      }
      {
        AdaptiveCells localCells;
        std::vector<Point> objects = {Point{{0}}};
        AdaptiveCells const globalCells =
          adaptiveCells(grid, &objects, 1, &localCells, &partition, comm);

        assert(objects.size() == 1);

        assert(localCells.size() == 1);
        assert(localCells.code(0) == 0);
        assert(localCells[0] == (BBox{{{0}}, {{0}}}));
        assert(localCells.delimiter(0) == 0);
        assert(localCells.delimiter(1) == 1);

        assert(globalCells.size() == 1);
        assert(globalCells.code(0) == 0);
        assert(globalCells[0] == (BBox{{{0}}, {{0}}}));
        assert(globalCells.delimiter(0) == 0);
        assert(globalCells.delimiter(1) == std::size_t(commSize));

        assert(partition.delimiters.front() == 0);
        assert(partition.delimiters.back() == Guard);
      }
      {
        AdaptiveCells localCells;
        std::vector<Point> objects = {Point{{0}}, Point{{0.5}}};
        AdaptiveCells const globalCells =
          adaptiveCells(grid, &objects, 1, &localCells, &partition, comm);

        assert(objects.size() == 2);

        assert(localCells.size() == 1);
        assert(localCells.code(0) == 0);
        assert(localCells[0] == (BBox{{{0}}, {{0.5}}}));
        assert(localCells.delimiter(0) == 0);
        assert(localCells.delimiter(1) == 2);

        assert(globalCells.size() == 1);
        assert(globalCells.code(0) == 0);
        assert(globalCells[0] == (BBox{{{0}}, {{0.5}}}));
        assert(globalCells.delimiter(0) == 0);
        assert(globalCells.delimiter(1) == 2 * std::size_t(commSize));

        assert(partition.delimiters.front() == 0);
        assert(partition.delimiters.back() == Guard);
      }
    }
    // 1 level
    {
      Grid const grid(Point{{0}}, Point{{1}}, 1);
      {
        AdaptiveCells localCells;
        std::vector<Point> objects;
        AdaptiveCells const globalCells =
          adaptiveCells(grid, &objects, 1, &localCells, &partition, comm);
        assert(objects.empty());
        assert(localCells.size() == 0);
        assert(globalCells.size() == 0);
        assert(partition.delimiters.front() == 0);
        assert(partition.delimiters.back() == Guard);
      }
      {
        AdaptiveCells localCells;
        std::vector<Point> objects = {Point{{0}}};
        AdaptiveCells const globalCells =
          adaptiveCells(grid, &objects, 1, &localCells, &partition, comm);

        assert(objects.size() == 1);

        assert(localCells.size() == 1);
        assert(localCells.code(0) == 0);
        assert(localCells[0] == (BBox{{{0}}, {{0}}}));
        assert(localCells.delimiter(0) == 0);
        assert(localCells.delimiter(1) == 1);

        assert(globalCells.size() == 1);
        assert(globalCells.code(0) == 0);
        assert(globalCells[0] == (BBox{{{0}}, {{0}}}));
        assert(globalCells.delimiter(0) == 0);
        assert(globalCells.delimiter(1) == std::size_t(commSize));

        assert(partition.delimiters.front() == 0);
        assert(partition.delimiters.back() == Guard);
      }
      {
        AdaptiveCells localCells;
        std::vector<Point> objects = {Point{{0}}, Point{{0.5}}};
        AdaptiveCells const globalCells =
          adaptiveCells(grid, &objects, 1, &localCells, &partition, comm);

        assert(objects.size() == 2);

        assert(localCells.size() == 2);
        assert(localCells.code(0) == 1);
        assert(localCells.code(1) == 3);
        assert(localCells[0] == (BBox{{{0}}, {{0}}}));
        assert(localCells[1] == (BBox{{{0.5}}, {{0.5}}}));
        assert(localCells.delimiter(0) == 0);
        assert(localCells.delimiter(1) == 1);
        assert(localCells.delimiter(2) == 2);

        assert(globalCells.size() == 2);
        assert(globalCells.code(0) == 1);
        assert(globalCells.code(1) == 3);
        assert(globalCells[0] == (BBox{{{0}}, {{0}}}));
        assert(globalCells[1] == (BBox{{{0.5}}, {{0.5}}}));
        assert(globalCells.delimiter(0) == 0);
        assert(globalCells.delimiter(1) == std::size_t(commSize));
        assert(globalCells.delimiter(2) == 2 * std::size_t(commSize));

        assert(partition.delimiters.front() == 0);
        assert(partition.delimiters.back() == Guard);
      }
      {
        AdaptiveCells localCells;
        std::vector<Point> objects;
        if (commRank == 0) {
          objects = std::vector<Point>{Point{{0}}, Point{{0.5}}};
        }
        AdaptiveCells const globalCells =
          adaptiveCells(grid, &objects, 1, &localCells, &partition, comm);

        if (commRank == 0) {
          assert(objects.size() == 2);
        }
        else {
          assert(objects.empty());
        }

        if (commRank == 0) {
          assert(localCells.size() == 2);
          assert(localCells.code(0) == 1);
          assert(localCells.code(1) == 3);
          assert(localCells[0] == (BBox{{{0}}, {{0}}}));
          assert(localCells[1] == (BBox{{{0.5}}, {{0.5}}}));
          assert(localCells.delimiter(0) == 0);
          assert(localCells.delimiter(1) == 1);
          assert(localCells.delimiter(2) == 2);
        }
        else {
          assert(localCells.size() == 0);
        }

        assert(globalCells.size() == 2);
        assert(globalCells.code(0) == 1);
        assert(globalCells.code(1) == 3);
        assert(globalCells[0] == (BBox{{{0}}, {{0}}}));
        assert(globalCells[1] == (BBox{{{0.5}}, {{0.5}}}));
        assert(globalCells.delimiter(0) == 0);
        assert(globalCells.delimiter(1) == 1);
        assert(globalCells.delimiter(2) == 2);

        assert(partition.delimiters.front() == 0);
        assert(partition.delimiters.back() == Guard);
      }
    }
  }

  //distribute(std::vector<_Object>* objects,
  //         MPI_Comm comm, double accuracyGoal = 0.01);
  // 1-D
  {
    typedef std::array<float, 1> Point;
    using stlib::sfc::distribute;

    MPI_Comm const comm = MPI_COMM_WORLD;

    {
      std::vector<Point> objects;
      distribute<float, 1>(&objects, comm);
      assert(objects.empty());
    }
    {
      std::vector<Point> objects;
      if (commRank == 0) {
        objects = std::vector<Point>{Point{{0}}};
      }
      distribute<float, 1>(&objects, comm);
      if (commRank == 0) {
        assert(objects.size() == 1);
      }
      else {
        assert(objects.empty());
      }
    }
    {
      std::vector<Point> objects = {Point{{0}}};
      distribute<float, 1>(&objects, comm);
      if (commRank == 0) {
        assert(objects.size() == std::size_t(commSize));
      }
      else {
        assert(objects.empty());
      }
    }
    {
      // When building the adaptive blocks, the number of levels of refinement
      // will be 0.
      std::vector<Point> objects = {Point{{float(commRank)}}};
      distribute<float, 1>(&objects, comm);
      if (commRank == 0) {
        assert(objects.size() == std::size_t(commSize));
      }
      else {
        assert(objects.empty());
      }
    }
    {
      std::vector<Point> objects;
      for (std::size_t i = 0; i != std::size_t(commSize); ++i) {
        objects.push_back(Point{{float(i)}});
      }
      distribute<float, 1>(&objects, comm, 0.5 / commSize);
      assert(objects.size() == std::size_t(commSize));
    }
  }

  MPI_Finalize();
  return 0;
}
