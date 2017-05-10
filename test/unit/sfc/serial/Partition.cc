// -*- C++ -*-

#include "stlib/sfc/Partition.h"
#include "stlib/sfc/AdaptiveCells.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"


using namespace stlib;

TEST_CASE("Partition.", "[Partition]")
{
  SECTION("1-D, 20 levels.") {
    typedef sfc::Traits<1> Traits;
    const std::size_t NumLevels = 20;
    typedef sfc::AdaptiveCells<Traits, void, true> AdaptiveCells;
    typedef AdaptiveCells::Float Float;
    typedef AdaptiveCells::Point Point;
    typedef AdaptiveCells::Code Code;
    typedef sfc::Partition<Traits> Partition;
    Code const Guard = Traits::GuardCode;

    // Uniformly-spaced points.
    std::vector<Point> objects(512);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      objects[i][0] = Float(i) / objects.size();
    }

    {
      AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
      cells.buildCells(&objects);
      for (std::size_t numParts = 1; numParts <= objects.size();
           numParts *= 2) {
        Partition partition(numParts);
        partition(cells);
        REQUIRE(partition.delimiters.size() == numParts + 1);
        REQUIRE(partition.delimiters.front() == Code(0));
        REQUIRE(partition.delimiters.back() == Code(-1));
        const std::size_t stride = objects.size() / numParts;
        for (std::size_t i = 1; i != partition.delimiters.size() - 1; ++i) {
          REQUIRE(partition.delimiters[i] ==
                  cells.grid().location(cells.code(i * stride)));
        }
        REQUIRE(partition.delimiters.back() == Guard);
      }
    }
  }
}
