// -*- C++ -*-

#include "stlib/sfc/Cell.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace stlib;

TEST_CASE("CellExtremePoints.", "[CellExtremePoints]") {
  std::size_t const Dimension = 1;
  typedef float Float;
  typedef std::array<Float, Dimension> Point;
  typedef geom::ExtremePoints<Float, Dimension> ExtremePoints;

  SECTION("CellExtremePoints. Construct from a range of objects.") {
    std::vector<Point> points = {{{0}}, {{1}}};
    sfc::BuildCell<ExtremePoints> buildCell;
    ExtremePoints x = buildCell(points.begin(), points.end());
    REQUIRE(x == (ExtremePoints{{{{{{{0}}, {{1}}}}}}}));
  }
}
