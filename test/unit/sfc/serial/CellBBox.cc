// -*- C++ -*-

#include "stlib/sfc/Cell.h"
#include "stlib/sfc/Traits.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("CellBBox. merge", "[CellBBox]") {
  std::size_t const Dimension = 1;
  typedef stlib::sfc::Traits<Dimension> Traits;
  typedef Traits::Float Float;
  typedef std::array<Float, Dimension> Point;
  typedef stlib::geom::BBox<Float, Dimension> BBox;

  SECTION("CellBBox. Construct from a range of objects.") {
    std::vector<Point> points = {{{0}}, {{1}}};
    stlib::sfc::BuildCell<BBox> buildCell;
    BBox x = buildCell(points.begin(), points.end());
    REQUIRE(x == (BBox{{{0}}, {{1}}}));
  }
}
