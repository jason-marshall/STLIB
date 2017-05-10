// -*- C++ -*-

#include "stlib/sfc/OrientedBBoxDistance.h"
#include "stlib/simd/shuffle.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace stlib;

TEST_CASE("OrientedBBoxDistance.", "[OrientedBBoxDistance]")
{
  typedef float Float;
  std::size_t const Dimension = 3;
  typedef sfc::OrientedBBoxDistance<Float, Dimension> OrientedBBoxDistance;
  typedef OrientedBBoxDistance::OrientedBBox OrientedBBox;
  typedef OrientedBBoxDistance::Point Point;
  Float const Eps = std::numeric_limits<Float>::epsilon();

  OrientedBBox const orientedBBox = {
    {{0, 0, 0}},
    {{{{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}}},
    {{1, 1, 1}}
  };
  OrientedBBoxDistance const f(orientedBBox);

  SECTION("getDirection") {
    REQUIRE(f.getDirection(Point{{0, 0, -2 * (1 + Eps)}}, 1) == 0);
    REQUIRE(f.getDirection(Point{{0, 0, -2 * (1 - Eps)}}, 1) ==
            std::size_t(-1));
    REQUIRE(f.getDirection(Point{{0, 0, 2 * (1 - Eps)}}, 1) ==
            std::size_t(-1));
    REQUIRE(f.getDirection(Point{{0, 0, 2 * (1 + Eps)}}, 1) == 1);
  }

  SECTION("areAnyRelevant") {
    std::size_t const VectorSize = simd::Vector<Float>::Size;

    Float const queryPointsMaxRadius = 0;
    std::vector<Float, simd::allocator<Float> > queryPointData;
    std::vector<Float, simd::allocator<Float> >
      upperBounds(VectorSize, -std::numeric_limits<Float>::infinity());

    SECTION("above top face center") {
      Point const queryPointsCenter = {{0, 0, 2}};
      std::size_t const direction = f.getDirection(queryPointsCenter,
                                                   queryPointsMaxRadius);
      simd::aosToHybridSoa<Dimension>(&queryPointsCenter,
                                      &queryPointsCenter + 1, &queryPointData);

      upperBounds[0] = (3 - 2) * (1 - Eps);
      REQUIRE_FALSE(f.areAnyRelevant(queryPointsCenter, queryPointsMaxRadius,
                                     direction, queryPointData, upperBounds));
      upperBounds[0] = (3 - 2) * (1 + Eps);
      REQUIRE(f.areAnyRelevant(queryPointsCenter, queryPointsMaxRadius,
                               direction, queryPointData, upperBounds));
    }

    SECTION("above bottom face center") {
      Point const queryPointsCenter = {{0, 0, -2}};
      std::size_t const direction = f.getDirection(queryPointsCenter,
                                                   queryPointsMaxRadius);
      simd::aosToHybridSoa<Dimension>(&queryPointsCenter,
                                      &queryPointsCenter + 1, &queryPointData);

      upperBounds[0] = (3 - 2) * (1 - Eps);
      REQUIRE_FALSE(f.areAnyRelevant(queryPointsCenter, queryPointsMaxRadius,
                                     direction, queryPointData, upperBounds));
      upperBounds[0] = (3 - 2) * (1 + Eps);
      REQUIRE(f.areAnyRelevant(queryPointsCenter, queryPointsMaxRadius,
                               direction, queryPointData, upperBounds));
    }

    SECTION("above corner 0") {
      Point const queryPointsCenter = {{-1, -1, 3}};
      std::size_t const direction = f.getDirection(queryPointsCenter,
                                                   queryPointsMaxRadius);
      simd::aosToHybridSoa<Dimension>(&queryPointsCenter,
                                      &queryPointsCenter + 1, &queryPointData);

      upperBounds[0] = (4 - 2) * (1 - Eps);
      REQUIRE_FALSE(f.areAnyRelevant(queryPointsCenter, queryPointsMaxRadius,
                                     direction, queryPointData, upperBounds));
      upperBounds[0] = (4 - 2) * (1 + Eps);
      REQUIRE(f.areAnyRelevant(queryPointsCenter, queryPointsMaxRadius,
                               direction, queryPointData, upperBounds));
    }
    SECTION("above corner 1") {
      Point const queryPointsCenter = {{1, -1, 3}};
      std::size_t const direction = f.getDirection(queryPointsCenter,
                                                   queryPointsMaxRadius);
      simd::aosToHybridSoa<Dimension>(&queryPointsCenter,
                                      &queryPointsCenter + 1, &queryPointData);

      upperBounds[0] = (4 - 2) * (1 - Eps);
      REQUIRE_FALSE(f.areAnyRelevant(queryPointsCenter, queryPointsMaxRadius,
                                     direction, queryPointData, upperBounds));
      upperBounds[0] = (4 - 2) * (1 + Eps);
      REQUIRE(f.areAnyRelevant(queryPointsCenter, queryPointsMaxRadius,
                               direction, queryPointData, upperBounds));
    }
    SECTION("above corner 2") {
      Point const queryPointsCenter = {{-1, 1, 3}};
      std::size_t const direction = f.getDirection(queryPointsCenter,
                                                   queryPointsMaxRadius);
      simd::aosToHybridSoa<Dimension>(&queryPointsCenter,
                                      &queryPointsCenter + 1, &queryPointData);

      upperBounds[0] = (4 - 2) * (1 - Eps);
      REQUIRE_FALSE(f.areAnyRelevant(queryPointsCenter, queryPointsMaxRadius,
                                     direction, queryPointData, upperBounds));
      upperBounds[0] = (4 - 2) * (1 + Eps);
      REQUIRE(f.areAnyRelevant(queryPointsCenter, queryPointsMaxRadius,
                               direction, queryPointData, upperBounds));
    }
    SECTION("above corner 3") {
      Point const queryPointsCenter = {{1, 1, 3}};
      std::size_t const direction = f.getDirection(queryPointsCenter,
                                                   queryPointsMaxRadius);
      simd::aosToHybridSoa<Dimension>(&queryPointsCenter,
                                      &queryPointsCenter + 1, &queryPointData);

      upperBounds[0] = (4 - 2) * (1 - Eps);
      REQUIRE_FALSE(f.areAnyRelevant(queryPointsCenter, queryPointsMaxRadius,
                                     direction, queryPointData, upperBounds));
      upperBounds[0] = (4 - 2) * (1 + Eps);
      REQUIRE(f.areAnyRelevant(queryPointsCenter, queryPointsMaxRadius,
                               direction, queryPointData, upperBounds));
    }
  }
}
