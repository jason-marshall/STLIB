// -*- C++ -*-

#include "stlib/sfc/LocationCode.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"


using namespace stlib;

TEST_CASE("LocationCode.", "[LocationCode]")
{
  SECTION("1-D, 0 levels.") {
    typedef sfc::Traits<1> Traits;
    typedef sfc::LocationCode<Traits> LocationCode;
    typedef LocationCode::Code Code;
    typedef LocationCode::Float Float;
    typedef LocationCode::Point Point;
    {
      // Default constructor.
      LocationCode x;
    }
    LocationCode b(Point{{0}}, Point{{1}}, 0);
    REQUIRE(b.lowerCorner() == Point{{0}});
    REQUIRE(b.lengths() == Point{{1}});
    REQUIRE(b.numLevels() == 0);
    REQUIRE(b.code(Point{{0}}) == Code(0));
    REQUIRE(b.code(Point{{1 - std::numeric_limits<Float>::epsilon()}}) == 
            Code(0));
    REQUIRE(b.isValid(0));
    REQUIRE_FALSE(b.isValid(1));
  }

  SECTION("1-D, 1 levels.") {
    typedef sfc::Traits<1> Traits;
    typedef sfc::LocationCode<Traits> LocationCode;
    typedef LocationCode::Code Code;
    typedef LocationCode::Float Float;
    typedef LocationCode::Point Point;
    const Float Eps = std::numeric_limits<Float>::epsilon();
    const Point LowerCorner = {{0}};
    const Point Lengths = {{1}};
    const std::size_t NumLevels = 1;
    LocationCode b(LowerCorner, Lengths, NumLevels);
    REQUIRE(b.lowerCorner() == LowerCorner);
    REQUIRE(b.lengths() == Lengths);
    REQUIRE(b.numLevels() == NumLevels);

    REQUIRE(b.code(Point{{0}}) == Code(0));// 0
    REQUIRE(b.code(Point{{0.5}}) == Code(1));// 1
    REQUIRE(b.code(Point{{1 - Eps}}) == Code(1));// 1

    REQUIRE(b.nextParent(0x0) == Code(0x2));
    REQUIRE(b.nextParent(0x1) == Code(0x2));

    REQUIRE(b.isValid(0));
    REQUIRE(b.isValid((Code(1) <<
                      (Traits::Dimension * NumLevels)) - 1));
    REQUIRE_FALSE(b.isValid(Code(1) <<
                            (Traits::Dimension * NumLevels)));
  }

  SECTION("1-D, 2 levels.") {
    typedef sfc::Traits<1> Traits;
    typedef sfc::LocationCode<Traits> LocationCode;
    typedef LocationCode::Code Code;
    typedef LocationCode::Float Float;
    typedef LocationCode::Point Point;
    const Float Eps = std::numeric_limits<Float>::epsilon();
    const Point LowerCorner = {{0}};
    const Point Lengths = {{1}};
    const std::size_t NumLevels = 2;
    LocationCode b(LowerCorner, Lengths, NumLevels);
    REQUIRE(b.lowerCorner() == LowerCorner);
    REQUIRE(b.lengths() == Lengths);
    REQUIRE(b.numLevels() == NumLevels);

    REQUIRE(b.code(Point{{0}}) == Code(0x0));// 00
    REQUIRE(b.code(Point{{0.5}}) == Code(0x2));// 10
    REQUIRE(b.code(Point{{1 - Eps}}) == Code(0x3));// 11

    REQUIRE(b.nextParent(0x0) == Code(0x2));
    REQUIRE(b.nextParent(0x1) == Code(0x2));
    REQUIRE(b.nextParent(0x2) == Code(0x4));
    REQUIRE(b.nextParent(0x3) == Code(0x4));

    REQUIRE(b.isValid(0));
    REQUIRE(b.isValid((Code(1) <<
                      (Traits::Dimension * NumLevels)) - 1));
    REQUIRE_FALSE(b.isValid(Code(1) <<
                            (Traits::Dimension * NumLevels)));
  }

  SECTION("1-D, 3 levels.") {
    typedef sfc::Traits<1> Traits;
    typedef sfc::LocationCode<Traits> LocationCode;
    typedef LocationCode::Code Code;
    typedef LocationCode::Float Float;
    typedef LocationCode::Point Point;
    const Float Eps = std::numeric_limits<Float>::epsilon();
    const Point LowerCorner = {{0}};
    const Point Lengths = {{1}};
    const std::size_t NumLevels = 3;
    LocationCode b(LowerCorner, Lengths, NumLevels);
    REQUIRE(b.lowerCorner() == LowerCorner);
    REQUIRE(b.lengths() == Lengths);
    REQUIRE(b.numLevels() == NumLevels);

    REQUIRE(b.code(Point{{0}}) == Code(0x0));// 000
    REQUIRE(b.code(Point{{0.5}}) == Code(0x4));// 100
    REQUIRE(b.code(Point{{1 - Eps}}) == Code(0x7));// 111

    REQUIRE(b.isValid(0));
    REQUIRE(b.isValid((Code(1) <<
                      (Traits::Dimension * NumLevels)) - 1));
    REQUIRE_FALSE(b.isValid(Code(1) <<
                            (Traits::Dimension * NumLevels)));
  }

  SECTION("1-D, 4 levels.") {
    typedef sfc::Traits<1> Traits;
    typedef sfc::LocationCode<Traits> LocationCode;
    typedef LocationCode::Code Code;
    typedef LocationCode::Float Float;
    typedef LocationCode::Point Point;
    const Float Eps = std::numeric_limits<Float>::epsilon();
    const Point LowerCorner = {{0}};
    const Point Lengths = {{1}};
    const std::size_t NumLevels = 4;
    LocationCode b(LowerCorner, Lengths, NumLevels);
    REQUIRE(b.lowerCorner() == LowerCorner);
    REQUIRE(b.lengths() == Lengths);
    REQUIRE(b.numLevels() == NumLevels);

    REQUIRE(b.code(Point{{0}}) == Code(0x0));// 0000
    REQUIRE(b.code(Point{{0.5}}) == Code(0x8));// 1000
    REQUIRE(b.code(Point{{1 - Eps}}) == Code(0xF));// 1111

    REQUIRE(b.isValid(0));
    REQUIRE(b.isValid((Code(1) <<
                      (Traits::Dimension * NumLevels)) - 1));
    REQUIRE_FALSE(b.isValid(Code(1) <<
                            (Traits::Dimension * NumLevels)));
  }

  SECTION("1-D, 6 levels.") {
    typedef sfc::Traits<1> Traits;
    typedef sfc::LocationCode<Traits> LocationCode;
    typedef LocationCode::Code Code;
    typedef LocationCode::Float Float;
    typedef LocationCode::Point Point;
    const Float Eps = std::numeric_limits<Float>::epsilon();
    const Point LowerCorner = {{0}};
    const Point Lengths = {{1}};
    const std::size_t NumLevels = 6;
    LocationCode b(LowerCorner, Lengths, NumLevels);
    REQUIRE(b.lowerCorner() == LowerCorner);
    REQUIRE(b.lengths() == Lengths);
    REQUIRE(b.numLevels() == NumLevels);

    REQUIRE(b.code(Point{{0}}) == Code(0x0));// 000000
    REQUIRE(b.code(Point{{0.5}}) == Code(0x20));// 100000
    REQUIRE(b.code(Point{{1 - Eps}}) == Code(0x3F));// 111111

    REQUIRE(b.isValid(0));
    REQUIRE(b.isValid((Code(1) <<
                      (Traits::Dimension * NumLevels)) - 1));
    REQUIRE_FALSE(b.isValid(Code(1) <<
                            (Traits::Dimension * NumLevels)));
  }

  SECTION("1-D, max levels.") {
    typedef sfc::Traits<1> Traits;
    typedef sfc::LocationCode<Traits> LocationCode;
    typedef LocationCode::Code Code;
    typedef LocationCode::Float Float;
    typedef LocationCode::Point Point;
    const Float Eps = std::numeric_limits<Float>::epsilon();
    const Point LowerCorner = {{0}};
    const Point Lengths = {{1}};
    const std::size_t NumLevels = LocationCode::MaxLevels;
    LocationCode b(LowerCorner, Lengths);
    REQUIRE(b.lowerCorner() == LowerCorner);
    REQUIRE(b.lengths() == Lengths);
    REQUIRE(b.numLevels() == NumLevels);

    REQUIRE(b.code(Point{{0}}) == Code(0x0));// 000000

    REQUIRE(b.isValid(0));
    REQUIRE(b.isValid((Code(1) <<
                      (Traits::Dimension * NumLevels)) - 1));
    REQUIRE_FALSE(b.isValid(Code(1) <<
                            (Traits::Dimension * NumLevels)));
  }

  SECTION("2-D, 3 levels.") {
    typedef sfc::Traits<2> Traits;
    typedef sfc::LocationCode<Traits> LocationCode;
    typedef LocationCode::Code Code;
    typedef LocationCode::Float Float;
    typedef LocationCode::Point Point;
    typedef LocationCode::Code Code;
    const Float Eps = std::numeric_limits<Float>::epsilon();
    const Point LowerCorner = {{0, 0}};
    const Point Lengths = {{1, 1}};
    const std::size_t NumLevels = 3;
    LocationCode b(LowerCorner, Lengths, NumLevels);
    REQUIRE(b.lowerCorner() == LowerCorner);
    REQUIRE(b.lengths() == Lengths);
    REQUIRE(b.numLevels() == NumLevels);

    REQUIRE(b.code(Point{{0, 0}}) == Code(0x0));// 000000
    REQUIRE(b.code(Point{{0.5, 0.5}}) == Code(0x30));// 110000
    REQUIRE(b.code(Point{{1 - Eps, 1 - Eps}}) == Code(0x3F));// 111111

    REQUIRE(b.isValid(0));
    REQUIRE(b.isValid((Code(1) <<
                      (Traits::Dimension * NumLevels)) - 1));
    REQUIRE_FALSE(b.isValid(Code(1) <<
                            (Traits::Dimension * NumLevels)));
  }

  SECTION("Tight bounding box constructor. 1-D, 0 levels.") {
    typedef sfc::Traits<1> Traits;
    typedef sfc::LocationCode<Traits> LocationCode;
    typedef LocationCode::Float Float;
    LocationCode b(geom::BBox<Float, Traits::Dimension>{{{0}}, {{1}}}, 1);
    REQUIRE(-0.1 < b.lowerCorner()[0]);
    REQUIRE(b.lowerCorner()[0] < 0);
    REQUIRE(1 < b.lengths()[0]);
    REQUIRE(b.lengths()[0] < 1.1);
    REQUIRE(b.numLevels() == 0);
  }

  SECTION("Tight bounding box constructor. 2-D, 3 levels.") {
    typedef sfc::Traits<2> Traits;
    typedef sfc::LocationCode<Traits> LocationCode;
    typedef LocationCode::Float Float;
    LocationCode b(geom::BBox<Float, Traits::Dimension>{{{0, 0}}, {{1, 1}}},
                   0.125);
    REQUIRE(-0.1 < b.lowerCorner()[0]);
    REQUIRE(b.lowerCorner()[0] < 0);
    REQUIRE(1 < b.lengths()[0]);
    REQUIRE(b.lengths()[0] < 1.1);
    REQUIRE(b.numLevels() == 3);
  }

  SECTION("Equality.") {
    typedef sfc::Traits<1> Traits;
    typedef sfc::LocationCode<Traits> LocationCode;
    typedef LocationCode::Point Point;
    const LocationCode x(Point{{0}}, Point{{1}}, 7);
    {
      const LocationCode y(Point{{0}}, Point{{1}}, 7);
      REQUIRE(x == y);
    }
    {
      const LocationCode y(Point{{1}}, Point{{1}}, 7);
      REQUIRE_FALSE((x == y));
    }
    {
      const LocationCode y(Point{{0}}, Point{{2}}, 7);
      REQUIRE_FALSE((x == y));
    }
    {
      const LocationCode y(Point{{0}}, Point{{1}}, 8);
      REQUIRE_FALSE((x == y));
    }
  }
}
