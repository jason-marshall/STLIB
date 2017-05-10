// -*- C++ -*-

#include "stlib/sfc/BlockCode.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"


using namespace stlib;

TEST_CASE("1-D, 0 levels", "[BlockCode]") {
  typedef sfc::Traits<1> Traits;
  typedef sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Code Code;
  typedef BlockCode::Float Float;
  typedef BlockCode::Point Point;
  Code const GuardCode = Traits::GuardCode;
  // Default constructor.
  {
    BlockCode b;
  }
  BlockCode b(Point{{0}}, Point{{1}}, 0);
  REQUIRE(b.lowerCorner() == Point{{0}});
  REQUIRE(b.lengths() == Point{{1}});
  REQUIRE(b.numLevels() == 0);
  REQUIRE(b.levelBits() == 0);
  REQUIRE(b.code(Point{{0}}) == Code(0));
  REQUIRE(b.code(Point{{1 - std::numeric_limits<Float>::epsilon()}}) == 
          Code(0));
  REQUIRE(b.isValid(0));
  REQUIRE_FALSE(b.isValid(1));
  REQUIRE(b.level(0) == 0);
  REQUIRE(b.level(1) == 0);
  REQUIRE(b.location(0) == Code(0));
  REQUIRE(b.next(0x0) == Code(0x1));
  // Coarsen.
  {
    std::vector<Code> objectCodes;
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 1);
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes(1, 0);
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 2);
    REQUIRE(cellCodes[0] == Code(0));
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes(2, 0);
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 2);
    REQUIRE(cellCodes[0] == Code(0));
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes(2, 0);
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 10);
    REQUIRE(cellCodes.size() == 2);
    REQUIRE(cellCodes[0] == Code(0));
    REQUIRE(cellCodes.back() == GuardCode);
  }
}


TEST_CASE("1-D, 1 levels", "[BlockCode]") {
  typedef sfc::Traits<1> Traits;
  typedef sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Code Code;
  typedef BlockCode::Float Float;
  typedef BlockCode::Point Point;
  Code const GuardCode = Traits::GuardCode;
  const Float Eps = std::numeric_limits<Float>::epsilon();
  const Point LowerCorner = {{0}};
  const Point Lengths = {{1}};
  const std::size_t NumLevels = 1;
  BlockCode b(LowerCorner, Lengths, NumLevels);
  REQUIRE(b.lowerCorner() == LowerCorner);
  REQUIRE(b.lengths() == Lengths);
  REQUIRE(b.numLevels() == NumLevels);
  REQUIRE(b.levelBits() == 1);

  REQUIRE(b.code(Point{{0}}) == Code(1));// 01
  REQUIRE(b.code(Point{{0.5}}) == Code(3));// 11
  REQUIRE(b.code(Point{{1 - Eps}}) == Code(3));// 11

  REQUIRE(b.isValid(0)); // 0 0
  REQUIRE_FALSE(b.isValid(2)); // 1 0
  REQUIRE(b.isValid(1)); // 0 1
  REQUIRE(b.isValid(3)); // 1 1
  REQUIRE_FALSE(b.isValid(4)); // 1 0 0

  REQUIRE(b.level(0) == 0);
  REQUIRE(b.level(1) == 1);
  REQUIRE(b.level(2) == 0);
  REQUIRE(b.level(3) == 1);

  REQUIRE(b.location(0x0) == Code(0x0)); // 0 0 -> 0 0
  REQUIRE(b.location(0x1) == Code(0x0)); // 0 1 -> 0 0
  REQUIRE(b.location(0x3) == Code(0x2)); // 1 1 -> 1 0

  REQUIRE(b.parent(0x1) == Code(0x0)); // 0 1 -> 0 0
  REQUIRE(b.parent(0x3) == Code(0x0)); // 1 1 -> 0 0

  REQUIRE(b.next(0x0) == Code(0x4)); // 0 0 -> 1 0 0
  REQUIRE(b.next(0x1) == Code(0x3)); // 0 1 -> 1 1
  REQUIRE(b.next(0x3) == Code(0x5)); // 1 1 -> 1 0 1

  // Coarsen.
  {
    std::vector<Code> objectCodes;
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 1);
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes(1, 1);
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 2);
    REQUIRE(cellCodes[0] == Code(0));
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes(2, 1);
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 2);
    REQUIRE(cellCodes[0] == Code(1));
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes(2, 3);
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 2);
    REQUIRE(cellCodes[0] == Code(3));
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes;
    objectCodes.push_back(1);
    objectCodes.push_back(3);
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 3);
    REQUIRE(cellCodes[0] == Code(1));
    REQUIRE(cellCodes[1] == Code(3));
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes;
    objectCodes.push_back(1);
    objectCodes.push_back(3);
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 2);
    REQUIRE(cellCodes.size() == 2);
    REQUIRE(cellCodes[0] == Code(0));
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes;
    objectCodes.push_back(1);
    objectCodes.push_back(3);
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 10);
    REQUIRE(cellCodes.size() == 2);
    REQUIRE(cellCodes[0] == Code(0));
    REQUIRE(cellCodes.back() == GuardCode);
  }
}


TEST_CASE("1-D, 2 levels", "[BlockCode]") {
  typedef sfc::Traits<1> Traits;
  typedef sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Code Code;
  typedef BlockCode::Float Float;
  typedef BlockCode::Point Point;
  Code const GuardCode = Traits::GuardCode;
  const Float Eps = std::numeric_limits<Float>::epsilon();
  const Point LowerCorner = {{0}};
  const Point Lengths = {{1}};
  const std::size_t NumLevels = 2;
  BlockCode b(LowerCorner, Lengths, NumLevels);
  REQUIRE(b.lowerCorner() == LowerCorner);
  REQUIRE(b.lengths() == Lengths);
  REQUIRE(b.numLevels() == NumLevels);
  REQUIRE(b.levelBits() == 2);

  REQUIRE(b.code(Point{{0}}) == Code(0x2));// 00 10
  REQUIRE(b.code(Point{{0.5}}) == Code(0xA));// 10 10
  REQUIRE(b.code(Point{{1 - Eps}}) == Code(0xE));// 11 10

  REQUIRE(b.isValid(0x0)); // 00 00
  REQUIRE_FALSE(b.isValid(0x4)); // 01 00
  REQUIRE_FALSE(b.isValid(0x8)); // 10 00
  REQUIRE_FALSE(b.isValid(0xC)); // 11 00

  REQUIRE(b.isValid(0x1)); // 00 01
  REQUIRE_FALSE(b.isValid(0x5)); // 01 01
  REQUIRE(b.isValid(0x9)); // 10 01
  REQUIRE_FALSE(b.isValid(0xD)); // 11 01

  REQUIRE(b.isValid(0x1)); // 00 10
  REQUIRE(b.isValid(0x1)); // 01 10
  REQUIRE(b.isValid(0x1)); // 10 10
  REQUIRE(b.isValid(0x1)); // 11 10

  REQUIRE_FALSE(b.isValid(0x3)); // 00 11
  REQUIRE_FALSE(b.isValid(0x10)); // 1 00 00

  for (std::size_t i = 0; i <= NumLevels; ++i) {
    REQUIRE(b.level(i) == i);
  }
  REQUIRE(b.level(4) == 0);

  REQUIRE(b.parent(0x1) == Code(0x0)); // 00 01 -> 00 00
  REQUIRE(b.parent(0x9) == Code(0x0)); // 10 01 -> 00 00
  REQUIRE(b.parent(0x2) == Code(0x1)); // 00 10 -> 00 01
  REQUIRE(b.parent(0x2) == Code(0x1)); // 00 10 -> 00 01
  REQUIRE(b.parent(0x6) == Code(0x1)); // 01 10 -> 00 01
  REQUIRE(b.parent(0xA) == Code(0x9)); // 10 10 -> 10 01
  REQUIRE(b.parent(0xE) == Code(0x9)); // 11 10 -> 10 01

  REQUIRE(b.next(0x0) == Code(0x10)); // 00 00 -> 1 00 00
  REQUIRE(b.next(0x1) == Code(0x9)); // 00 01 -> 10 01
  REQUIRE(b.next(0x9) == Code(0x11)); // 10 01 -> 1 00 01
  REQUIRE(b.next(0x2) == Code(0x6)); // 00 10 -> 01 10
  REQUIRE(b.next(0x6) == Code(0xA)); // 01 10 -> 10 10
  REQUIRE(b.next(0xA) == Code(0xE)); // 10 10 -> 11 10
  REQUIRE(b.next(0xE) == Code(0x12)); // 11 10 -> 1 00 10

  // Sort.
  {
    std::vector<Point> objects;
    std::vector<Code> objectCodes;
    sort(b, &objects, &objectCodes);
    REQUIRE(objects.empty());
    REQUIRE(objectCodes.empty());
  }
  {
    std::vector<Point> objects;
    objects.push_back(Point{{0}});
    objects.push_back(Point{{0.25}});
    objects.push_back(Point{{0.5}});
    objects.push_back(Point{{0.75}});
    std::vector<Code> objectCodes;
    sort(b, &objects, &objectCodes);
    REQUIRE(objects.size() == 4);
    REQUIRE(objects[0] == (Point{{0}}));
    REQUIRE(objects[1] == (Point{{0.25}}));
    REQUIRE(objects[2] == (Point{{0.5}}));
    REQUIRE(objects[3] == (Point{{0.75}}));
    REQUIRE(objectCodes.size() == 4);
    REQUIRE(objectCodes[0] == Code(2)); // 00 10
    REQUIRE(objectCodes[1] == Code(6)); // 01 10
    REQUIRE(objectCodes[2] == Code(10)); // 10 10
    REQUIRE(objectCodes[3] == Code(14)); // 11 10
  }
  {
    std::vector<Point> objects;
    objects.push_back(Point{{0.75}});
    objects.push_back(Point{{0.5}});
    objects.push_back(Point{{0.25}});
    objects.push_back(Point{{0}});
    std::vector<Code> objectCodes;
    sort(b, &objects, &objectCodes);
    REQUIRE(objects.size() == 4);
    REQUIRE(objects[0] == (Point{{0}}));
    REQUIRE(objects[1] == (Point{{0.25}}));
    REQUIRE(objects[2] == (Point{{0.5}}));
    REQUIRE(objects[3] == (Point{{0.75}}));
    REQUIRE(objectCodes.size() == 4);
    REQUIRE(objectCodes[0] == Code(2)); // 00 10
    REQUIRE(objectCodes[1] == Code(6)); // 01 10
    REQUIRE(objectCodes[2] == Code(10)); // 10 10
    REQUIRE(objectCodes[3] == Code(14)); // 11 10
  }
  // Coarsen.
  {
    std::vector<Code> objectCodes;
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 1);
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes(1, 2);
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 2);
    REQUIRE(cellCodes[0] == Code(0));
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes(2, 2);
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 2);
    REQUIRE(cellCodes[0] == Code(2));
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes(2, 14); // 11 10
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 2);
    REQUIRE(cellCodes[0] == Code(14));
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes;
    objectCodes.push_back(2); // 00 10
    objectCodes.push_back(14); // 11 10
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 3);
    REQUIRE(cellCodes[0] == Code(1)); // 00 01
    REQUIRE(cellCodes[1] == Code(9)); // 10 01
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes;
    objectCodes.push_back(2); // 00 10
    objectCodes.push_back(6); // 01 10
    objectCodes.push_back(14); // 11 10
    std::vector<Code> cellCodes;
    coarsen(b, objectCodes, &cellCodes, 1);
    REQUIRE(cellCodes.size() == 4);
    REQUIRE(cellCodes[0] == Code(2)); // 00 10
    REQUIRE(cellCodes[1] == Code(6)); // 01 10
    REQUIRE(cellCodes[2] == Code(9)); // 10 01
    REQUIRE(cellCodes.back() == GuardCode);
  }
  {
    std::vector<Code> objectCodes;
    objectCodes.push_back(2); // 00 10
    objectCodes.push_back(6); // 01 10
    objectCodes.push_back(10); // 10 10
    objectCodes.push_back(14); // 11 10
    {
      std::vector<Code> cellCodes;
      coarsen(b, objectCodes, &cellCodes, 1);
      REQUIRE(cellCodes.size() == 5);
      REQUIRE(cellCodes[0] == Code(2)); // 00 10
      REQUIRE(cellCodes[1] == Code(6)); // 01 10
      REQUIRE(cellCodes[2] == Code(10)); // 10 10
      REQUIRE(cellCodes[3] == Code(14)); // 11 10
      REQUIRE(cellCodes.back() == GuardCode);
    }
    {
      std::vector<Code> cellCodes;
      coarsen(b, objectCodes, &cellCodes, 2);
      REQUIRE(cellCodes.size() == 3);
      REQUIRE(cellCodes[0] == Code(1)); // 00 01
      REQUIRE(cellCodes[1] == Code(9)); // 10 01
      REQUIRE(cellCodes.back() == GuardCode);
    }
    {
      std::vector<Code> cellCodes;
      coarsen(b, objectCodes, &cellCodes, 3);
      REQUIRE(cellCodes.size() == 3);
      REQUIRE(cellCodes[0] == Code(1)); // 00 01
      REQUIRE(cellCodes[1] == Code(9)); // 10 01
      REQUIRE(cellCodes.back() == GuardCode);
    }
    {
      std::vector<Code> cellCodes;
      coarsen(b, objectCodes, &cellCodes, 4);
      REQUIRE(cellCodes.size() == 2);
      REQUIRE(cellCodes[0] == Code(0)); // 00 00
      REQUIRE(cellCodes.back() == GuardCode);
    }
    {
      std::vector<Code> cellCodes;
      coarsen(b, objectCodes, &cellCodes, 5);
      REQUIRE(cellCodes.size() == 2);
      REQUIRE(cellCodes[0] == Code(0)); // 00 00
      REQUIRE(cellCodes.back() == GuardCode);
    }
  }
  {
    std::vector<Code> objectCodes;
    objectCodes.push_back(2); // 00 10
    objectCodes.push_back(2); // 00 10
    objectCodes.push_back(6); // 01 10
    objectCodes.push_back(10); // 10 10
    objectCodes.push_back(14); // 11 10
    {
      std::vector<Code> cellCodes;
      coarsen(b, objectCodes, &cellCodes, 1);
      REQUIRE(cellCodes.size() == 5);
      REQUIRE(cellCodes[0] == Code(2)); // 00 10
      REQUIRE(cellCodes[1] == Code(6)); // 01 10
      REQUIRE(cellCodes[2] == Code(10)); // 10 10
      REQUIRE(cellCodes[3] == Code(14)); // 11 10
      REQUIRE(cellCodes.back() == GuardCode);
    }
    {
      std::vector<Code> cellCodes;
      coarsen(b, objectCodes, &cellCodes, 2);
      REQUIRE(cellCodes.size() == 4);
      REQUIRE(cellCodes[0] == Code(2)); // 00 10
      REQUIRE(cellCodes[1] == Code(6)); // 01 10
      REQUIRE(cellCodes[2] == Code(9)); // 10 01
      REQUIRE(cellCodes.back() == GuardCode);
    }
    {
      std::vector<Code> cellCodes;
      coarsen(b, objectCodes, &cellCodes, 3);
      REQUIRE(cellCodes.size() == 3);
      REQUIRE(cellCodes[0] == Code(1)); // 00 01
      REQUIRE(cellCodes[1] == Code(9)); // 10 01
      REQUIRE(cellCodes.back() == GuardCode);
    }
    {
      std::vector<Code> cellCodes;
      coarsen(b, objectCodes, &cellCodes, 4);
      REQUIRE(cellCodes.size() == 3);
      REQUIRE(cellCodes[0] == Code(1)); // 00 01
      REQUIRE(cellCodes[1] == Code(9)); // 10 01
      REQUIRE(cellCodes.back() == GuardCode);
    }
    {
      std::vector<Code> cellCodes;
      coarsen(b, objectCodes, &cellCodes, 5);
      REQUIRE(cellCodes.size() == 2);
      REQUIRE(cellCodes[0] == Code(0)); // 00 00
      REQUIRE(cellCodes.back() == GuardCode);
    }
    {
      std::vector<Code> cellCodes;
      coarsen(b, objectCodes, &cellCodes, 6);
      REQUIRE(cellCodes.size() == 2);
      REQUIRE(cellCodes[0] == Code(0)); // 00 00
      REQUIRE(cellCodes.back() == GuardCode);
    }
  }
}


TEST_CASE("1-D, 3 levels", "[BlockCode]") {
  typedef sfc::Traits<1> Traits;
  typedef sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Code Code;
  typedef BlockCode::Float Float;
  typedef BlockCode::Point Point;
  const Float Eps = std::numeric_limits<Float>::epsilon();
  const Point LowerCorner = {{0}};
  const Point Lengths = {{1}};
  const std::size_t NumLevels = 3;
  BlockCode b(LowerCorner, Lengths, NumLevels);
  REQUIRE(b.lowerCorner() == LowerCorner);
  REQUIRE(b.lengths() == Lengths);
  REQUIRE(b.numLevels() == NumLevels);
  REQUIRE(b.levelBits() == 2);
  REQUIRE(b.code(Point{{0}}) == Code(0x3));// 000 11
  REQUIRE(b.code(Point{{0.5}}) == Code(0x13));// 100 11
  REQUIRE(b.code(Point{{1 - Eps}}) == Code(0x1F));// 111 11
  for (std::size_t i = 0; i <= NumLevels; ++i) {
    REQUIRE(b.level(i) == i);
  }
  REQUIRE(b.level(4) == 0);
}


TEST_CASE("1-D, 4 levels", "[BlockCode]") {
  typedef sfc::Traits<1> Traits;
  typedef sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Code Code;
  typedef BlockCode::Float Float;
  typedef BlockCode::Point Point;
  const Float Eps = std::numeric_limits<Float>::epsilon();
  const Point LowerCorner = {{0}};
  const Point Lengths = {{1}};
  const std::size_t NumLevels = 4;
  BlockCode b(LowerCorner, Lengths, NumLevels);
  REQUIRE(b.lowerCorner() == LowerCorner);
  REQUIRE(b.lengths() == Lengths);
  REQUIRE(b.numLevels() == NumLevels);
  REQUIRE(b.levelBits() == 3);
  REQUIRE(b.code(Point{{0}}) == Code(0x4));// 0000 100
  REQUIRE(b.code(Point{{0.5}}) ==
          Code(0x44));// 1000 100 = 100 0100
  REQUIRE(b.code(Point{{1 - Eps}}) == Code(0x7C));// 1111 100 = 111 1100
  for (std::size_t i = 0; i <= NumLevels; ++i) {
    REQUIRE(b.level(i) == i);
  }
  REQUIRE(b.level(8) == 0);
}


TEST_CASE("1-D, 6 levels", "[BlockCode]") {
  typedef sfc::Traits<1> Traits;
  typedef sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Code Code;
  typedef BlockCode::Float Float;
  typedef BlockCode::Point Point;
  const Float Eps = std::numeric_limits<Float>::epsilon();
  const Point LowerCorner = {{0}};
  const Point Lengths = {{1}};
  const std::size_t NumLevels = 6;
  BlockCode b(LowerCorner, Lengths, NumLevels);
  REQUIRE(b.lowerCorner() == LowerCorner);
  REQUIRE(b.lengths() == Lengths);
  REQUIRE(b.numLevels() == NumLevels);
  REQUIRE(b.levelBits() == 3);
  REQUIRE(b.code(Point{{0}}) == Code(0x6));// 000000 110
  REQUIRE(b.code(Point{{0.5}}) ==
          Code(0x106));// 100000 110 = 1 0000 0110
  REQUIRE(b.code(Point{{1 - Eps}}) ==
          Code(0x1FE));// 111111 110 = 1 1111 1110
  for (std::size_t i = 0; i <= NumLevels; ++i) {
    REQUIRE(b.level(i) == i);
  }
  REQUIRE(b.level(8) == 0);
}


TEST_CASE("1-D, max levels", "[BlockCode]") {
  typedef sfc::Traits<1> Traits;
  typedef sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Code Code;
  typedef BlockCode::Float Float;
  typedef BlockCode::Point Point;
  const Float Eps = std::numeric_limits<Float>::epsilon();
  const Point LowerCorner = {{0}};
  const Point Lengths = {{1}};
  const std::size_t NumLevels = BlockCode::MaxLevels;
  BlockCode b(LowerCorner, Lengths);
  REQUIRE(b.lowerCorner() == LowerCorner);
  REQUIRE(b.lengths() == Lengths);
  REQUIRE(b.numLevels() == NumLevels);
  REQUIRE(b.code(Point{{0}}) == Code(b.numLevels()));
  for (std::size_t i = 0; i <= NumLevels; ++i) {
    REQUIRE(b.level(i) == i);
  }
}


TEST_CASE("2-D, 3 levels", "[BlockCode]") {
  typedef sfc::Traits<2> Traits;
  typedef sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Code Code;
  typedef BlockCode::Float Float;
  typedef BlockCode::Point Point;
  typedef BlockCode::Code Code;
  const Float Eps = std::numeric_limits<Float>::epsilon();
  const Point LowerCorner = {{0, 0}};
  const Point Lengths = {{1, 1}};
  const std::size_t NumLevels = 3;
  BlockCode b(LowerCorner, Lengths, NumLevels);
  REQUIRE(b.lowerCorner() == LowerCorner);
  REQUIRE(b.lengths() == Lengths);
  REQUIRE(b.numLevels() == NumLevels);
  REQUIRE(b.levelBits() == 2);

  REQUIRE(b.code(Point{{0, 0}}) == Code(0x3));// 00000011
  REQUIRE(b.code(Point{{0.5, 0.5}}) == Code(0xC3));// 1100 0011
  REQUIRE(b.code(Point{{1 - Eps, 1 - Eps}}) ==
          Code(0xFF));// 11111111

  REQUIRE(b.isValid(0x0)); // 000000 00
  REQUIRE_FALSE(b.isValid(0x4)); // 000001 00
  REQUIRE_FALSE(b.isValid(0x8)); // 000010 00
  REQUIRE_FALSE(b.isValid(0x10)); // 000100 00
  REQUIRE_FALSE(b.isValid(0x20)); // 001000 00
  REQUIRE_FALSE(b.isValid(0x40)); // 010000 00
  REQUIRE_FALSE(b.isValid(0x80)); // 100000 00

  REQUIRE(b.isValid(0x1)); // 000000 01
  REQUIRE(b.isValid(0x41)); // 010000 01
  REQUIRE(b.isValid(0x81)); // 100000 01
  REQUIRE_FALSE(b.isValid(0x21)); // 001000 01

  REQUIRE(b.isValid(0xFF)); //  111111 11
  REQUIRE_FALSE(b.isValid(0x100)); // 1 000000 00

  for (Code i = 0; i <= NumLevels; ++i) {
    REQUIRE(b.level(i) == i);
    REQUIRE(b.level(i + 0x4) == i);
  }

  // 111111 11 -> 111111 11
  REQUIRE(b.atLevel(0xFF, 3) == Code(0xFF));
  // 111111 11 -> 111100 10
  REQUIRE(b.atLevel(0xFF, 2) == Code(0xF2));
  // 111111 11 -> 110000 01
  REQUIRE(b.atLevel(0xFF, 1) == Code(0xC1));
  // 111111 11 -> 000000 00
  REQUIRE(b.atLevel(0xFF, 0) == Code(0x00));
  for (std::size_t i = 0; i <= NumLevels; ++i) {
    REQUIRE(b.atLevel(0x00, i) == i);
  }

  REQUIRE(b.next(0x0) == Code(0x100)); // 000000 00 -> 1 000000 00
  REQUIRE(b.next(0x1) == Code(0x41)); // 000000 01 -> 010000 01
  REQUIRE(b.next(0x41) == Code(0x81)); // 010000 01 -> 100000 01
  REQUIRE(b.next(0x81) == Code(0xC1)); // 100000 01 -> 110000 01
  REQUIRE(b.next(0xC1) == Code(0x101)); // 110000 01 -> 1 000000 01
}


TEST_CASE("Tight bounding box constructor. 1-D, 0 levels.", "[BlockCode]") {
  typedef sfc::Traits<1> Traits;
  typedef sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Float Float;
  BlockCode b(geom::BBox<Float, Traits::Dimension>{{{0}}, {{1}}}, 1);
  REQUIRE(-0.1 < b.lowerCorner()[0]);
  REQUIRE(b.lowerCorner()[0] < 0);
  REQUIRE(1 < b.lengths()[0]);
  REQUIRE(b.lengths()[0] < 1.1);
  REQUIRE(b.numLevels() == 0);
  REQUIRE(b.levelBits() == 0);
}

TEST_CASE("Tight bounding box constructor. 2-D, 3 levels.", "[BlockCode]") {
  typedef sfc::Traits<2> Traits;
  typedef sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Float Float;
  BlockCode b(geom::BBox<Float, Traits::Dimension>{{{0, 0}}, {{1, 1}}},
              0.125);
  REQUIRE(-0.1 < b.lowerCorner()[0]);
  REQUIRE(b.lowerCorner()[0] < 0);
  REQUIRE(1 < b.lengths()[0]);
  REQUIRE(b.lengths()[0] < 1.1);
  REQUIRE(b.numLevels() == 3);
  REQUIRE(b.levelBits() == 2);
}


TEST_CASE("Equality.", "[BlockCode]") {
  typedef sfc::Traits<1> Traits;
  typedef sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Point Point;
  const BlockCode x(Point{{0}}, Point{{1}}, 7);
  {
    const BlockCode y(Point{{0}}, Point{{1}}, 7);
    REQUIRE(x == y);
  }
  {
    const BlockCode y(Point{{1}}, Point{{1}}, 7);
    REQUIRE(!(x == y));
  }
  {
    const BlockCode y(Point{{0}}, Point{{2}}, 7);
    REQUIRE(!(x == y));
  }
  {
    const BlockCode y(Point{{0}}, Point{{1}}, 8);
    REQUIRE(!(x == y));
  }
}
