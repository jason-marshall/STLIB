// -*- C++ -*-

#include "stlib/sfc/MortonOrder.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace stlib;

TEST_CASE("MortonOrder.", "[MortonOrder]")
{
  SECTION("1-D. 8 bits.") {
    std::size_t const Dimension = 1;
    typedef std::uint8_t Code;
    typedef std::array<Code, Dimension> DiscretePoint;
    sfc::MortonOrder<Dimension, Code> order;
    for (std::size_t numLevels = 0;
         numLevels <= std::numeric_limits<Code>::digits;
         ++numLevels) {
      std::size_t const MaxIndex = (std::size_t(1) << numLevels) - 1;
      for (std::size_t i = 0; i <= MaxIndex; ++i) {
        REQUIRE(order.code(DiscretePoint{{Code(i)}}, numLevels) ==
                i);
      }
    }
  }

  SECTION("1-D. 64 bits.") {
    std::size_t const Dimension = 1;
    typedef std::uint64_t Code;
    typedef std::array<Code, Dimension> DiscretePoint;
    sfc::MortonOrder<Dimension, Code> order;
    REQUIRE(order.code(DiscretePoint{{0}}, 0) == Code(0));
    REQUIRE(order.code(DiscretePoint{{0}},
                       std::numeric_limits<Code>::digits) ==
            Code(0));
    REQUIRE(order.code(DiscretePoint
                       {{std::numeric_limits<Code>::max()}},
                       std::numeric_limits<Code>::digits) ==
            std::numeric_limits<Code>::max());
  }

  SECTION("2-D. 64 bits.") {
    std::size_t const Dimension = 2;
    typedef std::uint64_t Code;
    typedef std::array<Code, Dimension> DiscretePoint;
    sfc::MortonOrder<Dimension, Code> order;
    // Check that the lower corner maps to zero.
    REQUIRE(order.code((DiscretePoint{{0, 0}}), 0) == Code(0));
    REQUIRE(order.code((DiscretePoint{{0, 0}}),
                       std::numeric_limits<Code>::digits /
                       Dimension) == Code(0));
    // Check that the codes are distinct.
    std::size_t const NumLevels = 4;
    std::size_t const Extent = std::size_t(1) << NumLevels;
    std::vector<Code> codes;
    codes.reserve(Extent * Extent);
    for (std::size_t j = 0; j != Extent; ++j) {
      for (std::size_t i = 0; i != Extent; ++i) {
        codes.push_back(order.code(DiscretePoint{{i, j}}, NumLevels));
      }
    }
    std::sort(codes.begin(), codes.end());
    for (std::size_t i = 0; i != codes.size(); ++i) {
      REQUIRE(codes[i] == i);
    }
  }

  SECTION("3-D. 64 bits.") {
    std::size_t const Dimension = 3;
    typedef std::uint64_t Code;
    typedef std::array<Code, Dimension> DiscretePoint;
    sfc::MortonOrder<Dimension, Code> order;
    // Check that the lower corner maps to zero.
    REQUIRE(order.code((DiscretePoint{{0, 0, 0}}), 0) == Code(0));
    REQUIRE(order.code((DiscretePoint{{0, 0, 0}}),
                       std::numeric_limits<Code>::digits /
                       Dimension) == Code(0));
    // Check that the codes are distinct.
    std::size_t const NumLevels = 4;
    std::size_t const Extent = std::size_t(1) << NumLevels;
    std::vector<Code> codes;
    codes.reserve(Extent * Extent * Extent);
    for (std::size_t k = 0; k != Extent; ++k) {
      for (std::size_t j = 0; j != Extent; ++j) {
        for (std::size_t i = 0; i != Extent; ++i) {
          codes.push_back(order.code(DiscretePoint{{i, j, k}}, NumLevels));
        }
      }
    }
    std::sort(codes.begin(), codes.end());
    for (std::size_t i = 0; i != codes.size(); ++i) {
      REQUIRE(codes[i] == i);
    }
  }
}
