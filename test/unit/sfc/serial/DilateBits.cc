// -*- C++ -*-

#include "stlib/sfc/DilateBits.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace stlib;

TEST_CASE("DilateBits.", "[DilateBits]")
{
  SECTION("1-D. Bits 1 through 8.") {
    sfc::DilateBits<1> dilate;
    for (std::size_t i = 0; i != 1 << 9; ++i) {
      REQUIRE(i == dilate(i, 9));
    }
  }
  SECTION("2-D.") {
    sfc::DilateBits<2> dilate;
    REQUIRE(dilate(std::uint8_t(0), 8) == 0);
    for (std::size_t i = 0; i != 4; ++i) {
      REQUIRE(dilate(std::uint8_t(1) << i, 4) == uint8_t(1) << 2 * i);
    }
    for (std::size_t i = 0; i != 32; ++i) {
      REQUIRE(dilate(std::uint64_t(1) << i, 32) == uint64_t(1) << 2 * i);
    }
  }
  SECTION("3-D.") {
    const std::size_t D = 3;
    sfc::DilateBits<D> dilate;
    REQUIRE(dilate(std::uint8_t(0), 8) == 0);
    for (std::size_t i = 0; i != 2; ++i) {
      REQUIRE(dilate(std::uint8_t(1) << i, 2) == uint8_t(1) << D * i);
    }
    for (std::size_t i = 0; i != 21; ++i) {
      REQUIRE(dilate(std::uint64_t(1) << i, 21) == uint64_t(1) << D * i);
    }
  }
}
