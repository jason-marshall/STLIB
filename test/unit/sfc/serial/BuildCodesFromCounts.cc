// -*- C++ -*-

#include "stlib/sfc/BuildCodesFromCounts.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"


TEST_CASE("1-D, 3 levels", "[BuildCodesFromCounts]") {
  typedef stlib::sfc::Traits<1> Traits;
  typedef stlib::sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Code Code;
  typedef std::pair<Code, std::size_t> Pair;
  typedef BlockCode::Point Point;
  using stlib::sfc::buildCodesFromCounts;

  Point const LowerCorner = {{0}};
  Point const Lengths = {{1}};
  std::size_t const NumLevels = 3;
  BlockCode blockCode(LowerCorner, Lengths, NumLevels);

  {
    std::vector<Pair> pairs;
    std::vector<Code> codes;
    buildCodesFromCounts(blockCode, pairs, 0, &codes);
    REQUIRE(codes.empty());
  }
  {
    std::vector<Pair> pairs = {{0, 1}};
    std::vector<Code> codes;
    buildCodesFromCounts(blockCode, pairs, 1, &codes);
    REQUIRE(codes == std::vector<Code>{0});
    buildCodesFromCounts(blockCode, pairs, 10, &codes);
    REQUIRE(codes == std::vector<Code>{0});
  }
  {
    std::vector<Pair> pairs = {{0, 10}};
    std::vector<Code> codes;
    buildCodesFromCounts(blockCode, pairs, 1, &codes);
    REQUIRE(codes == std::vector<Code>{0});
    buildCodesFromCounts(blockCode, pairs, 10, &codes);
    REQUIRE(codes == std::vector<Code>{0});
  }
  {
    std::vector<Pair> pairs = {{3, 1}}; // 000 11
    std::vector<Code> codes;
    buildCodesFromCounts(blockCode, pairs, 1, &codes);
    REQUIRE(codes == std::vector<Code>{0});
    buildCodesFromCounts(blockCode, pairs, 10, &codes);
    REQUIRE(codes == std::vector<Code>{0});
  }
  {
    std::vector<Pair> pairs = {{0x1F, 1}}; // 111 11
    std::vector<Code> codes;
    buildCodesFromCounts(blockCode, pairs, 1, &codes);
    REQUIRE(codes == std::vector<Code>{0});
    buildCodesFromCounts(blockCode, pairs, 10, &codes);
    REQUIRE(codes == std::vector<Code>{0});
  }

  {
    std::vector<Pair> pairs = {{0x1, 1},   // 000 01
                               {0x11, 1}}; // 100 01
    std::vector<Code> codes;
    buildCodesFromCounts(blockCode, pairs, 1, &codes);
    REQUIRE(codes == (std::vector<Code>{0x1, 0x11}));
    buildCodesFromCounts(blockCode, pairs, 2, &codes);
    REQUIRE(codes == std::vector<Code>{0});
  }
  {
    std::vector<Pair> pairs = {{0x1, 10},   // 000 01
                               {0x11, 10}}; // 100 01
    std::vector<Code> codes;
    buildCodesFromCounts(blockCode, pairs, 1, &codes);
    REQUIRE(codes == (std::vector<Code>{0x1, 0x11}));
    buildCodesFromCounts(blockCode, pairs, 10, &codes);
    REQUIRE(codes == (std::vector<Code>{0x1, 0x11}));
    buildCodesFromCounts(blockCode, pairs, 20, &codes);
    REQUIRE(codes == std::vector<Code>{0});
  }

  {
    std::vector<Pair> pairs = {{0x2, 1},   // 000 10
                               {0xA, 1},   // 010 10
                               {0x12, 1},  // 100 10
                               {0x1A, 1}}; // 110 10
    std::vector<Code> codes;

    buildCodesFromCounts(blockCode, pairs, 1, &codes);
    REQUIRE(codes == (std::vector<Code>{0x2, 0xA, 0x12, 0x1A}));

    buildCodesFromCounts(blockCode, pairs, 2, &codes);
    // 000 01, 100 01
    REQUIRE(codes == (std::vector<Code>{0x1, 0x11}));

    buildCodesFromCounts(blockCode, pairs, 3, &codes);
    // 000 01, 100 01
    REQUIRE(codes == (std::vector<Code>{0x1, 0x11}));

    buildCodesFromCounts(blockCode, pairs, 4, &codes);
    REQUIRE(codes == std::vector<Code>{0});
  }
  {
    std::vector<Pair> pairs = {{0x2, 1},   // 000 10
                               {0xA, 2},   // 010 10
                               {0x12, 4},  // 100 10
                               {0x1A, 8}}; // 110 10
    std::vector<Code> codes;

    buildCodesFromCounts(blockCode, pairs, 1, &codes);
    // 000 10, 010 10, 100 10, 110 10
    REQUIRE(codes == (std::vector<Code>{0x2, 0xA, 0x12, 0x1A}));

    buildCodesFromCounts(blockCode, pairs, 2, &codes);
    // 000 10, 010 10, 100 10, 110 10
    REQUIRE(codes == (std::vector<Code>{0x2, 0xA, 0x12, 0x1A}));

    buildCodesFromCounts(blockCode, pairs, 3, &codes);
    // 000 01, 100 10, 110 10
    REQUIRE(codes == (std::vector<Code>{0x1, 0x12, 0x1A}));

    buildCodesFromCounts(blockCode, pairs, 4, &codes);
    // 000 01, 100 10, 110 10
    REQUIRE(codes == (std::vector<Code>{0x1, 0x12, 0x1A}));

    buildCodesFromCounts(blockCode, pairs, 12, &codes);
    // 000 01, 100 01
    REQUIRE(codes == (std::vector<Code>{0x1, 0x11}));

    buildCodesFromCounts(blockCode, pairs, 15, &codes);
    // 000 00
    REQUIRE(codes == (std::vector<Code>{0x0}));
  }

  {
    std::vector<Pair> pairs = {{3, 1},  // 000 11
                               {7, 1}}; // 001 11
    std::vector<Code> codes;
    buildCodesFromCounts(blockCode, pairs, 1, &codes);
    REQUIRE(codes == (std::vector<Code>{3, 7}));
    buildCodesFromCounts(blockCode, pairs, 2, &codes);
    REQUIRE(codes == std::vector<Code>{0});
  }
  {
    std::vector<Pair> pairs = {{3, 1},  // 000 11
                               {0x1F, 1}}; // 111 11
    std::vector<Code> codes;
    buildCodesFromCounts(blockCode, pairs, 1, &codes);
    // 000 01, 100 01
    REQUIRE(codes == (std::vector<Code>{0x1, 0x11}));
    buildCodesFromCounts(blockCode, pairs, 2, &codes);
    REQUIRE(codes == std::vector<Code>{0});
  }
}


