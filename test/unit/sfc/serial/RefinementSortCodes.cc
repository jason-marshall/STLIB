// -*- C++ -*-

#include "stlib/sfc/RefinementSortCodes.h"
#include "stlib/sfc/AdaptiveCells.h"

using namespace stlib;

int
main()
{
  sfc::Traits<1>::Code const Guard = sfc::Traits<1>::GuardCode;

  // 1-D, 0 levels
  {
    typedef sfc::Traits<1> Traits;
    typedef Traits::Code Code;
    typedef sfc::BlockCode<Traits> BlockCode;
    typedef BlockCode::Point Point;

    BlockCode blockCode(Point{{0}}, Point{{1}}, 0);
    {
      std::vector<Code> codes;
      std::size_t const highestLevel =
        sfc::refinementSortCodes(blockCode, &codes, 0);
      assert(highestLevel == 0);
      assert(codes == (std::vector<Code>{}));
    }
    {
      std::vector<Code> original(10, blockCode.code(Point{{0}}));
      std::vector<Code> codes(original);
      std::size_t const highestLevel =
        sfc::refinementSortCodes(blockCode, &codes, 1);
      assert(highestLevel == 0);
      assert(codes == original);
    }
  }
  // 1-D, 1 level
  {
    typedef sfc::Traits<1> Traits;
    typedef sfc::BlockCode<Traits> BlockCode;
    typedef BlockCode::Code Code;
    typedef BlockCode::Point Point;

    BlockCode blockCode(Point{{0}}, Point{{1}}, 1);
    {
      std::vector<Code> codes;
      std::size_t const highestLevel =
        sfc::refinementSortCodes(blockCode, &codes, 0);
      assert(highestLevel == 0);
      assert(codes == (std::vector<Code>{}));
    }
    // Put all of the points at the origin.
    {
      std::vector<Code> original(10, blockCode.code(Point{{0}}));
      std::vector<Code> codes(original);
      {
        std::size_t const highestLevel =
          sfc::refinementSortCodes(blockCode, &codes, 1);
        assert(highestLevel == blockCode.numLevels());
        assert(codes == original);
      }
      {
        std::size_t const highestLevel =
          sfc::refinementSortCodes(blockCode, &codes, codes.size());
        assert(highestLevel == 0);
        assert(codes == original);
      }
    }
    // Put all of the points at the midpoint.
    {
      Code const code = blockCode.code(Point{{0.5}});
      assert(code == 3);
      std::vector<Code> original(10, code);
      std::vector<Code> codes(original);
      {
        std::size_t const highestLevel =
          sfc::refinementSortCodes(blockCode, &codes, 1);
        assert(highestLevel == blockCode.numLevels());
        assert(codes == original);
      }
      {
        std::size_t const highestLevel =
          sfc::refinementSortCodes(blockCode, &codes, codes.size());
        assert(highestLevel == 0);
        assert(codes == original);
      }
    }
  }

  // 1-D, 4 levels
  {
    typedef sfc::Traits<1> Traits;
    typedef Traits::Code Code;
    typedef sfc::BlockCode<Traits> BlockCode;
    typedef BlockCode::Point Point;
    typedef BlockCode::Float Float;

    BlockCode blockCode(Point{{0}}, Point{{1}}, 4);
    std::vector<Code> original;
    {
      std::size_t const Extent = 1 << blockCode.numLevels();
      for (std::size_t i = 0; i != Extent; ++i) {
        original.push_back(blockCode.code(Point{{Float(i) / Extent}}));
      }
    }
    std::vector<Code> codes(original);
    std::size_t const highestLevel =
      sfc::refinementSortCodes(blockCode, &codes, 1);
    assert(highestLevel == blockCode.numLevels());
    assert(codes.size() == original.size());
    assert(std::is_sorted(codes.begin(), codes.end()));
  }

  // 3-D, 0 levels
  {
    typedef sfc::Traits<3> Traits;
    typedef Traits::Code Code;
    typedef sfc::BlockCode<Traits> BlockCode;
    typedef BlockCode::Point Point;

    BlockCode blockCode(Point{{0, 0, 0}}, Point{{1, 1, 1}}, 0);
    {
      std::vector<Code> codes;
      std::size_t const highestLevel =
        sfc::refinementSortCodes(blockCode, &codes, 0);
      assert(highestLevel == 0);
      assert(codes == (std::vector<Code>{}));
    }
    {
      std::vector<Code>
        original(10, blockCode.code(Point{{0, 0, 0}}));
      std::vector<Code> codes(original);
      std::size_t const highestLevel =
        sfc::refinementSortCodes(blockCode, &codes, 1);
      assert(highestLevel == 0);
      assert(codes == original);
    }
  }

  // 3-D, 4 levels
  {
    typedef sfc::Traits<3> Traits;
    typedef Traits::Code Code;
    typedef sfc::BlockCode<Traits> BlockCode;
    typedef BlockCode::Point Point;
    typedef BlockCode::Float Float;

    BlockCode blockCode(Point{{0, 0, 0}}, Point{{1, 1, 1}}, 4);
    {
      std::vector<Code> original;
      {
        std::size_t const Extent = 1 << blockCode.numLevels();
        for (std::size_t k = 0; k != Extent; ++k) {
          for (std::size_t j = 0; j != Extent; ++j) {
            for (std::size_t i = 0; i != Extent; ++i) {
              original.push_back(blockCode.code(
                                   Point{{Float(i) / Extent,
                                         Float(j) / Extent,
                                         Float(k) / Extent}}));
            }
          }
        }
      }
      for (std::size_t n = 0; n <= 4; ++n) {
        std::vector<Code> codes(original);
        std::size_t const maxElementsPerCell = std::size_t(1) << (3 * n);
        std::size_t const highestLevel =
          sfc::refinementSortCodes(blockCode, &codes, maxElementsPerCell);
        assert(highestLevel == 4 - n);
        assert(codes.size() == original.size());
        assert(stlib::ext::sum(codes) == stlib::ext::sum(original));
      }
    }
  }

  // 1-D, 15 levels.
  {
    typedef stlib::sfc::Traits<1> Traits;
    typedef stlib::sfc::BlockCode<Traits> BlockCode;
    typedef BlockCode::Code Code;
    typedef BlockCode::Point Point;
    std::size_t const NumLevels = 15;
    BlockCode blockCode(Point{{0}}, Point{{1}}, NumLevels);
    using stlib::sfc::refinementSortCodes;

    {
      std::vector<Code> codes;
      assert(refinementSortCodes(blockCode, &codes, 1) == 0);
    }
    {
      std::vector<Code> codes = {NumLevels};
      assert(refinementSortCodes(blockCode, &codes, 1) == 0);
    }
    {
      std::vector<Code> codes = {NumLevels, NumLevels};
      assert(refinementSortCodes(blockCode, &codes, 1) == NumLevels);
    }
    {
      std::vector<Code> codes = {NumLevels, NumLevels};
      assert(refinementSortCodes(blockCode, &codes, 2) == 0);
    }
    {
      std::vector<Code> codes = {0xF, 0x1F};
      assert(refinementSortCodes(blockCode, &codes, 1) == NumLevels);
    }
    {
      std::vector<Code> codes = {0xF, 0x2F};
      assert(refinementSortCodes(blockCode, &codes, 1) == NumLevels - 1);
    }
  }

  // objectCodesToCellCodeCountPairs()
  {
    typedef stlib::sfc::Traits<1> Traits;
    typedef Traits::Code Code;
    typedef std::pair<Code, std::size_t> Pair;

    {
      std::vector<Code> values;
      std::vector<Pair> pairs;
      stlib::sfc::objectCodesToCellCodeCountPairs<Traits>(values, &pairs);
      assert(pairs == (std::vector<Pair>{{Guard, 0}}));
    }
    {
      std::vector<Code> values = {0};
      std::vector<Pair> pairs;
      stlib::sfc::objectCodesToCellCodeCountPairs<Traits>(values, &pairs);
      std::vector<Pair> const output = {{0, 1}, {Guard, 0}};
      assert(pairs == output);
      stlib::sfc::objectCodesToCellCodeCountPairs<Traits>(values, &pairs);
      assert(pairs == output);
    }
    {
      std::vector<Code> values = {1, 2, 2, 3, 3, 3};
      std::vector<Pair> pairs;
      stlib::sfc::objectCodesToCellCodeCountPairs<Traits>(values, &pairs);
      std::vector<Pair> const output = {{1, 1}, {2, 2}, {3, 3}, {Guard, 0}};
      assert(pairs == output);
      stlib::sfc::objectCodesToCellCodeCountPairs<Traits>(values, &pairs);
      assert(pairs == output);
    }
  }
}
