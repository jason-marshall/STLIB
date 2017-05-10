// -*- C++ -*-

#include "stlib/levelSet/outside.h"

using namespace stlib;

int
main()
{
  typedef float Number;
  typedef levelSet::Grid<Number, 3, 8> Grid;
  typedef Grid::BBox BBox;
  typedef Grid::IndexList IndexList;

  // setSign()
  {
    Grid grid(BBox{{{0, 0, 0}}, {{1, 1, 1}}}, IndexList{{1, 1, 1}});

    const IndexList byteExtents = {{sizeof(std::size_t), 8, 8}};
    const IndexList integerExtents = {{
        byteExtents[0] / sizeof(std::size_t),
        byteExtents[1], byteExtents[2]
      }
    };
    std::vector<std::size_t> sign(stlib::ext::product(integerExtents));

    // Unrefined, negative.
    grid[0].fillValue = -1;
    std::fill(sign.begin(), sign.end(),
              std::numeric_limits<std::size_t>::max());
    setSign(grid, byteExtents, &sign);
    for (std::size_t i = 0; i != sign.size(); ++i) {
      assert((unsigned char)(sign[i]) == 0);
    }

    // Unrefined, positive.
    grid[0].fillValue = 1;
    std::fill(sign.begin(), sign.end(),
              std::numeric_limits<std::size_t>::max());
    setSign(grid, byteExtents, &sign);
    for (std::size_t i = 0; i != sign.size(); ++i) {
      assert((unsigned char)(sign[i]) == 0xFF);
    }

    // Refine.
    {
      std::vector<std::size_t> indices(1, 0);
      grid.refine(indices);
    }

    // Negative.
    std::fill(grid[0].begin(), grid[0].end(), Number(-1));
    std::fill(sign.begin(), sign.end(),
              std::numeric_limits<std::size_t>::max());
    setSign(grid, byteExtents, &sign);
    for (std::size_t i = 0; i != sign.size(); ++i) {
      assert((unsigned char)(sign[i]) == 0);
    }

    // Positive.
    std::fill(grid[0].begin(), grid[0].end(), Number(1));
    std::fill(sign.begin(), sign.end(),
              std::numeric_limits<std::size_t>::max());
    setSign(grid, byteExtents, &sign);
    for (std::size_t i = 0; i != sign.size(); ++i) {
      assert((unsigned char)(sign[i]) == 0xFF);
    }
  }

  // setBoundaryCondition()
  {
    // std::size_t extents.
    const IndexList Extents = {{1, 8, 8}};

    // Inside.
    {
      std::vector<std::size_t> sign(stlib::ext::product(Extents),
                                    std::numeric_limits<std::size_t>::max());
      std::vector<std::size_t> outside(sign.size(), 0);

      levelSet::setBoundaryCondition(Extents, sign, &outside);
      for (std::size_t k = 0; k != Extents[2]; ++k) {
        for (std::size_t j = 0; j != Extents[1]; ++j) {
          if (0 < j && j < Extents[1] - 1 &&
              0 < k && k < Extents[2] - 1) {
            assert(outside[j + k * Extents[1]] == 0);
          }
          else {
            assert(outside[j + k * Extents[1]] ==
                   std::numeric_limits<std::size_t>::max());
          }
        }
      }
    }

    // Outside.
    {
      std::vector<std::size_t> sign(stlib::ext::product(Extents),
                                    std::numeric_limits<std::size_t>::max());
      std::vector<std::size_t> outside(sign.size(), 0);

      levelSet::setBoundaryCondition(Extents, sign, &outside);
      for (std::size_t k = 0; k != Extents[2]; ++k) {
        for (std::size_t j = 0; j != Extents[1]; ++j) {
          if (0 < j && j < Extents[1] - 1 &&
              0 < k && k < Extents[2] - 1) {
            assert(outside[j + k * Extents[1]] == 0);
          }
          else {
            assert(outside[j + k * Extents[1]] ==
                   std::numeric_limits<std::size_t>::max());
          }
        }
      }
    }
  }

  // sweepXLocal()
  {
    // Single integer.
    {
      const std::size_t sign = 0;
      std::size_t outside = 0;
      levelSet::sweepXLocal(1, &sign, &outside);
      assert(outside == 0);
    }
    {
      const std::size_t sign = 1;
      std::size_t outside = 1;
      levelSet::sweepXLocal(1, &sign, &outside);
      assert(outside == 1);
    }
    {
      const std::size_t sign = 3;
      std::size_t outside = 1;
      levelSet::sweepXLocal(1, &sign, &outside);
      assert(outside == 3);
    }
    {
      const std::size_t sign = 3;
      std::size_t outside = 2;
      levelSet::sweepXLocal(1, &sign, &outside);
      assert(outside == 3);
    }
    {
      const std::size_t sign = 5;
      std::size_t outside = 1;
      levelSet::sweepXLocal(1, &sign, &outside);
      assert(outside == 1);
    }
    // Two integers.
    const int MaxShift = std::numeric_limits<std::size_t>::digits - 1;
    const std::size_t HiBit = std::size_t(1) << MaxShift;
    {
      const std::array<std::size_t, 2> sign = {{0, 0}};
      std::array<std::size_t, 2> outside = {{0, 0}};
      levelSet::sweepXLocal(sign.size(), &sign[0], &outside[0]);
      assert(outside[0] == 0 && outside[1] == 0);
    }
    {
      const std::array<std::size_t, 2> sign = {{HiBit, 1}};
      std::array<std::size_t, 2> outside = {{HiBit, 0}};
      levelSet::sweepXLocal(sign.size(), &sign[0], &outside[0]);
      assert(outside[0] == HiBit && outside[1] == 1);
    }
    {
      const std::array<std::size_t, 2> sign = {{HiBit, 1}};
      std::array<std::size_t, 2> outside = {{0, 1}};
      levelSet::sweepXLocal(sign.size(), &sign[0], &outside[0]);
      assert(outside[0] == HiBit && outside[1] == 1);
    }
    // Three integers.
    {
      const std::array<std::size_t, 3> sign = {{0, 0, 0}};
      std::array<std::size_t, 3> outside = {{0, 0, 0}};
      levelSet::sweepXLocal(sign.size(), &sign[0], &outside[0]);
      assert(outside[0] == 0 && outside[1] == 0);
    }
    {
      const std::array<std::size_t, 3> sign = {{HiBit, 1, 0}};
      std::array<std::size_t, 3> outside = {{HiBit, 0, 0}};
      levelSet::sweepXLocal(sign.size(), &sign[0], &outside[0]);
      assert(outside[0] == HiBit && outside[1] == 1);
    }
    {
      const std::array<std::size_t, 3> sign = {{HiBit, 1, 0}};
      std::array<std::size_t, 3> outside = {{0, 1, 0}};
      levelSet::sweepXLocal(sign.size(), &sign[0], &outside[0]);
      assert(outside[0] == HiBit && outside[1] == 1);
    }
    // vector.
    {
      const IndexList Extents = {{3, 1, 1}};
      std::vector<std::size_t> sign(3, 0), outside(3, 0);
      {
        sign[0] = 0;
        sign[1] = 0;
        outside[0] = 0;
        outside[1] = 0;
        levelSet::sweepXLocal(Extents, sign, &outside);
        assert(outside[0] == 0 && outside[1] == 0);
      }
      {
        sign[0] = HiBit;
        sign[1] = 1;
        outside[0] = HiBit;
        outside[1] = 0;
        levelSet::sweepXLocal(Extents, sign, &outside);
        assert(outside[0] == HiBit && outside[1] == 1);
      }
      {
        sign[0] = HiBit;
        sign[1] = 1;
        outside[0] = 0;
        outside[1] = 1;
        levelSet::sweepXLocal(Extents, sign, &outside);
        assert(outside[0] == HiBit && outside[1] == 1);
      }
    }
  }

  // sweepAdjacentRow()
  {
    // Single integer.
    {
      const std::size_t source = 0;
      const std::size_t sign = 0;
      std::size_t target = 0;
      levelSet::sweepAdjacentRow(1, &source, &sign, &target);
      assert(target == 0);
    }
    {
      const std::size_t source = 1;
      const std::size_t sign = 1;
      std::size_t target = 0;
      levelSet::sweepAdjacentRow(1, &source, &sign, &target);
      assert(target == 1);
    }
    {
      const std::size_t source = 1;
      const std::size_t sign = 2;
      std::size_t target = 0;
      levelSet::sweepAdjacentRow(1, &source, &sign, &target);
      assert(target == 2);
    }
    {
      const std::size_t source = 1;
      const std::size_t sign = 4;
      std::size_t target = 0;
      levelSet::sweepAdjacentRow(1, &source, &sign, &target);
      assert(target == 0);
    }
    {
      const std::size_t source = 2;
      const std::size_t sign = 1;
      std::size_t target = 0;
      levelSet::sweepAdjacentRow(1, &source, &sign, &target);
      assert(target == 1);
    }
    // CONTINUE
  }

  // sweepY()
  // CONTINUE

  // sweepZ()
  // CONTINUE

  // countBits()
  {
    std::vector<std::size_t> v;
    assert(levelSet::countBits(v) == 0);
    v.push_back(0);
    assert(levelSet::countBits(v) == 0);
    v[0] = 1;
    assert(levelSet::countBits(v) == 1);
    v[0] = 3;
    assert(levelSet::countBits(v) == 2);
  }

  // markOutside()
  // CONTINUE

  // markOutsideAsNegativeInf()
  // CONTINUE

  return 0;
}
