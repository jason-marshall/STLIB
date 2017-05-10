// -*- C++ -*-

#include "stlib/geom/kernel/BBoxDistance.h"

#include <cassert>


using namespace stlib;

template<typename _Float, std::size_t _D>
void
testNxnDist2(const geom::BBox<_Float, _D>& a, const geom::BBox<_Float, _D>& b)
{
  const _Float nxnD2 = geom::nxnDist2(a, b);
  assert(geom::minMinDist2(a, b) <= nxnD2);
  assert(nxnD2 <= geom::maxMaxDist2(a, b));
}

int
main()
{
  {
    using geom::minDist;
    assert(minDist(0, 0, 0, 0) == 0);

    assert(minDist(0, 0, 1, 1) == 1);
    assert(minDist(1, 1, 0, 0) == 1);

    assert(minDist(0, 1, 2, 3) == 1);
    assert(minDist(2, 3, 0, 1) == 1);

    assert(minDist(0, 1, 1, 2) == 0);
    assert(minDist(1, 2, 0, 1) == 0);

    assert(minDist(0, 3, 1, 2) == 0);
    assert(minDist(1, 2, 0, 3) == 0);
  }
  {
    using geom::maxDist;
    assert(maxDist(0, 0, 0, 0) == 0);

    assert(maxDist(0, 0, 1, 1) == 1);
    assert(maxDist(1, 1, 0, 0) == 1);

    assert(maxDist(0, 1, 2, 3) == 3);
    assert(maxDist(2, 3, 0, 1) == 3);

    assert(maxDist(0, 1, 1, 2) == 2);
    assert(maxDist(1, 2, 0, 1) == 2);

    assert(maxDist(0, 3, 1, 2) == 2);
    assert(maxDist(1, 2, 0, 3) == 2);
  }
  {
    using geom::maxMinDist;
    assert(maxMinDist(0, 0, 0, 0) == 0);

    assert(maxMinDist(0, 2, 3, 5) == 3);
    assert(maxMinDist(0, 2, 2, 4) == 2);
    assert(maxMinDist(0, 2, 1, 3) == 1);
    assert(maxMinDist(0, 2, 0, 2) == 1);
    assert(maxMinDist(0, 2, -1, 1) == 1);
    assert(maxMinDist(0, 2, -2, 0) == 2);
    assert(maxMinDist(0, 2, -3, -1) == 3);

    assert(maxMinDist(0, 4, 5, 7) == 5);
    assert(maxMinDist(0, 4, 4, 6) == 4);
    assert(maxMinDist(0, 4, 3, 5) == 3);
    assert(maxMinDist(0, 4, 2, 4) == 2);
    assert(maxMinDist(0, 4, 1, 3) == 1);
    assert(maxMinDist(0, 4, 0, 2) == 2);
    assert(maxMinDist(0, 4, -1, 1) == 3);
    assert(maxMinDist(0, 4, -2, 0) == 4);
    assert(maxMinDist(0, 4, -3, -1) == 5);

    assert(maxMinDist(0, 2, 3, 7) == 3);
    assert(maxMinDist(0, 2, 2, 6) == 2);
    assert(maxMinDist(0, 2, 1, 5) == 1);
    assert(maxMinDist(0, 2, 0, 4) == 2);
    assert(maxMinDist(0, 2, -1, 3) == 2);
    assert(maxMinDist(0, 2, -2, 2) == 2);
    assert(maxMinDist(0, 2, -3, 1) == 1);
    assert(maxMinDist(0, 2, -4, 0) == 2);
    assert(maxMinDist(0, 2, -5, -1) == 3);
  }
  {
    using geom::minMinDist2;

    typedef double Float;
    const std::size_t D = 3;
    typedef geom::BBox<Float, D> BBox;
    assert(minMinDist2(BBox{
      {{
          0, 0, 0
        }
      }, {{1, 1, 1}}
    },
    BBox{
      {{
          2, 2, 2
        }
      }, {{3, 3, 3}}
    }) == 3);
    assert(minMinDist2(BBox{
      {{
          0, 0, 0
        }
      }, {{2, 1, 1}}
    },
    BBox{
      {{
          2, 2, 2
        }
      }, {{3, 3, 3}}
    }) == 2);
    assert(minMinDist2(BBox{
      {{
          0, 0, 0
        }
      }, {{2, 2, 1}}
    },
    BBox{
      {{
          2, 2, 2
        }
      }, {{3, 3, 3}}
    }) == 1);
    assert(minMinDist2(BBox{
      {{
          0, 0, 0
        }
      }, {{2, 2, 2}}
    },
    BBox{
      {{
          2, 2, 2
        }
      }, {{3, 3, 3}}
    }) == 0);
    assert(minMinDist2(BBox{
      {{
          0, 0, 0
        }
      }, {{4, 4, 4}}
    },
    BBox{
      {{
          2, 2, 2
        }
      }, {{3, 3, 3}}
    }) == 0);
  }
  {
    using geom::maxMaxDist2;

    typedef double Float;
    const std::size_t D = 2;
    typedef geom::BBox<Float, D> BBox;
    {
      const BBox a = {{{0, 0}}, {{1, 1}}};
      const BBox b = {{{2, 2}}, {{3, 3}}};
      assert(maxMaxDist2(a, b) == 9 + 9);
      assert(maxMaxDist2(b, a) == maxMaxDist2(a, b));
    }
    {
      const BBox a = {{{0, 0}}, {{1, 1}}};
      const BBox b = {{{0, 0}}, {{2, 3}}};
      assert(maxMaxDist2(a, b) == 4 + 9);
      assert(maxMaxDist2(b, a) == maxMaxDist2(a, b));
    }
    {
      const BBox a = {{{0, 0}}, {{5, 7}}};
      const BBox b = {{{1, 1}}, {{2, 3}}};
      assert(maxMaxDist2(a, b) == 4 * 4 + 6 * 6);
      assert(maxMaxDist2(b, a) == maxMaxDist2(a, b));
    }
  }

  {
    using geom::nxnDist2;

    typedef double Float;
    const std::size_t D = 2;
    typedef geom::BBox<Float, D> BBox;
    {
      const BBox a = {{{0, 0}}, {{0, 0}}};
      const BBox b = {{{0, 0}}, {{0, 0}}};
      assert(nxnDist2(a, b) == 0);
      testNxnDist2(a, b);
    }
    {
      const BBox a = {{{0, 0}}, {{0, 0}}};
      const BBox b = {{{0, 0}}, {{1, 1}}};
      assert(nxnDist2(a, b) == 1);
      testNxnDist2(a, b);
    }
    {
      const BBox a = {{{0, 0}}, {{0, 0}}};
      const BBox b = {{{1, 0}}, {{2, 1}}};
      assert(nxnDist2(a, b) == 2);
      testNxnDist2(a, b);
    }
    {
      const BBox a = {{{0, 0}}, {{0, 0}}};
      const BBox b = {{{1, 1}}, {{2, 2}}};
      assert(nxnDist2(a, b) == 5);
      testNxnDist2(a, b);
    }
    {
      const BBox a = {{{0, 0}}, {{0, 0}}};
      const BBox b = {{{0, 1}}, {{1, 2}}};
      assert(nxnDist2(a, b) == 2);
      testNxnDist2(a, b);
    }
    {
      const BBox a = {{{0, 0}}, {{0, 0}}};
      const BBox b = {{{ -1, 1}}, {{0, 2}}};
      assert(nxnDist2(a, b) == 2);
      testNxnDist2(a, b);
    }
    {
      const BBox a = {{{0, 0}}, {{0, 0}}};
      const BBox b = {{{ -2, 1}}, {{ -1, 2}}};
      assert(nxnDist2(a, b) == 5);
      testNxnDist2(a, b);
    }

    {
      const BBox a = {{{0, 0}}, {{1, 1}}};
      const BBox b = {{{0, 0}}, {{1, 1}}};
      assert(nxnDist2(a, b) == 1.25);
      testNxnDist2(a, b);
    }
    {
      const BBox a = {{{0, 0}}, {{1, 1}}};
      const BBox b = {{{1, 0}}, {{2, 1}}};
      assert(nxnDist2(a, b) == 2);
      testNxnDist2(a, b);
    }
    {
      const BBox a = {{{0, 0}}, {{1, 1}}};
      const BBox b = {{{2, 0}}, {{3, 1}}};
      assert(nxnDist2(a, b) == 5);
      testNxnDist2(a, b);
    }
    {
      const BBox a = {{{0, 0}}, {{1, 1}}};
      const BBox b = {{{1, 1}}, {{2, 2}}};
      assert(nxnDist2(a, b) == 5);
      testNxnDist2(a, b);
    }
    {
      const BBox a = {{{0, 0}}, {{2, 2}}};
      const BBox b = {{{1, 1}}, {{3, 3}}};
      assert(nxnDist2(a, b) == 10);
      testNxnDist2(a, b);
    }
  }

  return 0;
}
