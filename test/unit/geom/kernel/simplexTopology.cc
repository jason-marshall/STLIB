// -*- C++ -*-

#include "stlib/geom/kernel/simplexTopology.h"

#include <cassert>

int
main()
{
  {
    using stlib::geom::computeOtherIndices;

    std::size_t a, b;
    computeOtherIndices(0, 1, &a, &b);
    assert(a == 2 && b == 3);
    computeOtherIndices(1, 0, &a, &b);
    assert(a == 2 && b == 3);
    computeOtherIndices(0, 2, &a, &b);
    assert(a == 1 && b == 3);
    computeOtherIndices(0, 3, &a, &b);
    assert(a == 1 && b == 2);
    computeOtherIndices(1, 2, &a, &b);
    assert(a == 0 && b == 3);
    computeOtherIndices(1, 3, &a, &b);
    assert(a == 0 && b == 2);
    computeOtherIndices(2, 3, &a, &b);
    assert(a == 0 && b == 1);
  }
  {
    using stlib::geom::simplexIndexedFace;

    assert(simplexIndexedFace<1>(0) == (std::array<std::size_t, 1>{{1}}));
    assert(simplexIndexedFace<1>(1) == (std::array<std::size_t, 1>{{0}}));

    assert(simplexIndexedFace<2>(0) == (std::array<std::size_t, 2>{{1, 2}}));
    assert(simplexIndexedFace<2>(1) == (std::array<std::size_t, 2>{{2, 0}}));
    assert(simplexIndexedFace<2>(2) == (std::array<std::size_t, 2>{{0, 1}}));

    assert(simplexIndexedFace<3>(0) == (std::array<std::size_t, 3>{{1, 2, 3}}));
    assert(simplexIndexedFace<3>(1) == (std::array<std::size_t, 3>{{2, 0, 3}}));
    assert(simplexIndexedFace<3>(2) == (std::array<std::size_t, 3>{{0, 1, 3}}));
    assert(simplexIndexedFace<3>(3) == (std::array<std::size_t, 3>{{1, 0, 2}}));
  }
  {
    // A 3-simplex.
    std::array<int, 3 + 1> s = {{0, 1, 2, 3}};

    // hasElement()
    assert(stlib::ext::hasElement(s, 0) &&
           stlib::ext::hasElement(s, 1) &&
           stlib::ext::hasElement(s, 2) &&
           stlib::ext::hasElement(s, 3) &&
           ! stlib::ext::hasElement(s, 4));

    {
      // hasFace()
      using stlib::geom::hasFace;

      for (int i = 0; i != 4; ++i) {
        assert(hasFace(s, std::array<int, 1>{{i}}));
      }
      assert(! hasFace(s, std::array<int, 1>{{4}}));

      for (int i = 0; i != 4; ++i) {
        for (int j = 0; j != 4; ++j) {
          if (i != j) {
            assert(hasFace(s, std::array<int, 2>{{i, j}}));
          }
        }
      }
      assert(! hasFace(s, std::array<int, 2>{{0, 4}}));

      for (int i = 0; i != 4; ++i) {
        for (int j = 0; j != 4; ++j) {
          if (i != j) {
            for (int k = 0; k != 4; ++k) {
              if (k != i && k != j) {
                assert(hasFace(s, std::array<int, 3>{{i, j, k}}));
              }
            }
          }
        }
      }
      assert(! hasFace(s, std::array<int, 3>{{0, 1, 4}}));

      assert(hasFace(s, std::array<int, 4>{{0, 1, 2, 3}}));
      assert(! hasFace(s, std::array<int, 4>{{0, 1, 2, 4}}));
    }

    // index
    for (std::size_t i = 0; i != 4; ++i) {
      assert(stlib::ext::index(s, i) == i);
    }

    // getFace
    for (std::size_t i = 0; i != 4; ++i) {
      assert(stlib::geom::hasFace(s, stlib::geom::getFace(s, i)));
    }

  }

  return 0;
}
