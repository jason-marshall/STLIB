// -*- C++ -*-

#include "stlib/geom/kernel/SimplexMp.h"

#include <cassert>

template<typename _FloatCoord, typename _FloatOffset, std::size_t _D,
         std::size_t _N>
void
test()
{
  using SimplexMp = stlib::geom::SimplexMp
    <_FloatCoord, _FloatOffset, _D, _N>;
  static_assert(std::is_pod<SimplexMp>::value,
                "SimplexMp is a POD type.");
  using Point = typename SimplexMp::Point;
  using Simplex = std::array<Point, _N + 1>;
  using ExtremePoints = stlib::geom::ExtremePoints<_FloatCoord, _D>;
  using BBox = stlib::geom::BBox<_FloatCoord, _D>;

  using stlib::geom::toMixedPrecision;
  using stlib::geom::toUniformPrecision;

  Simplex s;
  for (std::size_t i = 0; i != _N + 1; ++i) {
    for (std::size_t j = 0; j != _D; ++j) {
      s[i][j] = i + j;
    }
  }
  // Convert to the vertex offsets representation and back.
  SimplexMp const svo = toMixedPrecision<_FloatOffset>(s);
  assert(toUniformPrecision(svo) == s);

  // Empty vectors.
  {
    std::vector<Simplex> const a;
    std::vector<SimplexMp> const b = toMixedPrecision<_FloatOffset>(a);
    assert(toUniformPrecision(b) == a);
  }
  // Non-empty vectors.
  {
    std::vector<Simplex> const a(3, s);
    std::vector<SimplexMp> const b = toMixedPrecision<_FloatOffset>(a);
    assert(toUniformPrecision(b) == a);
  }

  // Build a bounding box around the SimplexMp.
  {
    BBox const box = stlib::geom::specificBBox<BBox>(svo);
    for (std::size_t i = 0; i != s.size(); ++i) {
      assert(stlib::geom::isInside(box, s[i]));
    }
  }

  // Build an extreme points bounding structure around the SimplexMp.
  {
    BBox const box =
      stlib::geom::specificBBox<BBox>
      (stlib::geom::extremePoints<ExtremePoints>(svo));
    for (std::size_t i = 0; i != s.size(); ++i) {
      assert(stlib::geom::isInside(box, s[i]));
    }
  }
}

template<typename _FloatCoord, typename _FloatOffset>
void
testDimensions()
{
  test<_FloatCoord, _FloatOffset, 0, 0>();
  test<_FloatCoord, _FloatOffset, 0, 1>();
  test<_FloatCoord, _FloatOffset, 1, 0>();
  test<_FloatCoord, _FloatOffset, 1, 1>();
  test<_FloatCoord, _FloatOffset, 2, 0>();
  test<_FloatCoord, _FloatOffset, 2, 1>();
  test<_FloatCoord, _FloatOffset, 2, 2>();
  test<_FloatCoord, _FloatOffset, 3, 0>();
  test<_FloatCoord, _FloatOffset, 3, 1>();
  test<_FloatCoord, _FloatOffset, 3, 2>();
  test<_FloatCoord, _FloatOffset, 3, 3>();
}

int
main()
{
  testDimensions<float, float>();
  testDimensions<float, double>();
  testDimensions<double, float>();
  testDimensions<double, double>();
  
  return 0;
}
