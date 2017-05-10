// -*- C++ -*-

#include "stlib/geom/kernel/OrientedBBox.h"
#include "stlib/geom/kernel/BBoxDistance.h"
#include "stlib/simd/shuffle.h"


USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

template<typename _Float, std::size_t _D>
inline
void
checkAxesOrthonormal(geom::OrientedBBox<_Float, _D> const& x)
{
  _Float const Tol = 10 * _D * std::numeric_limits<_Float>::epsilon();
  for (std::size_t i = 0; i != _D; ++i) {
    assert(std::abs(stlib::ext::magnitude(x.axes[i]) - 1) < Tol);
  }
  for (std::size_t i = 0; i != _D; ++i) {
    for (std::size_t j = i + 1; j != _D; ++j) {
      assert(std::abs(stlib::ext::dot(x.axes[i], x.axes[j])) < Tol);
    }
  }
}


template<typename _Float, std::size_t _D>
inline
void
checkPointsInside(geom::OrientedBBox<_Float, _D> const& x,
                  std::vector<std::array<_Float, _D> > const& points)
{
  _Float const Tol = 10 * _D * std::numeric_limits<_Float>::epsilon();
  geom::BBox<_Float, _D> const b = x.bbox();
  // Check that the transformed points are inside the associated AABB.
  for (std::size_t i = 0; i != points.size(); ++i) {
    assert(minDist2(b, x.transform(points[i])) < Tol);
  }
}


template<typename _Float, std::size_t _D>
inline
void
checkDistance(geom::OrientedBBox<_Float, _D> const& x)
{
  _Float const Tol = 10 * _D * std::numeric_limits<_Float>::epsilon();
  geom::BBox<_Float, _D> const b = x.bbox();
  // Check that the corners of the OBB are inside the associated AABB.
  assert(minDist2(b, x.transform(x.center +
                                 x.radii[0] * x.axes[0] +
                                 x.radii[1] * x.axes[1] +
                                 x.radii[2] * x.axes[2])) < Tol);
  assert(minDist2(b, x.transform(x.center -
                                 x.radii[0] * x.axes[0] -
                                 x.radii[1] * x.axes[1] -
                                 x.radii[2] * x.axes[2])) < Tol);
  // Test points outside the faces.
  for (std::size_t i = 0; i != _D; ++i) {
    _Float const d2 =
      minDist2(b, x.transform(x.center +
                              (x.radii[i] + _Float(1)) * x.axes[i]));
    assert(std::abs(d2 - 1) < Tol);
  }
}


template<typename _Float, std::size_t _D>
inline
void
check(geom::OrientedBBox<_Float, _D> const& x,
      std::vector<std::array<_Float, _D> > const& points)
{
  checkAxesOrthonormal(x);
  checkPointsInside(x, points);
  checkDistance(x);
}


int
main()
{
  {
    typedef double Float;
    std::size_t const Dimension = 1;
    typedef geom::OrientedBBox<Float, Dimension> OrientedBBox;

    OrientedBBox x = {{{0}}, {{{{1}}}}, {{1}}};
    assert(x == x);
  }

  {
    typedef float Float;
    std::size_t const Dimension = 3;
    typedef geom::OrientedBBox<Float, Dimension> OrientedBBox;
    typedef OrientedBBox::Point Point;
    Float const Tol = 10 * Dimension * std::numeric_limits<Float>::epsilon();
    OrientedBBox x;

    {
      // Single point at the origin.
      std::vector<Point> points;
      points.push_back(Point{{0, 0, 0}});
      x.buildPca(points);
      check(x, points);
      assert(x.center == (Point{{0, 0, 0}}));
      assert(x.axes[0] == (Point{{1, 0, 0}}));
      assert(x.axes[1] == (Point{{0, 1, 0}}));
      assert(x.axes[2] == (Point{{0, 0, 1}}));
      assert(x.radii == (Point{{0, 0, 0}}));
      x.buildPcaRotate(points);
      check(x, points);
    }
    {
      // Single point not at the origin.
      std::vector<Point> points;
      points.push_back(Point{{1, 2, 3}});
      x.buildPca(points);
      check(x, points);
      assert(x.center == (Point{{1, 2, 3}}));
      assert(x.axes[0] == (Point{{1, 0, 0}}));
      assert(x.axes[1] == (Point{{0, 1, 0}}));
      assert(x.axes[2] == (Point{{0, 0, 1}}));
      assert(x.radii == (Point{{0, 0, 0}}));
      x.buildPcaRotate(points);
      check(x, points);
    }
    {
      // Two points.
      std::vector<Point> points;
      points.push_back(Point{{1, 0, 0}});
      points.push_back(Point{{3, 0, 0}});
      x.buildPca(points);
      check(x, points);
      assert(x.center == (Point{{2, 0, 0}}));
      assert(x.axes[0] == (Point{{1, 0, 0}}));
      assert(x.axes[1] == (Point{{0, 1, 0}}));
      assert(x.axes[2] == (Point{{0, 0, 1}}));
      assert(x.radii == (Point{{1, 0, 0}}));
      x.buildPcaRotate(points);
      check(x, points);
    }
    {
      // Two distinct points.
      std::vector<Point> points;
      points.push_back(Point{{1, 0, 0}});
      points.push_back(Point{{1, 0, 0}});
      points.push_back(Point{{1, 0, 0}});
      points.push_back(Point{{3, 0, 0}});
      x.buildPca(points);
      check(x, points);
      assert(x.center == (Point{{2, 0, 0}}));
      assert(x.axes[0] == (Point{{1, 0, 0}}));
      assert(x.axes[1] == (Point{{0, 1, 0}}));
      assert(x.axes[2] == (Point{{0, 0, 1}}));
      assert(x.radii == (Point{{1, 0, 0}}));
      geom::BBox<Float, Dimension> const b = x.bbox();
      assert(b.lower == -x.radii);
      assert(b.upper == x.radii);
      assert(x.transform(x.center) == (Point{{0, 0, 0}}));
      assert(x.transform(x.center + (Point{{2, 0, 0}})) ==
             (Point{{2, 0, 0}}));
      x.buildPcaRotate(points);
      check(x, points);
    }
    {
      // Two points.
      std::vector<Point> points;
      points.push_back(Point{{0, 0, 0}});
      points.push_back(Point{{1, 1, 0}});
      x.buildPca(points);
      check(x, points);
      assert(x.center == (Point{{0.5, 0.5, 0}}));
      assert(stlib::ext::magnitude(x.axes[0] - (Point{{Float(-0.5 * std::sqrt(2)),
                  Float(-0.5 * std::sqrt(2)), 0}})) < Tol);
      checkAxesOrthonormal(x);
      assert(stlib::ext::magnitude(x.radii -
                       (Point{{Float(0.5 * std::sqrt(2)), 0, 0}}))
             < Tol);
      x.buildPcaRotate(points);
      check(x, points);
    }
    {
      // Four points.
      std::vector<Point> points;
      points.push_back(Point{{0, 0, 0}});
      points.push_back(Point{{4, 0, 0}});
      points.push_back(Point{{2, -1, 0}});
      points.push_back(Point{{2, 1, 0}});
      x.buildPca(points);
      check(x, points);
      assert(x.center == (Point{{2, 0, 0}}));
      assert(x.axes[0] == (Point{{1, 0, 0}}));
      assert(x.axes[1] == (Point{{0, 1, 0}}));
      assert(x.axes[2] == (Point{{0, 0, 1}}));
      assert(x.radii == (Point{{2, 1, 0}}));
      x.buildPcaRotate(points);
      check(x, points);
    }
    {
      // Random points.
      std::vector<Point> points;
      Float const scaling = 1. / RAND_MAX;
      for (std::size_t i = 0; i != 10; ++i) {
        points.push_back(Point{{scaling * rand(), scaling * rand(),
                scaling * rand()}});
      }
      x.buildPca(points);
      check(x, points);

      // Convert the points to hybrid SoA format.
      std::vector<Float, simd::allocator<Float> > shuffled;
      simd::aosToHybridSoa(points, &shuffled);
      // Transform the points.
      std::vector<Float, simd::allocator<Float> > transformedData;
      x.transform(shuffled, &transformedData);
      // Convert back to regular AoS.
      simd::hybridSoaToAos<Dimension>(&transformedData);
      std::vector<Point> transformed(points.size());
      memcpy(&transformed[0][0], &transformedData[0],
             transformed.size() * sizeof(Point));
      // Compare to the points transformed one at a time.
      for (std::size_t i = 0; i != points.size(); ++i) {
        assert(stlib::ext::magnitude(x.transform(points[i]) - transformed[i])
               < Tol);
      }

      x.buildPcaRotate(points);
      check(x, points);
    }
  }

  return 0;
}
