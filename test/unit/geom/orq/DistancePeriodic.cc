// -*- C++ -*-

#include "stlib/geom/orq/DistancePeriodic.h"

using namespace stlib;

template<typename _Float>
void
test1()
{
  const std::size_t N = 1;
  typedef geom::DistancePeriodic<_Float, N> DistancePeriodic;
  typedef typename DistancePeriodic::Point Point;
  const _Float Eps = 10 * std::numeric_limits<_Float>::epsilon();

  // pointInDomain()
  {
    const geom::BBox<_Float, N> Domain = {{{0}}, {{1}}};
    const DistancePeriodic f(Domain);
    assert(f.pointInDomain(Point{{0}}) ==
           Point{{0}});
    assert(f.pointInDomain(Point{{1}}) ==
           Point{{0}});
    assert(f.pointInDomain(Point{{-1}}) ==
           Point{{0}});
    assert(f.pointInDomain(Point{{2}}) ==
           Point{{0}});
    assert(f.pointInDomain(Point{{0.5}}) ==
           Point{{0.5}});
    assert(f.pointInDomain(Point{{-0.5}}) ==
           Point{{0.5}});
    assert(f.pointInDomain(Point{{1.5}}) ==
           Point{{0.5}});
  }
  {
    const geom::BBox<_Float, N> Domain = {{{3}}, {{5}}};
    const DistancePeriodic f(Domain);
    assert(f.pointInDomain(Point{{3}}) ==
           Point{{3}});
    assert(f.pointInDomain(Point{{5}}) ==
           Point{{3}});
    assert(f.pointInDomain(Point{{1}}) ==
           Point{{3}});
    assert(f.pointInDomain(Point{{7}}) ==
           Point{{3}});
    assert(f.pointInDomain(Point{{4}}) ==
           Point{{4}});
    assert(f.pointInDomain(Point{{2}}) ==
           Point{{4}});
    assert(f.pointInDomain(Point{{6}}) ==
           Point{{4}});
  }

  // distance()
  {
    const geom::BBox<_Float, N> Domain = {{{0}}, {{1}}};
    const DistancePeriodic f(Domain);
    {
      _Float d = f.distance(Point{{0}},
                            Point{{0}});
      assert(std::abs(d - 0) < Eps);
    }
    {
      _Float d = f.distance(Point{{0}},
                            Point{{1}});
      assert(std::abs(d - 0) < Eps);
    }
    {
      _Float d = f.distance(Point{{1}},
                            Point{{1}});
      assert(std::abs(d - 0) < Eps);
    }
    {
      _Float d = f.distance(Point{{0}},
                            Point{{0.5}});
      assert(std::abs(d - 0.5) < Eps);
    }
    {
      _Float d = f.distance(Point{{0}},
                            Point{{0.25}});
      assert(std::abs(d - 0.25) < Eps);
    }
    {
      _Float d = f.distance(Point{{0}},
                            Point{{0.75}});
      assert(std::abs(d - 0.25) < Eps);
    }
  }
  {
    const geom::BBox<_Float, N> Domain = {{{3}}, {{5}}};
    const DistancePeriodic f(Domain);
    {
      _Float d = f.distance(Point{{3}},
                            Point{{3}});
      assert(std::abs(d - 0) < Eps);
    }
    {
      _Float d = f.distance(Point{{3}},
                            Point{{5}});
      assert(std::abs(d - 0) < Eps);
    }
    {
      _Float d = f.distance(Point{{3}},
                            Point{{4}});
      assert(std::abs(d - 1) < Eps);
    }
    {
      _Float d = f.distance(Point{{3}},
                            Point{{3.5}});
      assert(std::abs(d - 0.5) < Eps);
    }
    {
      _Float d = f.distance(Point{{3}},
                            Point{{4.5}});
      assert(std::abs(d - 0.5) < Eps);
    }
  }
}

template<typename _Float>
void
test2()
{
  const std::size_t N = 2;
  typedef geom::DistancePeriodic<_Float, N> DistancePeriodic;
  typedef typename DistancePeriodic::Point Point;
  const _Float Eps = 10 * std::numeric_limits<_Float>::epsilon();

  // pointInDomain()
  {
    const geom::BBox<_Float, N> Domain = {{{0, 0}}, {{1, 1}}};
    const DistancePeriodic f(Domain);
    assert(f.pointInDomain(Point{{0, 0}}) ==
           (Point{{0, 0}}));
    assert(f.pointInDomain(Point{{1, 1}}) ==
           (Point{{0, 0}}));
    assert(f.pointInDomain(Point{{-1, 2}}) ==
           (Point{{0, 0}}));
    assert(f.pointInDomain(Point{{1.5, -0.5}}) ==
           (Point{{0.5, 0.5}}));
  }
  {
    const geom::BBox<_Float, N> Domain = {{{2, 3}}, {{5, 7}}};
    const DistancePeriodic f(Domain);
    assert(f.pointInDomain(Point{{2, 3}}) ==
           (Point{{2, 3}}));
    assert(f.pointInDomain(Point{{5, 7}}) ==
           (Point{{2, 3}}));
    assert(f.pointInDomain(Point{{-1, 11}}) ==
           (Point{{2, 3}}));
    assert(f.pointInDomain(Point{{6.5, 9}}) ==
           (Point{{3.5, 5}}));
  }

  // distance()
  {
    const geom::BBox<_Float, N> Domain = {{{0, 0}}, {{1, 1}}};
    const DistancePeriodic f(Domain);
    {
      _Float d = f.distance(Point{{0, 0}},
                            Point{{0, 0}});
      assert(std::abs(d - 0) < Eps);
    }
    {
      _Float d = f.distance(Point{{0, 0}},
                            Point{{1, 1}});
      assert(std::abs(d - 0) < Eps);
    }
    {
      _Float d = f.distance(Point{{1, 1}},
                            Point{{1, 1}});
      assert(std::abs(d - 0) < Eps);
    }
    {
      _Float d = f.distance(Point{{0, 0}},
                            Point{{0.5, 0.5}});
      assert(std::abs(d - 0.5 * std::sqrt(2.)) < Eps);
    }
    {
      _Float d = f.distance(Point{{0, 0}},
                            Point{{0.25, 0.25}});
      assert(std::abs(d - 0.25 * std::sqrt(2.)) < Eps);
    }
    {
      _Float d = f.distance(Point{{0, 0}},
                            Point{{0.75, 0.75}});
      assert(std::abs(d - 0.25 * std::sqrt(2.)) < Eps);
    }
  }
  {
    const geom::BBox<_Float, N> Domain = {{{2, 3}}, {{5, 7}}};
    const DistancePeriodic f(Domain);
    {
      _Float d = f.distance(Point{{2, 3}},
                            Point{{2, 3}});
      assert(std::abs(d - 0) < Eps);
    }
    {
      _Float d = f.distance(Point{{2, 3}},
                            Point{{5, 7}});
      assert(std::abs(d - 0) < Eps);
    }
    {
      _Float d = f.distance(Point{{2, 3}},
                            Point{{4, 3}});
      assert(std::abs(d - 1) < Eps);
    }
  }
}

int
main()
{
  test1<float>();
  test1<double>();
  test2<float>();
  test2<double>();

  return 0;
}
