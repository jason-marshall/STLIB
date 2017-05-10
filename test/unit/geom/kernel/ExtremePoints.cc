// -*- C++ -*-

#include "stlib/geom/kernel/ExtremePoints.h"

#include <vector>

using namespace stlib;

using namespace geom;

int
main()
{
  {
    std::size_t constexpr D = 0;
    using Float = double;
    using ExtremePoints = geom::ExtremePoints<Float, D>;
    using BBox = geom::BBox<Float, D>;
    {
      ExtremePoints const x = extremePoints<ExtremePoints>();
      assert(isEmpty(x));
      assert(isEmpty(stlib::geom::specificBBox<BBox>(x)));
    }
  }

  {
    std::size_t constexpr D = 2;
    using Float = double;
    using ExtremePoints = geom::ExtremePoints<Float, D>;
    using Point = ExtremePoints::Point;
    using BBox = geom::BBox<Float, D>;
    {
      ExtremePoints const x = extremePoints<ExtremePoints>();
      assert(isEmpty(x));
    }
    {
      ExtremePoints const x = extremePoints<ExtremePoints>(Point{{0, 0}});
      assert(x == (ExtremePoints{{{{{{{0, 0}}, {{0, 0}}}},
                  {{{{0, 0}}, {{0, 0}}}}}}}));
    }
    {
      ExtremePoints const x = extremePoints<ExtremePoints>(Point{{1, 2}});
      assert(x == (ExtremePoints{{{{{{{1, 2}}, {{1, 2}}}},
                  {{{{1, 2}}, {{1, 2}}}}}}}));
    }
    {
      ExtremePoints const x =
        extremePoints<ExtremePoints>(std::pair<Point, int>{{{1, 2}}, 3});
      assert(x == (ExtremePoints{{{{{{{1, 2}}, {{1, 2}}}},
                  {{{{1, 2}}, {{1, 2}}}}}}}));
    }
    {
      ExtremePoints const x =
        extremePoints<ExtremePoints>(std::array<Point, 1>{{{{1, 2}}}});
      assert(x == (ExtremePoints{{{{{{{1, 2}}, {{1, 2}}}},
                  {{{{1, 2}}, {{1, 2}}}}}}}));
    }
    {
      ExtremePoints const x =
        extremePoints<ExtremePoints>(std::array<Point, 2>
          {{{{0, 0}}, {{1, 2}}}});
      assert(x == (ExtremePoints{{{{{{{0, 0}}, {{1, 2}}}},
                  {{{{0, 0}}, {{1, 2}}}}}}}));
    }
    
    {
      assert(isEmpty(specificBBox<BBox>(extremePoints<ExtremePoints>())));
    }
    {
      ExtremePoints const x = {{{{{{{0, 0}}, {{1, 0}}}},
                                 {{{{0, 0}}, {{0, 1}}}}}}};
      assert(specificBBox<BBox>(x) == (BBox{{{0, 0}}, {{1, 1}}}));
    }
    {
      std::array<Point, 1> points = {{{{2, 3}}}};
      ExtremePoints const x =
        extremePoints<ExtremePoints>(points.begin(), points.end());
      assert(x.points[0][0] == points[0]);
      assert(x.points[0][1] == points[0]);
      assert(x.points[1][0] == points[0]);
      assert(x.points[1][1] == points[0]);
    }
    {
      std::array<Point, 2> points = {{{{2, 3}}, {{5, 7}}}};
      ExtremePoints const x =
        extremePoints<ExtremePoints>(points.begin(), points.end());
      assert(x.points[0][0] == points[0]);
      assert(x.points[0][1] == points[1]);
      assert(x.points[1][0] == points[0]);
      assert(x.points[1][1] == points[1]);
    }
    {
      std::array<Point, 2> points = {{{{5, 7}}, {{2, 3}}}};
      ExtremePoints const x =
        extremePoints<ExtremePoints>(points.begin(), points.end());
      assert(x.points[0][0] == points[1]);
      assert(x.points[0][1] == points[0]);
      assert(x.points[1][0] == points[1]);
      assert(x.points[1][1] == points[0]);
    }
    {
      std::array<Point, 2> points = {{{{2, 7}}, {{5, 3}}}};
      ExtremePoints const x =
        extremePoints<ExtremePoints>(points.begin(), points.end());
      assert(x.points[0][0] == points[0]);
      assert(x.points[0][1] == points[1]);
      assert(x.points[1][0] == points[1]);
      assert(x.points[1][1] == points[0]);
    }

    {
      std::array<Point, 1> px = {{{{2, 3}}}};
      ExtremePoints x = extremePoints<ExtremePoints>(px.begin(), px.end());
      std::array<Point, 1> py = {{{{5, 7}}}};
      ExtremePoints y = extremePoints<ExtremePoints>(py.begin(), py.end());
      assert(x == x);
      assert(x != y);
      x += y;
      assert(x.points[0][0] == px[0]);
      assert(x.points[0][1] == py[0]);
      assert(x.points[1][0] == px[0]);
      assert(x.points[1][1] == py[0]);
    }
  }

  return 0;
}
