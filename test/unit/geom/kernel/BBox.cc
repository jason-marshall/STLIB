// -*- C++ -*-

#include "stlib/geom/kernel/BBox.h"

#include <iostream>
#include <vector>


template<typename _T>
void
testDefaultBBox()
{
  using stlib::geom::bbox;
  using stlib::geom::defaultBBoxForEach;
  typedef stlib::geom::BBox<_T, 1> BBox;

  // BBox
  {
    typedef stlib::geom::BBox<_T, 1> Object;
    assert(isEmpty(bbox<Object>()));
    assert((bbox(Object{{{0}},{{1}}}) == BBox{{{0}},{{1}}}));
    std::vector<Object> objects = {Object{{{0}},{{1}}}, Object{{{1}},{{2}}}};
    assert((bbox(objects.begin(), objects.end()) == BBox{{{0}},{{2}}}));
    assert((defaultBBoxForEach(objects.begin(), objects.begin()) ==
            std::vector<BBox>{}));
    assert((defaultBBoxForEach(objects.begin(), objects.end()) ==
            std::vector<BBox>{BBox{{{0}},{{1}}}, BBox{{{1}},{{2}}}}));
  }
         
  // std::array<_T, _D>
  {
    typedef std::array<_T, 1> Object;
    assert(isEmpty(bbox<Object>()));
    assert((bbox(Object{{23}}) == BBox{{{23}},{{23}}}));
    std::vector<Object> objects = {Object{{0}}, Object{{2}}};
    assert((bbox(objects.begin(), objects.end()) == BBox{{{0}},{{2}}}));
    assert((defaultBBoxForEach(objects.begin(), objects.begin()) ==
            std::vector<BBox>{}));
    assert((defaultBBoxForEach(objects.begin(), objects.end()) ==
            std::vector<BBox>{BBox{{{0}},{{0}}}, BBox{{{2}},{{2}}}}));
  }

  // std::array<std::array<_T, _D>, _N>
  {
    typedef std::array<std::array<_T, 1>, 1> Object;
    assert(isEmpty(bbox<Object>()));
    assert((bbox(Object{{{{23}}}}) == BBox{{{23}},{{23}}}));
    std::vector<Object> objects = {Object{{{{0}}}}, Object{{{{2}}}}};
    assert((bbox(objects.begin(), objects.end()) == BBox{{{0}},{{2}}}));
    assert((defaultBBoxForEach(objects.begin(), objects.begin()) ==
            std::vector<BBox>{}));
    assert((defaultBBoxForEach(objects.begin(), objects.end()) ==
            std::vector<BBox>{BBox{{{0}},{{0}}}, BBox{{{2}},{{2}}}}));
  }

  // std::pair<std::array<_T, 1>, int>
  {
    using Object = std::pair<std::array<_T, 1>, int>;
    assert(isEmpty(bbox<Object>()));
    assert((bbox(Object{{{23}}, 29}) == BBox{{{23}},{{23}}}));
    std::vector<Object> objects = {Object{{{0}}, 2}, Object{{{2}}, 3}};
    assert((bbox(objects.begin(), objects.end()) == BBox{{{0}},{{2}}}));
    assert((defaultBBoxForEach(objects.begin(), objects.begin()) ==
            std::vector<BBox>{}));
    assert((defaultBBoxForEach(objects.begin(), objects.end()) ==
            std::vector<BBox>{BBox{{{0}},{{0}}}, BBox{{{2}},{{2}}}}));
  }
}


int
main()
{
  using stlib::geom::BBox;
  using stlib::geom::specificBBox;
  using Index = std::array<int, 3>;

  {
    // sizeof
    std::cout << "sizeof(int) = " << sizeof(int) << '\n';
    std::cout << "sizeof(BBox<int, 3>) = "
              << sizeof(BBox<int, 3>) << '\n';
    std::cout << "sizeof(double) = " << sizeof(double) << '\n';
    std::cout << "sizeof(BBox<double, 2>) = "
              << sizeof(BBox<double, 2>) << '\n';
  }
  {
    // Default constructor
    std::cout << "BBox<int, 1>() = "
              << BBox<int, 1>() << '\n';
    std::cout << "BBox<double, 2>() = "
              << BBox<double, 2>() << '\n';
  }
  // Equality.
  {
    assert((BBox<double, 1>{{{1}}, {{0}}}) == (BBox<double, 1>{{{2}}, {{0}}}));
  }
  // Build with specificBBox().
  {
    // Build from a sequence of points.
    {
      std::array<std::array<float, 1>, 2> points = {{{{0}}, {{1}}}};
      // Use a range of objects.
      {
        BBox<float, 1> const x =
          stlib::geom::specificBBox<BBox<float, 1> >
          (points.begin(), points.end());
        assert(x.lower == (std::array<float, 1>{{0}}));
        assert(x.upper == (std::array<float, 1>{{1}}));
      }
      // Use a range of objects and specify the function that converts the 
      // input object to a boundable object.
      {
        BBox<float, 1> const x =
          stlib::geom::specificBBox<BBox<float, 1> >
          (points.begin(), points.end(),
           [](std::array<float, 1> const& a){return a;});
        assert(x.lower == (std::array<float, 1>{{0}}));
        assert(x.upper == (std::array<float, 1>{{1}}));
      }
    }
    // Convert from double-precision to single-precision.
    {
      std::array<std::array<double, 1>, 2> points = {{{{0}}, {{1}}}};
      // Use a range of objects.
      {
        BBox<float, 1> x =
          stlib::geom::specificBBox<BBox<float, 1> >
          (points.begin(), points.end());
        assert(x.lower == (std::array<float, 1>{{0}}));
        assert(x.upper == (std::array<float, 1>{{1}}));
      }
      // Use a range of objects and specify the function that converts the 
      // input object to a boundable object.
      {
        BBox<float, 1> const x =
          stlib::geom::specificBBox<BBox<float, 1> >
          (points.begin(), points.end(),
           [](std::array<double, 1> const& a){return a;});
        assert(x.lower == (std::array<float, 1>{{0}}));
        assert(x.upper == (std::array<float, 1>{{1}}));
      }
    }
    // Build from a sequence of bounding boxes.
    {
      std::array<BBox<float, 1>, 2> boxes = {{
          {{{0}}, {{1}}},
          {{{1}}, {{2}}}
        }};
      // Use a range of objects.
      {
        BBox<float, 1> const x =
          stlib::geom::specificBBox<BBox<float, 1> >
          (boxes.begin(), boxes.end());
        assert(x.lower == (std::array<float, 1>{{0}}));
        assert(x.upper == (std::array<float, 1>{{2}}));
      }
      // Use a range of objects and specify the function that converts the 
      // input object to a boundable object.
      {
        BBox<float, 1> const x =
          stlib::geom::specificBBox<BBox<float, 1> >
          (boxes.begin(), boxes.end(),
           [](BBox<float, 1> const& a){return a;});
        assert(x.lower == (std::array<float, 1>{{0}}));
        assert(x.upper == (std::array<float, 1>{{2}}));
      }
    }
    // The object type is an index into an auxiliary array.
    {
      using Point = std::array<float, 1>;
      std::vector<Point> const points = {{{2}}, {{3}}, {{5}}};
      std::vector<std::size_t> const indices = {2, 1, 0};
      auto const x =
        stlib::geom::specificBBox<BBox<float, 1> >
        (indices.begin(), indices.end(),
         [&points](std::size_t i){return points[i];});
      assert(x.lower == (std::array<float, 1>{{2}}));
      assert(x.upper == (std::array<float, 1>{{5}}));
    }
  }
  {
    typedef BBox<int, 3> BBox;
    typedef BBox::Point Point;

    // iterator constructor
    std::vector<Point> v;
    v.push_back(Point{{1, 2, 3}});
    v.push_back(Point{{-2, 2, 4}});
    v.push_back(Point{{2, 5, -3}});
    {
      BBox bv = stlib::geom::specificBBox<BBox>(v.begin(), v.end());
      assert((BBox{{{-2, 2, -3}}, {{2, 5, 4}}}) == bv);
    }

    // copy constructor
    BBox b = {{{1, 2, 3}}, {{2, 3, 5}}};
    {
      BBox b2 = b;
      assert(b2 == b);
    }

    // assignment operator
    {
      BBox b2;
      b2 = b;
      assert(b2 == b);
    }

    // accessors
    assert(1 == b.lower[0] &&
           2 == b.lower[1] &&
           3 == b.lower[2] &&
           2 == b.upper[0] &&
           3 == b.upper[1] &&
           5 == b.upper[2]);

    assert(b.lower == (Point{{1, 2, 3}}));
    assert(b.upper == (Point{{2, 3, 5}}));
    assert(content(b) == 2);

    // offset
    {
      BBox x = {{{1, 2, 3}}, {{2, 3, 5}}};
      offset(&x, 0);
      assert(x.lower == (Point{{1, 2, 3}}));
      assert(x.upper == (Point{{2, 3, 5}}));
      offset(&x, 1);
      assert(x.lower == (Point{{0, 1, 2}}));
      assert(x.upper == (Point{{3, 4, 6}}));
      offset(&x, -1);
      assert(x.lower == (Point{{1, 2, 3}}));
      assert(x.upper == (Point{{2, 3, 5}}));
    }
  }
  {
    // Dimension
    assert((BBox<int, 1>::Dimension) == 1);
    assert((BBox<int, 2>::Dimension) == 2);
  }
  {
    // Empty bounding box.
    assert(isEmpty(specificBBox<BBox<float, 0> >()));
    assert(isEmpty(specificBBox<BBox<int, 1> >()));
    assert(isEmpty(specificBBox<BBox<float, 1> >()));
    assert((specificBBox<BBox<float, 1> >()) == (specificBBox<BBox<float, 1> >()));
    // Any two empty bounding boxes are equal.
    assert((specificBBox<BBox<float, 1> >()) == (BBox<float, 1>{{{1}}, {{0}}}));
  }
  {
    // isEmpty()

    typedef BBox<int, 1> BBox;
    assert(! isEmpty(BBox{{{0}}, {{0}}}));
    assert(isEmpty(BBox{{{1}}, {{0}}}));
    assert(! isEmpty(BBox{{{0}}, {{1}}}));
  }
  {
    // isEmpty()

    typedef BBox<int, 3> BBox;
    typedef BBox::Point Point;
    BBox b;

    b.lower = Point{{0, 0, 0}};
    b.upper = Point{{0, 0, 0}};
    assert(! isEmpty(b));

    b.lower = Point{{1, 2, 3}};
    b.upper = Point{{2, 3, 5}};
    assert(! isEmpty(b));

    b.lower = Point{{1, 0, 0}};
    b.upper = Point{{0, 0, 0}};
    assert(isEmpty(b));

    b.lower = Point{{0, 1, 0}};
    b.upper = Point{{0, 0, 0}};
    assert(isEmpty(b));

    b.lower = Point{{0, 0, 1}};
    b.upper = Point{{0, 0, 0}};
    assert(isEmpty(b));
  }
  {
    typedef BBox<float, 2> BBox;
    assert(centroid(BBox{{{1, 2}}, {{5, 7}}}) ==
           (std::array<float, 2>{{3, 4.5}}));
  }
  {
    // +=
    typedef BBox<float, 1> BBox;
    // Add an empty bounding box.
    {
      BBox b = specificBBox<BBox>();
      b += b;
      assert(isEmpty(b));
    }
    // Add an empty bounding box of a different type.
    {
      BBox b = specificBBox<BBox>();
      b += specificBBox<stlib::geom::BBox<double, 1> >();
      assert(isEmpty(b));
    }
    // Add points.
    {
      BBox b = specificBBox<BBox>();
      b += std::array<float, 1>{{0}};
      assert(b == (BBox{{{0}}, {{0}}}));
      b += std::array<float, 1>{{1}};
      assert(b == (BBox{{{0}}, {{1}}}));
      b += specificBBox<BBox>();
      assert(b == (BBox{{{0}}, {{1}}}));
    }
    // Add a line segment.
    {
      BBox b = specificBBox<BBox>();
      b += std::array<std::array<float, 1>, 2>{{{{1}}, {{0}}}};
      assert(b == (BBox{{{0}}, {{1}}}));
    }
  }
  {
    // isInside() for points
    typedef BBox<double, 3> BBox;
    typedef BBox::Point Point;
    assert(! isInside(specificBBox<BBox>(), Point{{0, 0, 0}}));

    BBox const b = {{{1., 2., 3.}}, {{2., 3., 5.}}};
    assert(isInside(b, Point{{1., 2., 3.}}));
    assert(isInside(b, Point{{2., 2., 3.}}));
    assert(isInside(b, Point{{1., 3., 3.}}));
    assert(isInside(b, Point{{1., 2., 5.}}));
    assert(isInside(b, Point{{2., 3., 5.}}));
    assert(isInside(b, Point{{1.5, 2., 3.}}));
    assert(isInside(b, Point{{1.5, 2.5, 3.5}}));
    assert(!isInside(b, Point{{-1., 2.5, 3.5}}));
    assert(!isInside(b, Point{{1.5, 1., 3.5}}));
    assert(!isInside(b, Point{{1.5, 4., 3.5}}));
    assert(!isInside(b, Point{{1.5, 2.5, 5.5}}));
    assert(!isInside(b, Point{{1.5, 2.5, 0.}}));
    assert(!isInside(b, Point{{10., 20., 30.}}));
    assert(!isInside(b, Point{{0., 0., 0.}}));
  }
  {
    // isInside() for BBox's
    typedef BBox<double, 3> BBox;

    // Nothing is inside an empty bounding box, not even another empty 
    // bounding box.
    assert(! isInside(specificBBox<BBox>(), specificBBox<BBox>()));
    // An empty bounding box is inside any non-empty bounding box.
    assert(isInside(BBox{{{0, 0, 0}}, {{0, 0, 0}}}, specificBBox<BBox>()));

    BBox const b = {{{1., 2., 3.}}, {{2., 3., 5.}}};

    assert(isInside(b, BBox{{{1., 2., 3.}}, {{2., 3., 5.}}}));
    assert(isInside(b, BBox{{{1., 2., 3.}}, {{1., 2., 3.}}}));
    assert(isInside(b, BBox{{{1.5, 2., 3.}}, {{2., 3., 5.}}}));
    assert(isInside(b, BBox{{{1., 2.5, 3.}}, {{2., 3., 5.}}}));
    assert(isInside(b, BBox{{{1., 2., 3.5}}, {{2., 3., 5.}}}));

    assert(! isInside(b, BBox{{{0., 2., 3.}}, {{2., 3., 5.}}}));
    assert(! isInside(b, BBox{{{1., 1., 3.}}, {{2., 3., 5.}}}));
    assert(! isInside(b, BBox{{{1., 2., 2.}}, {{2., 3., 5.}}}));
    assert(! isInside(b, BBox{{{1., 2., 3.}}, {{3., 3., 5.}}}));
    assert(! isInside(b, BBox{{{1., 2., 3.}}, {{2., 4., 5.}}}));
    assert(! isInside(b, BBox{{{1., 2., 3.}}, {{2., 3., 6.}}}));
  }
  {
    // isInside() for objects.
    typedef BBox<double, 2> BBox;
    typedef std::array<std::array<double, 2>, 2> Segment;
    Segment const segment = {{{{0, 0}}, {{1, 1}}}};
    assert(! isInside(specificBBox<BBox>(), segment));
    assert(! isInside(BBox{{{0, 0}}, {{0, 0}}}, segment));
    assert(isInside(BBox{{{0, 0}}, {{1, 1}}}, segment));
  }
  {
    // doOverlap
    typedef BBox<float, 1> BBox;
    assert(! doOverlap(specificBBox<BBox>(), specificBBox<BBox>()));
    assert(! doOverlap(BBox{{{1}}, {{0}}}, BBox{{{1}}, {{0}}}));
    assert(! doOverlap(BBox{{{1}}, {{0}}}, BBox{{{0}}, {{1}}}));
    assert(! doOverlap(BBox{{{0}}, {{1}}}, specificBBox<BBox>()));
    assert(doOverlap(BBox{{{0}}, {{1}}}, BBox{{{1}}, {{2}}}));
  }
  {
    // do_overlap

    typedef BBox<int, 1> BBox;
    typedef BBox::Point Point;
    Point a0 = {{0}}, a1 = {{1}}, b0 = {{2}}, b1 = {{3}};
    assert(! doOverlap(BBox{a0, a1}, BBox{b0, b1}));

    a0[0] = 0;
    a1[0] = 5;
    b0[0] = 2;
    b1[0] = 3;
    assert(doOverlap(BBox{a0, a1}, BBox{b0, b1}));

    a0[0] = 0;
    a1[0] = 5;
    b0[0] = 2;
    b1[0] = 8;
    assert(doOverlap(BBox{a0, a1}, BBox{b0, b1}));
  }
  {
    typedef BBox<int, 3> BBox;
    // do_overlap
    assert(! doOverlap(BBox{{{0, 0, 0}}, {{1, 1, 1}}},
                       BBox{{{2, 2, 2}}, {{3, 3, 3}}}));

    assert(doOverlap(BBox{{{1, 2, 3}}, {{7, 8, 9}}},
                     BBox{{{3, 4, 5}}, {{4, 5, 6}}}));

    assert(doOverlap(BBox{{{1, 2, 3}}, {{7, 8, 9}}},
                     BBox{{{3, 4, 5}}, {{8, 9, 10}}}));

  }
  {
    // intersection()
    typedef BBox<float, 1> BBox;
    assert(intersection(specificBBox<BBox>(), specificBBox<BBox>()) ==
           specificBBox<BBox>());
    assert(intersection(BBox{{{1}}, {{0}}}, BBox{{{1}}, {{0}}}) ==
           specificBBox<BBox>());
    assert(intersection(BBox{{{1}}, {{0}}}, BBox{{{0}}, {{1}}}) ==
           specificBBox<BBox>());
    assert(intersection(BBox{{{0}}, {{1}}}, specificBBox<BBox>()) ==
           specificBBox<BBox>());
    assert(intersection(BBox{{{0}}, {{1}}}, BBox{{{1}}, {{2}}}) ==
           (BBox{{{1}}, {{1}}}));
  }
  {
    using stlib::geom::squaredDistanceBetweenIntervals;
    // squaredDistanceBetweenIntervals
    assert(squaredDistanceBetweenIntervals(0., 1., 3., 4.) == 4.);
    assert(squaredDistanceBetweenIntervals(2., 3., 3., 4.) == 0.);
    assert(squaredDistanceBetweenIntervals(2., 5., 3., 4.) == 0.);
    assert(squaredDistanceBetweenIntervals(4., 5., 3., 4.) == 0.);
    assert(squaredDistanceBetweenIntervals(6., 7., 3., 4.) == 4.);
  }
  {
    typedef BBox<double, 3> BBox;
    // squaredDistance
    assert(squaredDistance(BBox{{{0., 0., 0.}},
          {{1., 1., 1.}}},
        BBox{{{2., 2., 2.}},
          {{3., 3., 3.}}}) == 3.);
    assert(squaredDistance(BBox{{{0., 0., 0.}},
          {{2., 1., 1.}}},
        BBox{{{2., 2., 2.}},
          {{3., 3., 3.}}}) == 2.);
    assert(squaredDistance(BBox{{{0., 0., 0.}},
          {{2., 2., 1.}}},
        BBox{{{2., 2., 2.}},
          {{3., 3., 3.}}}) == 1.);
    assert(squaredDistance(BBox{{{0., 0., 0.}},
          {{2., 2., 2.}}},
        BBox{{{2., 2., 2.}},
          {{3., 3., 3.}}}) == 0.);
    assert(squaredDistance(BBox{{{0., 0., 0.}},
          {{4., 4., 4.}}},
        BBox{{{2., 2., 2.}},
          {{3., 3., 3.}}}) == 0.);
  }
  {
    using stlib::geom::scanConvert;
    typedef BBox<double, 3> BB;
    typedef BB::Point Point;
    // scan_convert
    std::vector<Index> indexSet1, indexSet2;

    BB b;
    b.lower = Point{{1., 2., 3.}};
    b.upper = Point{{2., 3., 1.}};
    scanConvert<int>(std::back_inserter(indexSet1), b);
    assert(indexSet1.size() == 0);
    assert(indexSet1.size() == indexSet2.size());

    b.lower = Point{{0., 0., 0.}};
    b.upper = Point{{1., 1., 1.}};
    scanConvert<int>(std::back_inserter(indexSet1), b);
    assert(indexSet1.size() == 8);
    int i, j, k;
    for (k = 0; k <= 1; ++k) {
      for (j = 0; j <= 1; ++j) {
        for (i = 0; i <= 1; ++i) {
          indexSet2.push_back(Index{{i, j, k}});
        }
      }
    }
    assert(indexSet1 == indexSet2);
    indexSet1.clear();
    indexSet2.clear();

    b.lower = Point{{1.1, 2.2, 3.3}};
    b.upper = Point{{4.4, 5.5, 6.6}};
    scanConvert<int>(std::back_inserter(indexSet1), b);
    assert(indexSet1.size() == 27);
    for (k = 4; k <= 6; ++k) {
      for (j = 3; j <= 5; ++j) {
        for (i = 2; i <= 4; ++i) {
          indexSet2.push_back(Index{{i, j, k}});
        }
      }
    }
    assert(indexSet1 == indexSet2);
    indexSet1.clear();

    typedef BBox<int, 3> BBox;
    // Scan convert on a domain.
    scanConvert(std::back_inserter(indexSet1), b,
                BBox{{{0, 0, 0}}, {{10, 10, 10}}});
    assert(indexSet1.size() == 27);
    assert(indexSet1 == indexSet2);
    indexSet1.clear();
    indexSet2.clear();

    // Scan convert on a domain.
    scanConvert(std::back_inserter(indexSet1), b,
                BBox{{{0, 0, 0}}, {{0, 0, 0}}});
    assert(indexSet1.size() == 0);
    assert(indexSet1 == indexSet2);
    indexSet1.clear();
    indexSet2.clear();
  }

  // maxAbsCoord()
  {
    typedef BBox<float, 3> BBox;

    assert(maxAbsCoord(stlib::geom::specificBBox<BBox>()) == 0);
    assert(maxAbsCoord(BBox{{{0, 0, 0}}, {{0, 0, 0}}}) == 0);
    assert(maxAbsCoord(BBox{{{0, 0, 0}}, {{1, 2, 3}}}) == 3);
    assert(maxAbsCoord(BBox{{{-1, -2, -3}}, {{0, 0, 0}}}) == 3);
  }

  // content()
  {
    using stlib::geom::content;
    typedef BBox<float, 3> BBox;

    assert(content(stlib::geom::specificBBox<BBox>()) == 0);
    assert(content(BBox{{{0, 0, 0}}, {{1, 2, 3}}}) == 6);
  }

  // specificBBox()
  {
    using stlib::geom::specificBBox;

    // BBox
    assert((specificBBox<BBox<float, 1> >(BBox<float, 1>{{{0}},{{1}}}) ==
            BBox<float, 1>{{{0}},{{1}}}));
    assert((specificBBox<BBox<float, 1> >(BBox<double, 1>{{{0}},{{1}}}) ==
            BBox<float, 1>{{{0}},{{1}}}));
    assert((specificBBox<BBox<double, 1> >(BBox<float, 1>{{{0}},{{1}}}) ==
            BBox<double, 1>{{{0}},{{1}}}));
    assert((specificBBox<BBox<double, 1> >(BBox<double, 1>{{{0}},{{1}}}) ==
            BBox<double, 1>{{{0}},{{1}}}));
            
    // std::array<_Float2, _D>
    assert((specificBBox<BBox<float, 1> >(std::array<float, 1>{{23}}) ==
            BBox<float, 1>{{{23}},{{23}}}));
    assert((specificBBox<BBox<float, 1> >(std::array<double, 1>{{23}}) ==
            BBox<float, 1>{{{23}},{{23}}}));
    assert((specificBBox<BBox<double, 1> >(std::array<float, 1>{{23}}) ==
            BBox<double, 1>{{{23}},{{23}}}));
    assert((specificBBox<BBox<double, 1> >(std::array<double, 1>{{23}}) ==
            BBox<double, 1>{{{23}},{{23}}}));

    // std::array<std::array<_Float2, _D>, _N>
    assert((specificBBox<BBox<float, 1> >
            (std::array<std::array<float, 1>, 1>{{{{23}}}}) ==
            BBox<float, 1>{{{23}},{{23}}}));
    assert((specificBBox<BBox<float, 1> >
            (std::array<std::array<double, 1>, 1>{{{{23}}}}) ==
            BBox<float, 1>{{{23}},{{23}}}));
    assert((specificBBox<BBox<double, 1> >
            (std::array<std::array<float, 1>, 1>{{{{23}}}}) ==
            BBox<double, 1>{{{23}},{{23}}}));
    assert((specificBBox<BBox<double, 1> >
            (std::array<std::array<double, 1>, 1>{{{{23}}}}) ==
            BBox<double, 1>{{{23}},{{23}}}));
  }

  testDefaultBBox<int>();
  testDefaultBBox<float>();
  testDefaultBBox<double>();

  return 0;
}
