// -*- C++ -*-

#include "stlib/container/MultiIndexRange.h"

#include <sstream>

using namespace stlib;

int
main()
{
  // 1-D.
  {
    typedef container::MultiIndexRange<1> MultiIndexRange;
    typedef MultiIndexRange::Index Index;
    typedef MultiIndexRange::SizeList SizeList;
    typedef MultiIndexRange::IndexList IndexList;

    {
      const SizeList extents = {{0}};
      const IndexList bases = {{0}};
      const IndexList steps = {{1}};
      const MultiIndexRange x(extents);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);
    }
    {
      const SizeList extents = {{0}};
      const IndexList bases = {{0}};
      const IndexList steps = {{1}};
      MultiIndexRange x;
      x.initialize(extents);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);
    }

    {
      const SizeList extents = {{0}};
      const IndexList bases = {{ -2}};
      const IndexList steps = {{1}};
      const MultiIndexRange x(extents, bases);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);
    }
    {
      const SizeList extents = {{0}};
      const IndexList bases = {{ -2}};
      const IndexList steps = {{1}};
      MultiIndexRange x;
      x.initialize(extents, bases);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);
    }

    {
      const SizeList extents = {{0}};
      const IndexList bases = {{0}};
      const IndexList upper = {{0}};
      const IndexList steps = {{1}};
      const MultiIndexRange x(bases, upper);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);
      assert(x.upper() == upper);
    }
    {
      const SizeList extents = {{0}};
      const IndexList bases = {{0}};
      const IndexList upper = {{0}};
      const IndexList steps = {{1}};
      MultiIndexRange x;
      x.initialize(bases, upper);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);
      assert(x.upper() == upper);
    }

    {
      const SizeList extents = {{0}};
      const IndexList bases = {{ -2}};
      const IndexList upper = {{ -2}};
      const IndexList steps = {{1}};
      const MultiIndexRange x(bases, upper);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);
      assert(x.upper() == upper);
    }
    {
      const SizeList extents = {{0}};
      const IndexList bases = {{ -2}};
      const IndexList upper = {{ -2}};
      const IndexList steps = {{1}};
      MultiIndexRange x;
      x.initialize(bases, upper);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);
      assert(x.upper() == upper);
    }

    {
      const SizeList extents = {{2}};
      const IndexList bases = {{3}};
      const IndexList upper = {{5}};
      const IndexList steps = {{1}};
      const MultiIndexRange x(bases, upper);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);
      assert(x.upper() == upper);
    }
    {
      const SizeList extents = {{2}};
      const IndexList bases = {{3}};
      const IndexList upper = {{5}};
      const IndexList steps = {{1}};
      MultiIndexRange x;
      x.initialize(bases, upper);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);
      assert(x.upper() == upper);
    }

    {
      SizeList extents = {{0}};
      IndexList bases = {{ -2}};
      IndexList steps = {{3}};
      MultiIndexRange x(extents, bases, steps);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);

      {
        MultiIndexRange y = x;
        assert(x == y);
      }
      {
        MultiIndexRange y;
        y = x;
        assert(x == y);
      }
      {
        std::ostringstream out;
        out << x;
        MultiIndexRange y;
        std::istringstream in(out.str());
        in >> y;
        assert(x == y);
      }
    }

    // Overlap.
    {
      MultiIndexRange x(SizeList{{5}});
      MultiIndexRange y(SizeList{{3}});
      MultiIndexRange z(SizeList{{3}});
      assert(z == overlap(x, y));
    }
    {
      MultiIndexRange x(SizeList{{5}});
      MultiIndexRange y(SizeList{{3}}, IndexList{{-7}});
      MultiIndexRange z(SizeList{{0}}, IndexList{{0}});
      assert(z == overlap(x, y));
    }
    {
      MultiIndexRange x(SizeList{{5}});
      MultiIndexRange y(SizeList{{3}}, IndexList{{-1}});
      MultiIndexRange z(SizeList{{2}}, IndexList{{0}});
      assert(z == overlap(x, y));
    }
    {
      MultiIndexRange x(SizeList{{5}});
      MultiIndexRange y(SizeList{{3}}, IndexList{{1}});
      MultiIndexRange z(SizeList{{3}}, IndexList{{Index(1)}});
      assert(z == overlap(x, y));
    }
    {
      MultiIndexRange x(SizeList{{5}});
      MultiIndexRange y(SizeList{{3}}, IndexList{{3}});
      MultiIndexRange z(SizeList{{2}}, IndexList{{3}});
      assert(z == overlap(x, y));
    }
    {
      MultiIndexRange x(SizeList{{5}});
      MultiIndexRange y(SizeList{{3}}, IndexList{{5}});
      MultiIndexRange z(SizeList{{0}}, IndexList{{5}});
      assert(z == overlap(x, y));
    }

    // isIn.
    {
      MultiIndexRange range(SizeList{{5}});
      assert(isIn(range, IndexList{{0}}));
      assert(isIn(range, IndexList{{4}}));
      assert(! isIn(range, IndexList{{-1}}));
      assert(! isIn(range, IndexList{{5}}));
    }
    {
      MultiIndexRange range(SizeList{{5}}, IndexList{{-1}});
      assert(isIn(range, IndexList{{-1}}));
      assert(isIn(range, IndexList{{3}}));
      assert(! isIn(range, IndexList{{-2}}));
      assert(! isIn(range, IndexList{{4}}));
    }
    {
      MultiIndexRange range(SizeList{{5}}, IndexList{{0}}, IndexList{{2}});
      assert(isIn(range, IndexList{{0}}));
      assert(isIn(range, IndexList{{2}}));
      assert(isIn(range, IndexList{{8}}));
      assert(! isIn(range, IndexList{{-1}}));
      assert(! isIn(range, IndexList{{1}}));
      assert(! isIn(range, IndexList{{10}}));
    }
  }
  // 2-D.
  {
    typedef container::MultiIndexRange<2> MultiIndexRange;
    typedef MultiIndexRange::SizeList SizeList;
    typedef MultiIndexRange::IndexList IndexList;

    {
      SizeList extents = {{0, 0}};
      IndexList bases = {{0, 0}};
      IndexList steps = {{1, 1}};
      MultiIndexRange x(extents);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);
    }
    {
      SizeList extents = {{2, 3}};
      IndexList bases = {{ -3, -5}};
      IndexList steps = {{1, 1}};
      MultiIndexRange x(extents, bases);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);
    }
    {
      SizeList extents = {{2, 3}};
      IndexList bases = {{ -3, -5}};
      IndexList steps = {{11, 13}};
      MultiIndexRange x(extents, bases, steps);
      assert(x.extents() == extents);
      assert(x.bases() == bases);
      assert(x.steps() == steps);

      {
        MultiIndexRange y = x;
        assert(x == y);
      }
      {
        MultiIndexRange y;
        y = x;
        assert(x == y);
      }
      {
        std::ostringstream out;
        out << x;
        MultiIndexRange y;
        std::istringstream in(out.str());
        in >> y;
        assert(x == y);
      }
    }
    // Overlap.
    {
      MultiIndexRange x(SizeList{{5, 7}});
      MultiIndexRange y(SizeList{{2, 3}}, IndexList{{-8, -9}});
      MultiIndexRange z(SizeList{{0, 0}}, IndexList{{0, 0}});
      assert(z == overlap(x, y));
    }
    {
      MultiIndexRange x(SizeList{{5, 7}});
      MultiIndexRange y(SizeList{{2, 3}}, IndexList{{-1, -9}});
      MultiIndexRange z(SizeList{{1, 0}}, IndexList{{0, 0}});
      assert(z == overlap(x, y));
    }
    {
      MultiIndexRange x(SizeList{{5, 7}});
      MultiIndexRange y(SizeList{{2, 3}}, IndexList{{-1, -1}});
      MultiIndexRange z(SizeList{{1, 2}}, IndexList{{0, 0}});
      assert(z == overlap(x, y));
    }
    {
      MultiIndexRange x(SizeList{{5, 7}});
      MultiIndexRange y(SizeList{{2, 3}}, IndexList{{1, -1}});
      MultiIndexRange z(SizeList{{2, 2}}, IndexList{{1, 0}});
      assert(z == overlap(x, y));
    }
    {
      MultiIndexRange x(SizeList{{5, 7}});
      MultiIndexRange y(SizeList{{2, 3}}, IndexList{{1, 2}});
      MultiIndexRange z(SizeList{{2, 3}}, IndexList{{1, 2}});
      assert(z == overlap(x, y));
    }
    {
      MultiIndexRange x(SizeList{{5, 7}});
      MultiIndexRange y(SizeList{{2, 3}}, IndexList{{4, 2}});
      MultiIndexRange z(SizeList{{1, 3}}, IndexList{{4, 2}});
      assert(z == overlap(x, y));
    }
    {
      MultiIndexRange x(SizeList{{5, 7}});
      MultiIndexRange y(SizeList{{2, 3}}, IndexList{{4, 5}});
      MultiIndexRange z(SizeList{{1, 2}}, IndexList{{4, 5}});
      assert(z == overlap(x, y));
    }
    {
      MultiIndexRange x(SizeList{{5, 7}});
      MultiIndexRange y(SizeList{{2, 3}}, IndexList{{6, 5}});
      MultiIndexRange z(SizeList{{0, 2}}, IndexList{{6, 5}});
      assert(z == overlap(x, y));
    }
    {
      MultiIndexRange x(SizeList{{5, 7}});
      MultiIndexRange y(SizeList{{2, 3}}, IndexList{{6, 8}});
      MultiIndexRange z(SizeList{{0, 0}}, IndexList{{6, 8}});
      assert(z == overlap(x, y));
    }
    // isIn.
    {
      MultiIndexRange range(SizeList{{5, 7}});
      assert(isIn(range, IndexList{{0, 0}}));
      assert(isIn(range, IndexList{{4, 6}}));
      assert(! isIn(range, IndexList{{-1, 0}}));
      assert(! isIn(range, IndexList{{0, -1}}));
      assert(! isIn(range, IndexList{{5, 0}}));
      assert(! isIn(range, IndexList{{0, 7}}));
    }
    {
      MultiIndexRange range(SizeList{{5, 7}}, IndexList{{-1, -2}});
      assert(isIn(range, IndexList{{-1, -2}}));
      assert(isIn(range, IndexList{{3, 4}}));
      assert(! isIn(range, IndexList{{-2, 0}}));
      assert(! isIn(range, IndexList{{0, -3}}));
      assert(! isIn(range, IndexList{{4, 0}}));
      assert(! isIn(range, IndexList{{0, 5}}));
    }
    {
      MultiIndexRange range(SizeList{{5, 7}}, IndexList{{0, 0}},
                            IndexList{{2, 3}});
      assert(isIn(range, IndexList{{0, 0}}));
      assert(isIn(range, IndexList{{8, 18}}));
      assert(! isIn(range, IndexList{{-1, 0}}));
      assert(! isIn(range, IndexList{{0, -1}}));
      assert(! isIn(range, IndexList{{10, 0}}));
      assert(! isIn(range, IndexList{{0, 21}}));
      assert(! isIn(range, IndexList{{1, 0}}));
      assert(! isIn(range, IndexList{{0, 1}}));
      assert(! isIn(range, IndexList{{0, 2}}));
    }
  }

  return 0;
}
