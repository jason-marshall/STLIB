// -*- C++ -*-

#include "stlib/container/SimpleMultiIndexRange.h"

#include <sstream>

using namespace stlib;

int
main()
{
  // 1-D.
  {
    typedef container::SimpleMultiIndexRange<1> SimpleMultiIndexRange;
    typedef SimpleMultiIndexRange::IndexList IndexList;

    {
      const IndexList extents = {{0}};
      const IndexList bases = {{0}};
      const SimpleMultiIndexRange x = {extents, bases};
      assert(x.extents == extents);
      assert(x.bases == bases);
    }

    // Overlap.
    {
      SimpleMultiIndexRange x = {{{5}}, {{0}}};
      SimpleMultiIndexRange y = {{{3}}, {{0}}};
      SimpleMultiIndexRange z = {{{3}}, {{0}}};
      assert(z == overlap(x, y));
    }
    {
      SimpleMultiIndexRange x = {{{5}}, {{0}}};
      SimpleMultiIndexRange y = {{{3}}, {{1}}};
      SimpleMultiIndexRange z = {{{3}}, {{1}}};
      assert(z == overlap(x, y));
    }
    {
      SimpleMultiIndexRange x = {{{5}}, {{0}}};
      SimpleMultiIndexRange y = {{{3}}, {{3}}};
      SimpleMultiIndexRange z = {{{2}}, {{3}}};
      assert(z == overlap(x, y));
    }
    {
      SimpleMultiIndexRange x = {{{5}}, {{0}}};
      SimpleMultiIndexRange y = {{{3}}, {{5}}};
      SimpleMultiIndexRange z = {{{0}}, {{5}}};
      assert(z == overlap(x, y));
    }

    // isIn.
    {
      SimpleMultiIndexRange range = {{{5}}, {{0}}};
      assert(isIn(range, IndexList{{0}}));
      assert(isIn(range, IndexList{{4}}));
      assert(! isIn(range, IndexList{{5}}));
    }
  }
  // 2-D.
  {
    typedef container::SimpleMultiIndexRange<2> SimpleMultiIndexRange;
    typedef SimpleMultiIndexRange::IndexList IndexList;

    {
      IndexList extents = {{0, 0}};
      IndexList bases = {{0, 0}};
      SimpleMultiIndexRange x = {extents, bases};
      assert(x.extents == extents);
      assert(x.bases == bases);
    }

    // Overlap.
    {
      SimpleMultiIndexRange x = {{{5, 7}}, {{0, 0}}};
      SimpleMultiIndexRange y = {{{2, 3}}, {{1, 2}}};
      SimpleMultiIndexRange z = {{{2, 3}}, {{1, 2}}};
      assert(z == overlap(x, y));
    }
    {
      SimpleMultiIndexRange x = {{{5, 7}}, {{0, 0}}};
      SimpleMultiIndexRange y = {{{2, 3}}, {{4, 2}}};
      SimpleMultiIndexRange z = {{{1, 3}}, {{4, 2}}};
      assert(z == overlap(x, y));
    }
    {
      SimpleMultiIndexRange x = {{{5, 7}}, {{0, 0}}};
      SimpleMultiIndexRange y = {{{2, 3}}, {{4, 5}}};
      SimpleMultiIndexRange z = {{{1, 2}}, {{4, 5}}};
      assert(z == overlap(x, y));
    }
    {
      SimpleMultiIndexRange x = {{{5, 7}}, {{0, 0}}};
      SimpleMultiIndexRange y = {{{2, 3}}, {{6, 5}}};
      SimpleMultiIndexRange z = {{{0, 2}}, {{6, 5}}};
      assert(z == overlap(x, y));
    }
    {
      SimpleMultiIndexRange x = {{{5, 7}}, {{0, 0}}};
      SimpleMultiIndexRange y = {{{2, 3}}, {{6, 8}}};
      SimpleMultiIndexRange z = {{{0, 0}}, {{6, 8}}};
      assert(z == overlap(x, y));
    }
    // isIn.
    {
      SimpleMultiIndexRange range = {{{5, 7}}, {{0, 0}}};
      assert(isIn(range, IndexList{{0, 0}}));
      assert(isIn(range, IndexList{{4, 6}}));
      assert(! isIn(range, IndexList{{5, 0}}));
      assert(! isIn(range, IndexList{{0, 7}}));
    }
  }

  return 0;
}
