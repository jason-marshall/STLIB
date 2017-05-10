// -*- C++ -*-


#include "stlib/levelSet/count.h"


using namespace stlib;

template<typename _T>
bool
isNaN(const _T x)
{
  return x != x;
}


int
main()
{
  typedef double Number;

  const Number NaN = std::numeric_limits<Number>::quiet_NaN();
  const Number Inf = std::numeric_limits<Number>::infinity();

  // hasKnown()
  {
    using levelSet::hasKnown;
    {
      std::array<double, 2> x = {{NaN, NaN}};
      assert(! hasKnown(x.begin(), x.end()));
    }
    {
      std::array<double, 2> x = {{NaN, Inf}};
      assert(hasKnown(x.begin(), x.end()));
    }
  }

  // hasUnknown()
  {
    using levelSet::hasUnknown;
    {
      std::array<double, 2> x = {{NaN, NaN}};
      assert(hasUnknown(x.begin(), x.end()));
    }
    {
      std::array<double, 2> x = {{NaN, Inf}};
      assert(hasUnknown(x.begin(), x.end()));
    }
    {
      std::array<double, 2> x = {{Inf, Inf}};
      assert(! hasUnknown(x.begin(), x.end()));
    }
  }

  // allSame()
  {
    using levelSet::allSame;
    {
      std::array<double, 2> x = {{NaN, NaN}};
      assert(allSame(x.begin(), x.end()));
    }
    {
      std::array<double, 2> x = {{1, 1}};
      assert(allSame(x.begin(), x.end()));
    }
    {
      std::array<double, 2> x = {{NaN, 1}};
      assert(! allSame(x.begin(), x.end()));
    }
    {
      std::array<double, 2> x = {{1, NaN}};
      assert(! allSame(x.begin(), x.end()));
    }
  }

  // hasUnknownAdjacentNeighbor()
  {
    using levelSet::hasUnknownAdjacentNeighbor;
    typedef container::SimpleMultiArray<Number, 1> Array;
    typedef Array::IndexList IndexList;
    const IndexList Extents = {{3}};
    Array a(Extents);
    IndexList i;
    {
      a[0] = 0;
      a[1] = 0;
      a[2] = 0;
      i[0] = 0;
      assert(! hasUnknownAdjacentNeighbor(a, i));
      i[0] = 1;
      assert(! hasUnknownAdjacentNeighbor(a, i));
      i[0] = 2;
      assert(! hasUnknownAdjacentNeighbor(a, i));
    }
    {
      a[0] = NaN;
      a[1] = 0;
      a[2] = 0;
      i[0] = 0;
      assert(! hasUnknownAdjacentNeighbor(a, i));
      i[0] = 1;
      assert(hasUnknownAdjacentNeighbor(a, i));
      i[0] = 2;
      assert(! hasUnknownAdjacentNeighbor(a, i));
    }
    {
      a[0] = 0;
      a[1] = NaN;
      a[2] = 0;
      i[0] = 0;
      assert(hasUnknownAdjacentNeighbor(a, i));
      i[0] = 1;
      assert(! hasUnknownAdjacentNeighbor(a, i));
      i[0] = 2;
      assert(hasUnknownAdjacentNeighbor(a, i));
    }
    {
      a[0] = 0;
      a[1] = 0;
      a[2] = NaN;
      i[0] = 0;
      assert(! hasUnknownAdjacentNeighbor(a, i));
      i[0] = 1;
      assert(hasUnknownAdjacentNeighbor(a, i));
      i[0] = 2;
      assert(! hasUnknownAdjacentNeighbor(a, i));
    }
  }

  // hasNeighborBelowThreshold()
  {
    using levelSet::hasNeighborBelowThreshold;
    typedef container::SimpleMultiArray<Number, 1> Array;
    typedef Array::IndexList IndexList;
    const IndexList Extents = {{3}};
    Array a(Extents);
    IndexList i;
    const Number T = 1;
    {
      a[0] = T;
      a[1] = T;
      a[2] = T;
      i[0] = 0;
      assert(! hasNeighborBelowThreshold(a, i, T));
      i[0] = 1;
      assert(! hasNeighborBelowThreshold(a, i, T));
      i[0] = 2;
      assert(! hasNeighborBelowThreshold(a, i, T));
    }
    {
      a[0] = T - 1;
      a[1] = T;
      a[2] = T;
      i[0] = 0;
      assert(! hasNeighborBelowThreshold(a, i, T));
      i[0] = 1;
      assert(hasNeighborBelowThreshold(a, i, T));
      i[0] = 2;
      assert(! hasNeighborBelowThreshold(a, i, T));
    }
    {
      a[0] = T;
      a[1] = T - 1;
      a[2] = T;
      i[0] = 0;
      assert(hasNeighborBelowThreshold(a, i, T));
      i[0] = 1;
      assert(! hasNeighborBelowThreshold(a, i, T));
      i[0] = 2;
      assert(hasNeighborBelowThreshold(a, i, T));
    }
    {
      a[0] = T;
      a[1] = T;
      a[2] = T - 1;
      i[0] = 0;
      assert(! hasNeighborBelowThreshold(a, i, T));
      i[0] = 1;
      assert(hasNeighborBelowThreshold(a, i, T));
      i[0] = 2;
      assert(! hasNeighborBelowThreshold(a, i, T));
    }
  }

  // countKnown()
  {
    using levelSet::countKnown;
    {
      std::array<double, 2> x = {{NaN, NaN}};
      assert(countKnown(x.begin(), x.end()) == 0);
    }
    {
      std::array<double, 2> x = {{Inf, NaN}};
      assert(countKnown(x.begin(), x.end()) == 1);
    }
    {
      std::array<double, 2> x = {{Inf, Inf}};
      assert(countKnown(x.begin(), x.end()) == 2);
    }
  }

  // countUnknown()
  {
    using levelSet::countUnknown;
    {
      std::array<double, 2> x = {{NaN, NaN}};
      assert(countUnknown(x.begin(), x.end()) == 2);
    }
    {
      std::array<double, 2> x = {{Inf, NaN}};
      assert(countUnknown(x.begin(), x.end()) == 1);
    }
    {
      std::array<double, 2> x = {{Inf, Inf}};
      assert(countUnknown(x.begin(), x.end()) == 0);
    }
  }

  return 0;
}
