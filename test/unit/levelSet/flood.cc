// -*- C++ -*-


#include "stlib/levelSet/flood.h"
#include "stlib/levelSet/boolean.h"


using namespace stlib;
using levelSet::floodFill;
using levelSet::floodFillInterval;
using levelSet::areFunctionsEqual;


int
main()
{
  typedef double T;
  typedef container::SimpleMultiArray<T, 1> Array;
  typedef Array::IndexList IndexList;
  const IndexList Extents = {{10}};
  const T Max = std::numeric_limits<T>::max();
  const T NaN = std::numeric_limits<T>::quiet_NaN();
  const T Inf = std::numeric_limits<T>::infinity();
  Array f(Extents), g(Extents);
  {
    //           0    1    2    3    4    5    6    7    8    9
    T fd[] = { NaN, Inf,   1,  -1, -Inf,  -1,   1, Inf, NaN, NaN};
    T gd[] = { Inf, Inf,   1,  -1, -Inf,  -1,   1, Inf, Inf, Inf};
    std::copy(fd, fd + f.size(), f.begin());
    std::copy(gd, gd + g.size(), g.begin());
    floodFill(&f);
    assert(areFunctionsEqual(f, g));

    std::copy(fd, fd + f.size(), f.begin());
    floodFillInterval(&f, -Max, Max);
    assert(areFunctionsEqual(f, g));
  }
  {
    const T Threshold = std::numeric_limits<T>::max();
    const T V = std::numeric_limits<T>::max();
    //           0    1    2    3    4    5    6    7    8    9
    T fd[] = { NaN, Inf,   1,  -1, -Inf,  -1,   1, Inf, NaN, NaN};
    T gd[] = {   V,   V,   1,  -1,  -V,  -1,   1,   V,   V,   V};
    std::copy(fd, fd + f.size(), f.begin());
    std::copy(gd, gd + g.size(), g.begin());
    floodFill(&f, Threshold, V);
    assert(areFunctionsEqual(f, g));

    std::copy(fd, fd + f.size(), f.begin());
    floodFillInterval(&f, -Threshold, Threshold, -V, V);
    assert(areFunctionsEqual(f, g));
  }
  {
    const T Threshold = std::numeric_limits<T>::max();
    const T V = 1;
    //           0    1    2    3    4    5    6    7    8    9
    T fd[] = { NaN, Inf,   1,  -1, -Inf,  -1,   1, Inf, NaN, NaN};
    T gd[] = {   V,   V,   1,  -1,  -V,  -1,   1,   V,   V,   V};
    std::copy(fd, fd + f.size(), f.begin());
    std::copy(gd, gd + g.size(), g.begin());
    floodFill(&f, Threshold, V);
    assert(areFunctionsEqual(f, g));

    std::copy(fd, fd + f.size(), f.begin());
    floodFillInterval(&f, -Threshold, Threshold, -V, V);
    assert(areFunctionsEqual(f, g));
  }

  {
    //           0    1    2    3    4    5    6    7    8    9
    T fd[] = { NaN, NaN,   1,  -1, NaN, NaN, NaN, NaN, NaN, NaN};
    T gd[] = { Inf, Inf,   1,  -1, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf};
    std::copy(fd, fd + f.size(), f.begin());
    std::copy(gd, gd + g.size(), g.begin());
    floodFill(&f);
    assert(areFunctionsEqual(f, g));

    std::copy(fd, fd + f.size(), f.begin());
    floodFillInterval(&f, -Max, Max);
    assert(areFunctionsEqual(f, g));
  }
  {
    //           0    1    2    3    4    5    6    7    8    9
    T fd[] = { NaN, NaN,  -1,   1, NaN, NaN, NaN, NaN, NaN, NaN};
    T gd[] = { -Inf, -Inf,  -1,   1, Inf, Inf, Inf, Inf, Inf, Inf};
    std::copy(fd, fd + f.size(), f.begin());
    std::copy(gd, gd + g.size(), g.begin());
    floodFill(&f);
    assert(areFunctionsEqual(f, g));

    std::copy(fd, fd + f.size(), f.begin());
    floodFillInterval(&f, -Max, Max);
    assert(areFunctionsEqual(f, g));
  }
  {
    //           0    1    2    3    4    5    6    7    8    9
    T fd[] = { NaN, NaN,   1, NaN, NaN, NaN, NaN, NaN, NaN, NaN};
    T gd[] = { Inf, Inf,   1, Inf, Inf, Inf, Inf, Inf, Inf, Inf};
    std::copy(fd, fd + f.size(), f.begin());
    std::copy(gd, gd + g.size(), g.begin());
    floodFill(&f);
    assert(areFunctionsEqual(f, g));

    std::copy(fd, fd + f.size(), f.begin());
    floodFillInterval(&f, -Max, Max);
    assert(areFunctionsEqual(f, g));
  }
  {
    //           0    1    2    3    4    5    6    7    8    9
    T fd[] = { NaN, NaN,  -1, NaN, NaN, NaN, NaN, NaN, NaN, NaN};
    T gd[] = { -Inf, -Inf,  -1, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf};
    std::copy(fd, fd + f.size(), f.begin());
    std::copy(gd, gd + g.size(), g.begin());
    floodFill(&f);
    assert(areFunctionsEqual(f, g));

    std::copy(fd, fd + f.size(), f.begin());
    floodFillInterval(&f, -Max, Max);
    assert(areFunctionsEqual(f, g));
  }

  return 0;
}
