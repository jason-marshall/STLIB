// -*- C++ -*-


#include "stlib/levelSet/boolean.h"
#include "stlib/container/EquilateralArray.h"


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
  using levelSet::areFunctionsEqual;
  using levelSet::complement;
  using levelSet::unite;
  using levelSet::intersect;
  using levelSet::difference;

  //
  // Test functions on two values.
  //
  {
    typedef double T;
    const T NaN = std::numeric_limits<T>::quiet_NaN();
    const T Inf = std::numeric_limits<T>::infinity();
    // Consider the following six values:
    // NaN, Inf, -Inf, 1, 0, -1

    //
    // unite()
    //
    // NaN, ___
    assert(isNaN(unite(NaN, NaN)));
    assert(isNaN(unite(NaN, Inf)));
    assert(unite(NaN, -Inf) == -Inf);
    assert(isNaN(unite(NaN, 1.)));
    assert(unite(NaN, 0.) == 0);
    assert(unite(NaN, -1.) == -1);

    // Inf, ___
    assert(isNaN(unite(Inf, NaN)));
    assert(unite(Inf, Inf) == Inf);
    assert(unite(Inf, -Inf) == -Inf);
    assert(unite(Inf, 1.) == 1);
    assert(unite(Inf, 0.) == 0);
    assert(unite(Inf, -1.) == -1);

    // -Inf, ___
    assert(unite(-Inf, NaN) == -Inf);
    assert(unite(-Inf, Inf) == -Inf);
    assert(unite(-Inf, -Inf) == -Inf);
    assert(unite(-Inf, 1.) == -Inf);
    assert(unite(-Inf, 0.) == -Inf);
    assert(unite(-Inf, -1.) == -Inf);

    // 1, ___
    assert(isNaN(unite(1., NaN)));
    assert(unite(1., Inf) == 1);
    assert(unite(1., -Inf) == -Inf);
    assert(unite(1., 1.) == 1);
    assert(unite(1., 0.) == 0);
    assert(unite(1., -1.) == -1);

    // 0, ___
    assert(unite(0., NaN) == 0);
    assert(unite(0., Inf) == 0);
    assert(unite(0., -Inf) == -Inf);
    assert(unite(0., 1.) == 0);
    assert(unite(0., 0.) == 0);
    assert(unite(0., -1.) == -1);

    // -1, ___
    assert(unite(-1., NaN) == -1);
    assert(unite(-1., Inf) == -1);
    assert(unite(-1., -Inf) == -Inf);
    assert(unite(-1., 1.) == -1);
    assert(unite(-1., 0.) == -1);
    assert(unite(-1., -1.) == -1);

    //
    // intersect()
    //
    // NaN, ___
    assert(isNaN(intersect(NaN, NaN)));
    assert(intersect(NaN, Inf) == Inf);
    assert(isNaN(intersect(NaN, -Inf)));
    assert(intersect(NaN, 1.) == 1);
    assert(intersect(NaN, 0.) == 0);
    assert(isNaN(intersect(NaN, -1.)));

    // Inf, ___
    assert(intersect(Inf, NaN) == Inf);
    assert(intersect(Inf, Inf) == Inf);
    assert(intersect(Inf, -Inf) == Inf);
    assert(intersect(Inf, 1.) == Inf);
    assert(intersect(Inf, 0.) == Inf);
    assert(intersect(Inf, -1.) == Inf);

    // -Inf, ___
    assert(isNaN(intersect(-Inf, NaN)));
    assert(intersect(-Inf, Inf) == Inf);
    assert(intersect(-Inf, -Inf) == -Inf);
    assert(intersect(-Inf, 1.) == 1);
    assert(intersect(-Inf, 0.) == 0);
    assert(intersect(-Inf, -1.) == -1);

    // 1, ___
    assert(intersect(1., NaN) == 1);
    assert(intersect(1., Inf) == Inf);
    assert(intersect(1., -Inf) == 1);
    assert(intersect(1., 1.) == 1);
    assert(intersect(1., 0.) == 1);
    assert(intersect(1., -1.) == 1);

    // 0, ___
    assert(intersect(0., NaN) == 0);
    assert(intersect(0., Inf) == Inf);
    assert(intersect(0., -Inf) == 0);
    assert(intersect(0., 1.) == 1);
    assert(intersect(0., 0.) == 0);
    assert(intersect(0., -1.) == 0);

    // -1, ___
    assert(isNaN(intersect(-1., NaN)));
    assert(intersect(-1., Inf) == Inf);
    assert(intersect(-1., -Inf) == -1);
    assert(intersect(-1., 1.) == 1);
    assert(intersect(-1., 0.) == 0);
    assert(intersect(-1., -1.) == -1);

    //
    // difference()
    //
    // NaN, ___
    assert(isNaN(difference(NaN, NaN)));
    assert(isNaN(difference(NaN, Inf)));
    assert(isNaN(difference(NaN, Inf)));
    assert(isNaN(difference(NaN, 1.)));
    assert(difference(NaN, 0.) == 0);
    assert(difference(NaN, -1.) == 1);

    // Inf, ___
    assert(difference(Inf, NaN) == Inf);
    assert(difference(Inf, Inf) == Inf);
    assert(difference(Inf, -Inf) == Inf);
    assert(difference(Inf, 1.) == Inf);
    assert(difference(Inf, 0.) == Inf);
    assert(difference(Inf, -1.) == Inf);

    // -Inf, ___
    assert(isNaN(difference(-Inf, NaN)));
    assert(difference(-Inf, Inf) == -Inf);
    assert(difference(-Inf, -Inf) == Inf);
    assert(difference(-Inf, 1.) == -1);
    assert(difference(-Inf, 0.) == 0);
    assert(difference(-Inf, -1.) == 1);

    // 1, ___
    assert(difference(1., NaN) == 1);
    assert(difference(1., Inf) == 1);
    assert(difference(1., -Inf) == Inf);
    assert(difference(1., 1.) == 1);
    assert(difference(1., 0.) == 1);
    assert(difference(1., -1.) == 1);

    // 0, ___
    assert(difference(0., NaN) == 0);
    assert(difference(0., Inf) == 0);
    assert(difference(0., -Inf) == Inf);
    assert(difference(0., 1.) == 0);
    assert(difference(0., 0.) == 0);
    assert(difference(0., -1.) == 1);

    // -1, ___
    assert(isNaN(difference(-1., NaN)));
    assert(difference(-1., Inf) == -1);
    assert(difference(-1., -Inf) == Inf);
    assert(difference(-1., 1.) == -1);
    assert(difference(-1., 0.) == 0);
    assert(difference(-1., -1.) == 1);
  }

  //
  // Test functions on arrays.
  //
  {
    typedef double T;
    const T N = std::numeric_limits<T>::quiet_NaN();
    const T I = std::numeric_limits<T>::infinity();
    const std::size_t Size = 32;
    // Consider the following six values:
    // N, I, -I, 1, 0, -1
    //        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
    T fd[] = {N, I, 1, 0, -1, -I, -1, 0, 1, I, N, N, N, N, N, N,
              N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N
             };

    T gd[] = {N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N,
              N, I, 1, 0, -1, -I, -1, 0, 1, I, N, N, N, N, N, N
             };

    T cd[] = {N, -I, -1, 0, 1, I, 1, 0, -1, -I, N, N, N, N, N, N,
              N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N
             };

    T ud[] = {N, N, N, 0, -1, -I, -1, 0, N, N, N, N, N, N, N, N,
              N, N, N, 0, -1, -I, -1, 0, N, N, N, N, N, N, N, N
             };

    T id[] = {N, I, 1, 0, N, N, N, 0, 1, I, N, N, N, N, N, N,
              N, I, 1, 0, N, N, N, 0, 1, I, N, N, N, N, N, N
             };

    T dd[] = {N, I, 1, 0, N, N, N, 0, 1, I, N, N, N, N, N, N,
              N, N, N, 0, 1, I, 1, 0, N, N, N, N, N, N, N, N
             };

    {
      typedef container::SimpleMultiArray<T, 1> Array;
      typedef Array::IndexList IndexList;
      const IndexList Extents = {{Size}};

      Array f(Extents), g(Extents), c(Extents), u(Extents), i(Extents),
            d(Extents);
      std::copy(fd, fd + f.size(), f.begin());
      std::copy(gd, gd + g.size(), g.begin());
      std::copy(cd, cd + c.size(), c.begin());
      std::copy(ud, ud + u.size(), u.begin());
      std::copy(id, id + i.size(), i.begin());
      std::copy(dd, dd + d.size(), d.begin());

      complement(&f);
      assert(areFunctionsEqual(f, c));

      std::copy(fd, fd + f.size(), f.begin());
      unite(&f, g);
      assert(areFunctionsEqual(f, u));

      std::copy(fd, fd + f.size(), f.begin());
      intersect(&f, g);
      assert(areFunctionsEqual(f, i));

      std::copy(fd, fd + f.size(), f.begin());
      difference(&f, g);
      assert(areFunctionsEqual(f, d));
    }
    {
      typedef levelSet::Grid<T, 1, Size> Grid;
      typedef Grid::BBox BBox;

      const BBox D = {{{0}}, {{1}}};
      const T s = 1;
      Grid f(D, s), g(D, s), r(D, s), c(D, s), u(D, s), i(D, s), d(D, s);
      std::vector<std::size_t> patchList(1, 0);
      f.refine(patchList);
      g.refine(patchList);
      r.refine(patchList);
      c.refine(patchList);
      u.refine(patchList);
      i.refine(patchList);
      d.refine(patchList);
      std::copy(fd, fd + Size, f[0].begin());
      std::copy(gd, gd + Size, g[0].begin());
      std::copy(cd, cd + Size, c[0].begin());
      std::copy(ud, ud + Size, u[0].begin());
      std::copy(id, id + Size, i[0].begin());
      std::copy(dd, dd + Size, d[0].begin());

      assert(areFunctionsEqual(f, f));

      complement(&f);
      assert(areFunctionsEqual(f, c));
      std::copy(fd, fd + Size, f[0].begin());

      unite(f, g, &r);
      assert(areFunctionsEqual(r, u));

      intersect(f, g, &r);
      assert(areFunctionsEqual(r, i));

      difference(f, g, &r);
      assert(areFunctionsEqual(r, d));
    }
  }

  return 0;
}
