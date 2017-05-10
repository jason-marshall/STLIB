// -*- C++ -*-

#include "stlib/geom/kernel/Hyperplane.h"

#include "stlib/geom/kernel/content.h"

#include <iostream>

#include <cassert>


template<typename _T>
void
test1()
{
  USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
  using stlib::geom::supportingHyperplane;

  constexpr std::size_t Dimension = 1;
  using Hyperplane = stlib::geom::Hyperplane<_T, Dimension>;
  using Point = typename Hyperplane::Point;
  using Simplex = std::array<Point, Dimension + 1>;

  assert(supportingHyperplane(Simplex{{{{1}}, {{3}}}}, 0) ==
         (Hyperplane{{{3}}, {{1}}}));
  assert(supportingHyperplane(Simplex{{{{1}}, {{3}}}}, 1) ==
         (Hyperplane{{{1}}, {{-1}}}));

  try {
    supportingHyperplane(Simplex{{{{1}}, {{1}}}}, 0);
    assert(false);
  }
  catch (std::runtime_error) {
  }
}


template<typename _T>
void
test2()
{
  USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
  using stlib::geom::supportingHyperplane;

  constexpr std::size_t Dimension = 2;
  using Hyperplane = stlib::geom::Hyperplane<_T, Dimension>;
  using Point = typename Hyperplane::Point;
  using Simplex = std::array<Point, Dimension + 1>;

  assert(supportingHyperplane(Simplex{{{{0, 0}}, {{1, 0}}, {{0, 1}}}}, 1) ==
         (Hyperplane{{{0, 0}}, {{-1, 0}}}));
  assert(supportingHyperplane(Simplex{{{{0, 0}}, {{1, 0}}, {{0, 1}}}}, 2) ==
         (Hyperplane{{{0, 0}}, {{0, -1}}}));

  assert(supportingHyperplane(Simplex{{{{0, 0}}, {{0, 1}}, {{1, 0}}}}, 1) ==
         (Hyperplane{{{0, 0}}, {{0, 1}}}));
  assert(supportingHyperplane(Simplex{{{{0, 0}}, {{0, 1}}, {{1, 0}}}}, 2) ==
         (Hyperplane{{{0, 0}}, {{1, 0}}}));

  assert(supportingHyperplane(Simplex{{{{0, 0}}, {{1, 0}}, {{1, 1}}}}, 0) ==
         (Hyperplane{{{1, 0}}, {{1, 0}}}));
  assert(supportingHyperplane(Simplex{{{{0, 0}}, {{1, 0}}, {{1, 1}}}}, 2) ==
         (Hyperplane{{{0, 0}}, {{0, -1}}}));

  // The simplex is degenerate, but not the selected face.
  assert(supportingHyperplane(Simplex{{{{0, 0}}, {{1, 0}}, {{0, 0}}}}, 0) ==
         (Hyperplane{{{1, 0}}, {{0, 1}}}));
  assert(supportingHyperplane(Simplex{{{{0, 0}}, {{1, 0}}, {{0, 0}}}}, 2) ==
         (Hyperplane{{{0, 0}}, {{0, -1}}}));

  try {
    supportingHyperplane(Simplex{{{{0, 0}}, {{1, 0}}, {{1, 0}}}}, 0);
    assert(false);
  }
  catch (std::runtime_error) {
  }
}


template<typename _T>
void
test3()
{
  USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

  using Hyperplane = stlib::geom::Hyperplane<_T, 3>;
  using Point = typename Hyperplane::Point;
  using Simplex = std::array<Point, Hyperplane::Dimension + 1>;
  using stlib::geom::supportingHyperplane;

  {
    // Tetrahedron with positive volume.
    Simplex const s = {{{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}}};
    for (std::size_t i = 0; i != s.size(); ++i) {
      assert(signedDistance(supportingHyperplane(s, i), s[i]) < 0);
    }
    assert(supportingHyperplane(s, 1) ==
           (Hyperplane{{{0, 0, 0}}, {{-1, 0, 0}}}));
    assert(supportingHyperplane(s, 2) ==
           (Hyperplane{{{0, 0, 0}}, {{0, -1, 0}}}));
    assert(supportingHyperplane(s, 3) ==
           (Hyperplane{{{0, 0, 0}}, {{0, 0, -1}}}));
  }
  {
    // Tetrahedron with negative volume.
    Simplex const s = {{{{0, 0, 0}}, {{1, 0, 0}}, {{0, 0, 1}}, {{0, 1, 0}}}};
    for (std::size_t i = 0; i != s.size(); ++i) {
      assert(signedDistance(supportingHyperplane(s, i), s[i]) > 0);
    }
    assert(supportingHyperplane(s, 1) ==
           (Hyperplane{{{0, 0, 0}}, {{1, 0, 0}}}));
    assert(supportingHyperplane(s, 2) ==
           (Hyperplane{{{0, 0, 0}}, {{0, 0, 1}}}));
    assert(supportingHyperplane(s, 3) ==
           (Hyperplane{{{0, 0, 0}}, {{0, 1, 0}}}));
  }

  {
    // Brace initialization.
    Point nm = {{1, 2, 3}};
    stlib::ext::normalize(&nm);
    Hyperplane const p = {Point{{2, 3, 5}}, nm};
    assert(isValid(p));
    assert((p.point == Point{{2, 3, 5}}));
    assert(p.normal == nm);

    assert(! isValid(Hyperplane{{{0, 0, 0}}, {{0, 0, 0}}}));
    assert(isValid(Hyperplane{{{0, 0, 0}}, {{1, 0, 0}}}));
    assert(! isValid(Hyperplane{{{0, 0, 0}}, {{1, 1, 0}}}));
    assert(! isValid(Hyperplane{{{std::numeric_limits<_T>::quiet_NaN(), 0, 0}},
          {{1, 0, 0}}}));
  }
  {
    // Equality/inequality.
    assert((Hyperplane{{{1, 2, 3}}, {{1, 0, 0}}}) == 
           (Hyperplane{{{1, 2, 3}}, {{1, 0, 0}}}));
    assert((Hyperplane{{{1, 2, 3}}, {{1, 0, 0}}}) == 
           (Hyperplane{{{1, 3, 4}}, {{1, 0, 0}}}));
    assert((Hyperplane{{{1, 2, 3}}, {{1, 0, 0}}}) != 
           (Hyperplane{{{2, 2, 3}}, {{1, 0, 0}}}));
    assert((Hyperplane{{{1, 2, 3}}, {{1, 0, 0}}}) != 
           (Hyperplane{{{1, 2, 3}}, {{0, 1, 0}}}));
  }
  {
    // +=
    Hyperplane x = {{{1, 2, 3}}, {{1, 0, 0}}};
    x += Point{{2, 3, 5}};
    assert(x == (Hyperplane{{{3, 5, 8}}, {{1, 0, 0}}}));
  }
  {
    // -=
    Hyperplane x = {{{1, 2, 3}}, {{1, 0, 0}}};
    x -= Point{{2, 3, 5}};
    assert(x == (Hyperplane{{{-1, -1, -2}}, {{1, 0, 0}}}));
  }
  {
    // unary +
    assert((Hyperplane{{{1, 2, 3}}, {{0, 1, 0}}}) ==
           (+Hyperplane{{{1, 2, 3}}, {{0, 1, 0}}}));
  }
  {
    // unary -
    assert((Hyperplane{{{1, 2, 3}}, {{0, 1, 0}}}) ==
           (-Hyperplane{{{1, 2, 3}}, {{0, -1, 0}}}));
  }
  {
    // distance
    Hyperplane const p = {{{1, 2, 3}}, {{1, 0, 0}}};
    assert(signedDistance(p, Point{{2, 2, 3}}) == 1);
    assert(signedDistance(p, Point{{1, 2, 3}}) == 0);
    assert(signedDistance(p, Point{{1, 2, 4}}) == 0);
    assert(signedDistance(p, Point{{0, 2, 3}}) == -1);
    // closest point
    Point cpt;
    assert(signedDistance(p, Point{{2, 2, 3}}, &cpt) == 1);
    assert(cpt == (Point{{1, 2, 3}}));
    assert(signedDistance(p, Point{{1, 2, 3}}, &cpt) == 0);
    assert(cpt == (Point{{1, 2, 3}}));
    assert(signedDistance(p, Point{{1, 2, 4}}, &cpt) == 0);
    assert(cpt == (Point{{1, 2, 4}}));
    assert(signedDistance(p, Point{{0, 2, 3}}, &cpt) == -1);
    assert(cpt == (Point{{1, 2, 3}}));
  }
}


template<typename _T, std::size_t _D>
void
test()
{
  using Hyperplane = stlib::geom::Hyperplane<_T, _D>;
  using Point = typename Hyperplane::Point;

  {
    // Default constructor
    Hyperplane p;
    std::cout << "Hyperplane{} = " << p << '\n';
  }
  {
    // Brace initialization.
    {
      Hyperplane const x = {{{}}, {{1}}};
      assert(isValid(x));
      assert(x.point == Point{{}});
      assert(x.normal == Point{{1}});
    }
    assert(! isValid(Hyperplane{{{}}, {{}}}));
    assert(isValid(Hyperplane{{{}}, {{1}}}));
    assert(! isValid(Hyperplane{{{}}, {{2}}}));
    assert(! isValid(Hyperplane{{{std::numeric_limits<_T>::quiet_NaN()}},
          {{1}}}));
  }
  {
    // Equality/inequality.
    assert((Hyperplane{{{1}}, {{1}}}) == 
           (Hyperplane{{{1}}, {{1}}}));
    assert((Hyperplane{{{1}}, {{1}}}) != 
           (Hyperplane{{{2}}, {{1}}}));
    assert((Hyperplane{{{1}}, {{1}}}) != 
           (Hyperplane{{{1}}, {{-1}}}));
  }
  {
    // +=
    Hyperplane x = {{{0}}, {{1}}};
    x += Point{{2}};
    assert(x == (Hyperplane{{{2}}, {{1}}}));
  }
  {
    // -=
    Hyperplane x = {{{0}}, {{1}}};
    x -= Point{{2}};
    assert(x == (Hyperplane{{{-2}}, {{1}}}));
  }
  {
    // unary +
    assert((Hyperplane{{{2}}, {{1}}}) == (+Hyperplane{{{2}}, {{1}}}));
  }
  {
    // unary -
    assert((Hyperplane{{{2}}, {{1}}}) == (-Hyperplane{{{2}}, {{-1}}}));
  }
  {
    // distance
    Hyperplane const p = {{{1}}, {{1}}};
    assert(signedDistance(p, Point{{2}}) == 1);
    assert(signedDistance(p, Point{{1}}) == 0);
    assert(signedDistance(p, Point{{0}}) == -1);
    // closest point
    Point cpt;
    assert(signedDistance(p, Point{{2}}, &cpt) == 1);
    assert(cpt == (Point{{1}}));
    assert(signedDistance(p, Point{{1}}, &cpt) == 0);
    assert(cpt == (Point{{1}}));
    assert(signedDistance(p, Point{{0}}, &cpt) == -1);
    assert(cpt == (Point{{1}}));
  }
}


int
main()
{
  test1<float>();
  test1<double>();
  test2<float>();
  test2<double>();
  test3<float>();
  test3<double>();

  test<float, 1>();
  test<double, 1>();
  test<float, 2>();
  test<double, 2>();
  test<float, 3>();
  test<double, 3>();

  return 0;
}

