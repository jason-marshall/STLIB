// -*- C++ -*-

#include "stlib/geom/shape/procrustes.h"
#include "stlib/ext/vector.h"
#include "stlib/numerical/equality.h"

#include <iostream>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

template<typename _T, std::size_t _D>
void
rotate(const std::vector<std::array<_T, _D> >& source,
       const container::EquilateralArray<_T, 2, _D>& rotation,
       std::vector<std::array<_T, _D> >* transformed)
{
  for (std::size_t i = 0; i != transformed->size(); ++i) {
    for (std::size_t j = 0; j != _D; ++j) {
      (*transformed)[i][j] = 0;
      for (std::size_t k = 0; k != _D; ++k) {
        (*transformed)[i][j] += source[i][k] * rotation(k, j);
      }
    }
  }
}


template<typename _T, std::size_t _D>
void
transform(const std::vector<std::array<_T, _D> >& source,
          const container::EquilateralArray<_T, 2, _D>& rotation,
          const _T scale,
          std::vector<std::array<_T, _D> >* transformed)
{
  for (std::size_t i = 0; i != transformed->size(); ++i) {
    for (std::size_t j = 0; j != _D; ++j) {
      (*transformed)[i][j] = 0;
      for (std::size_t k = 0; k != _D; ++k) {
        (*transformed)[i][j] += source[i][k] * rotation(k, j);
      }
      (*transformed)[i][j] *= scale;
    }
  }
}


template<typename _T, std::size_t _D>
_T
rmsd(const std::vector<std::array<_T, _D> >& x,
     const std::vector<std::array<_T, _D> >& y)
{
  assert(x.size() == y.size());
  if (x.empty()) {
    return 0;
  }
  _T rmsd = 0;
  for (std::size_t i = 0; i != x.size(); ++i) {
    rmsd += stlib::ext::squaredDistance(x[i], y[i]);
  }
  return rmsd / x.size();
}


void amosTest();


int
main()
{
  using numerical::areEqual;
  amosTest();

  //-------------------------------------------------------------------------
  // 2-D, 2 points.
  std::cout << "\n2-D, 2 points:\n";
  {
    typedef float Number;
    const std::size_t Dimension = 2;
    typedef std::array<Number, Dimension> Point;

    std::vector<Point> s(2), source(2), t(2), target(2), transformed(2);
    std::array<Number, Dimension> sourceCentroid, targetCentroid;
    container::EquilateralArray<Number, 2, Dimension> rotation;
    Number scale;

    //
    // Centered.
    //
    std::cout << "Centered:\n";
    s[0][0] = -1;
    s[0][1] = 0;
    s[1][0] = 1;
    s[1][1] = 0;
    t[0][0] = -1;
    t[0][1] = 0;
    t[1][0] = 1;
    t[1][1] = 0;
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &rotation, &scale);
    assert(areEqual(sourceCentroid, Point{{0, 0}}));
    assert(areEqual(targetCentroid, Point{{0, 0}}));
    transform(source, rotation, scale, &transformed);
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      assert(areEqual(transformed[i], target[i]));
    }
#if 0
    std::cout << "\nsource =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "rotation =\n"
              << rotation(0, 0) << ' ' << rotation(0, 1) << '\n'
              << rotation(1, 0) << ' ' << rotation(1, 1) << '\n'
              << "scale = " << scale << '\n';
#endif
    std::cout << "rmsd = " << rmsd(target, transformed) << "\n";

    s[0][0] = -1;
    s[0][1] = 0;
    s[1][0] = 1;
    s[1][1] = 0;
    t[0][0] = 0;
    t[0][1] = -1;
    t[1][0] = 0;
    t[1][1] = 1;
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &rotation, &scale);
    assert(areEqual(sourceCentroid, Point{{0, 0}}));
    assert(areEqual(targetCentroid, Point{{0, 0}}));
    transform(source, rotation, scale, &transformed);
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      assert(areEqual(transformed[i], target[i]));
    }
#if 0
    std::cout << "\nsource =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "rotation =\n"
              << rotation(0, 0) << ' ' << rotation(0, 1) << '\n'
              << rotation(1, 0) << ' ' << rotation(1, 1) << '\n'
              << "scale = " << scale << '\n';
#endif
    std::cout << "rmsd = " << rmsd(target, transformed) << "\n";

    // [-1 0] * [ 0 1] = [0 -1]
    // [ 1 0]   [-1 0]   [0  1]

    s[0][0] = -1;
    s[0][1] = 0;
    s[1][0] = 1;
    s[1][1] = 0;
    t[0][0] = 1;
    t[0][1] = 0;
    t[1][0] = -1;
    t[1][1] = 0;
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &rotation, &scale);
    assert(areEqual(sourceCentroid, Point{{0, 0}}));
    assert(areEqual(targetCentroid, Point{{0, 0}}));
    transform(source, rotation, scale, &transformed);
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      assert(areEqual(transformed[i], target[i]));
    }
#if 0
    std::cout << "\nsource =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "rotation =\n"
              << rotation(0, 0) << ' ' << rotation(0, 1) << '\n'
              << rotation(1, 0) << ' ' << rotation(1, 1) << '\n'
              << "scale = " << scale << '\n';
#endif
    std::cout << "rmsd = " << rmsd(target, transformed) << "\n\n";

    //
    // Not centered.
    //
    std::cout << "\nNot centered:\n";
    s[0][0] = 0;
    s[0][1] = 0;
    s[1][0] = 2;
    s[1][1] = 0;
    t[0][0] = 0;
    t[0][1] = 0;
    t[1][0] = 0;
    t[1][1] = 2;
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &rotation, &scale);
    assert(areEqual(sourceCentroid, Point{{1, 0}}));
    assert(areEqual(targetCentroid, Point{{0, 1}}));
    transform(source, rotation, scale, &transformed);
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      assert(areEqual(transformed[i], target[i]));
    }
#if 0
    std::cout << "\nsource =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "rotation =\n"
              << rotation(0, 0) << ' ' << rotation(0, 1) << '\n'
              << rotation(1, 0) << ' ' << rotation(1, 1) << '\n'
              << "scale = " << scale << '\n';
#endif
    std::cout << "rmsd = " << rmsd(target, transformed) << "\n\n";

    //
    // Scaled.
    //
    std::cout << "Scaled:\n";
    s[0][0] = -1;
    s[0][1] = 0;
    s[1][0] = 1;
    s[1][1] = 0;
    t[0][0] = -2;
    t[0][1] = 0;
    t[1][0] = 2;
    t[1][1] = 0;
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &rotation, &scale);
    assert(areEqual(sourceCentroid, Point{{0, 0}}));
    assert(areEqual(targetCentroid, Point{{0, 0}}));
    assert(areEqual(scale, Number(2.)));
    transform(source, rotation, scale, &transformed);
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      //assert(areEqual(transformed[i], target[i]));
    }
#if 0
    std::cout << "Scaled:\n"
              << "source =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "rotation =\n"
              << rotation(0, 0) << ' ' << rotation(0, 1) << '\n'
              << rotation(1, 0) << ' ' << rotation(1, 1) << '\n'
              << "scale = " << scale << "\n\n";
#endif
    std::cout << "rmsd = " << rmsd(target, transformed) << "\n";

  }

  //-------------------------------------------------------------------------
  // 2-D, 3 points.
  std::cout << "\n2-D, 3 points:\n";
  {
    typedef float Number;
    const std::size_t Dimension = 2;
    typedef std::array<Number, Dimension> Point;

    std::vector<Point> s(3), source(3), t(3), target(3), transformed(3);
    std::array<Number, Dimension> sourceCentroid, targetCentroid;
    container::EquilateralArray<Number, 2, Dimension> rotation;
    Number scale;

    //
    // Centered.
    //
    s[0][0] = -1;
    s[0][1] = 0;
    s[1][0] = 0;
    s[1][1] = 0;
    s[2][0] = 1;
    s[2][1] = 0;
    t[0][0] = -1;
    t[0][1] = 0;
    t[1][0] = 0;
    t[1][1] = 0;
    t[2][0] = 1;
    t[2][1] = 0;
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &rotation, &scale);
    assert(areEqual(sourceCentroid, Point{{0, 0}}));
    assert(areEqual(targetCentroid, Point{{0, 0}}));
    transform(source, rotation, scale, &transformed);
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      assert(areEqual(transformed[i], target[i]));
    }
#if 0
    std::cout << "\nsource =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "rotation =\n"
              << rotation(0, 0) << ' ' << rotation(0, 1) << '\n'
              << rotation(1, 0) << ' ' << rotation(1, 1) << '\n'
              << "scale = " << scale << '\n';
#endif

    s[0][0] = -1;
    s[0][1] = 0;
    s[1][0] = 0;
    s[1][1] = 0;
    s[2][0] = 1;
    s[2][1] = 0;
    t[0][0] = 0;
    t[0][1] = -1;
    t[1][0] = 0;
    t[1][1] = 0;
    t[2][0] = 0;
    t[2][1] = 1;
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &rotation, &scale);
    assert(areEqual(sourceCentroid, Point{{0, 0}}));
    assert(areEqual(targetCentroid, Point{{0, 0}}));
    transform(source, rotation, scale, &transformed);
#if 0
    std::cout << "\nsource =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "rotation =\n"
              << rotation(0, 0) << ' ' << rotation(0, 1) << '\n'
              << rotation(1, 0) << ' ' << rotation(1, 1) << '\n'
              << "scale = " << scale << '\n';
#endif
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      assert(areEqual(transformed[i], target[i]));
    }

    //
    // Scaled.
    //
    s[0][0] = -1;
    s[0][1] = 0;
    s[1][0] = 0;
    s[1][1] = 0;
    s[2][0] = 1;
    s[2][1] = 0;
    t[0][0] = -2;
    t[0][1] = 0;
    t[1][0] = 0;
    t[1][1] = 0;
    t[2][0] = 2;
    t[2][1] = 0;
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &rotation, &scale);
    assert(areEqual(sourceCentroid, Point{{0, 0}}));
    assert(areEqual(targetCentroid, Point{{0, 0}}));
    assert(areEqual(scale, Number(2.)));
    transform(source, rotation, scale, &transformed);
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      //assert(areEqual(transformed[i], target[i]));
    }
#if 0
    std::cout << "Scaled:\n"
              << "source =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "rotation =\n"
              << rotation(0, 0) << ' ' << rotation(0, 1) << '\n'
              << rotation(1, 0) << ' ' << rotation(1, 1) << '\n'
              << "scale = " << scale << "\n\n";
#endif

    //
    // Bent.
    //
    s[0][0] = -1;
    s[0][1] = 0;
    s[1][0] = 0;
    s[1][1] = 0;
    s[2][0] = 1;
    s[2][1] = 0;
    t[0][0] = 0;
    t[0][1] = -1;
    t[1][0] = 0;
    t[1][1] = 0;
    t[2][0] = std::sin(0.1);
    t[2][1] = std::cos(0.1);
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &rotation, &scale);
    assert(areEqual(sourceCentroid, Point{{0, 0}}));
    //assert(areEqual(targetCentroid, Point{{-1./3, -1./3}}));
    transform(source, rotation, scale, &transformed);
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      //assert(areEqual(transformed[i], target[i]));
    }
#if 0
    std::cout << "Bent:\n"
              << "source =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "rotation =\n"
              << rotation(0, 0) << ' ' << rotation(0, 1) << '\n'
              << rotation(1, 0) << ' ' << rotation(1, 1) << '\n'
              << "scale = " << scale << "\n\n";
#endif

    //
    // Very bent.
    //
    s[0][0] = -1;
    s[0][1] = 0;
    s[1][0] = 0;
    s[1][1] = 0;
    s[2][0] = 1;
    s[2][1] = 0;
    t[0][0] = 0;
    t[0][1] = -1;
    t[1][0] = 0;
    t[1][1] = 0;
    t[2][0] = -1;
    t[2][1] = 0;
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &rotation, &scale);
    assert(areEqual(sourceCentroid, Point{{0, 0}}));
    assert(areEqual(targetCentroid, Point{{-1. / 3, -1. / 3}}));
    transform(source, rotation, scale, &transformed);
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      //assert(areEqual(transformed[i], target[i]));
    }
#if 0
    std::cout << "Very bent:\n"
              << "source =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "rotation =\n"
              << rotation(0, 0) << ' ' << rotation(0, 1) << '\n'
              << rotation(1, 0) << ' ' << rotation(1, 1) << '\n'
              << "scale = " << scale << "\n\n";
#endif
  }

  //-------------------------------------------------------------------------
  // 2-D, 10 points.
  std::cout << "\n2-D, 10 points:\n";
  {
    typedef float Number;
    const std::size_t Dimension = 2;
    typedef std::array<Number, Dimension> Point;

    std::vector<Point> s(10), source(10), t(10), target(10), transformed(10);
    std::array<Number, Dimension> sourceCentroid, targetCentroid;
    container::EquilateralArray<Number, 2, Dimension> rotation, r;
    Number scale;

    rotation(0, 0) = 0;
    rotation(0, 1) = 1;
    rotation(1, 0) = -1;
    rotation(1, 1) = 0;

    s[0][0] = 0;
    s[0][1] = 0;
    s[1][0] = 1;
    s[1][1] = 0;
    s[2][0] = 2;
    s[2][1] = 1;
    s[3][0] = 3;
    s[3][1] = -1;
    s[4][0] = 4;
    s[4][1] = 2;
    s[5][0] = 0;
    s[5][1] = -2;
    s[6][0] = -1;
    s[6][1] = 3;
    s[7][0] = -2;
    s[7][1] = -3;
    s[8][0] = -3;
    s[8][1] = 4;
    s[9][0] = -4;
    s[9][1] = -4;
    rotate(s, rotation, &t);
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &r, &scale);
    assert(areEqual(sourceCentroid, Point{{0, 0}}));
    assert(areEqual(targetCentroid, Point{{0, 0}}));
    assert(areEqual(r, rotation));
    rotate(source, r, &transformed);
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      assert(areEqual(transformed[i], target[i], Number(10)));
    }
#if 0
    std::cout << "\nsource =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "rotation =\n"
              << r(0, 0) << ' ' << r(0, 1) << '\n'
              << r(1, 0) << ' ' << r(1, 1) << '\n'
              << "scale = " << scale << '\n';
#endif
  }

  //-------------------------------------------------------------------------
  // 3-D, 3 points.
  std::cout << "\n3-D, 3 points:\n";
  {
    typedef float Number;
    const std::size_t Dimension = 3;
    typedef std::array<Number, Dimension> Point;

    std::vector<Point> s(3), source(3), t(3), target(3), transformed(3);
    std::array<Number, Dimension> sourceCentroid, targetCentroid;
    container::EquilateralArray<Number, 2, Dimension> r;
    Number scale;

    //
    // Centered.
    //
    s[0][0] = -1;
    s[0][1] = 0;
    s[0][2] = 0;
    s[1][0] = 0;
    s[1][1] = 0;
    s[1][2] = 0;
    s[2][0] = 1;
    s[2][1] = 0;
    s[2][2] = 0;
    t[0][0] = 0;
    t[0][1] = -1;
    t[0][2] = 0;
    t[1][0] = 0;
    t[1][1] = 0;
    t[1][2] = 0;
    t[2][0] = 0;
    t[2][1] = 1;
    t[2][2] = 0;
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &r, &scale);
    assert(areEqual(sourceCentroid, Point{{0, 0, 0}}));
    assert(areEqual(targetCentroid, Point{{0, 0, 0}}));
    rotate(source, r, &transformed);
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      assert(areEqual(transformed[i], target[i]));
    }
    for (std::size_t i = 1; i != transformed.size(); ++i) {
      assert(areEqual(stlib::ext::euclideanDistance(source[i], source[i - 1]),
                      stlib::ext::euclideanDistance(s[i], s[i - 1])));
    }
#if 0
    std::cout << "\nsource =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "r =\n"
              << r(0, 0) << ' ' << r(0, 1) << ' ' << r(0, 2) << '\n'
              << r(1, 0) << ' ' << r(1, 1) << ' ' << r(1, 2) << '\n'
              << r(2, 0) << ' ' << r(2, 1) << ' ' << r(2, 2) << '\n'
              << "scale = " << scale << '\n';
#endif

    s[0][0] = -1;
    s[0][1] = -1;
    s[0][2] = 0;
    s[1][0] = 2;
    s[1][1] = -1;
    s[1][2] = 0;
    s[2][0] = -1;
    s[2][1] = 2;
    s[2][2] = 0;
    t[0][0] = 0;
    t[0][1] = -1;
    t[0][2] = -1;
    t[1][0] = 0;
    t[1][1] = 2;
    t[1][2] = -1;
    t[2][0] = 0;
    t[2][1] = -1;
    t[2][2] = 2;
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &r, &scale);
    assert(areEqual(sourceCentroid,
                    Point{{0, 0, 0}}));
    assert(areEqual(targetCentroid,
                    Point{{0, 0, 0}}));
    rotate(source, r, &transformed);
#if 1
    std::cout << "\nsource =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "r =\n"
              << r(0, 0) << ' ' << r(0, 1) << ' ' << r(0, 2) << '\n'
              << r(1, 0) << ' ' << r(1, 1) << ' ' << r(1, 2) << '\n'
              << r(2, 0) << ' ' << r(2, 1) << ' ' << r(2, 2) << '\n'
              << "scale = " << scale << '\n';
#endif
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      assert(areEqual(transformed[i], target[i]));
    }

    //
    // Scaled.
    //
    s[0][0] = -1;
    s[0][1] = 0;
    s[0][2] = 0;
    s[1][0] = 0;
    s[1][1] = 0;
    s[1][2] = 0;
    s[2][0] = 1;
    s[2][1] = 0;
    s[2][2] = 0;
    t[0][0] = 0;
    t[0][1] = -2;
    t[0][2] = 0;
    t[1][0] = 0;
    t[1][1] = 0;
    t[1][2] = 0;
    t[2][0] = 0;
    t[2][1] = 2;
    t[2][2] = 0;
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &r, &scale);
    assert(areEqual(sourceCentroid, Point{{0, 0, 0}}));
    assert(areEqual(targetCentroid, Point{{0, 0, 0}}));
    assert(areEqual(scale, Number(2)));
    rotate(source, r, &transformed);
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      //assert(areEqual(transformed[i], target[i]));
    }
#if 0
    std::cout << "Scaled:\n"
              << "source =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "r =\n"
              << r(0, 0) << ' ' << r(0, 1) << ' ' << r(0, 2) << '\n'
              << r(1, 0) << ' ' << r(1, 1) << ' ' << r(1, 2) << '\n'
              << r(2, 0) << ' ' << r(2, 1) << ' ' << r(2, 2) << '\n'
              << "scale = " << scale << '\n';
#endif

    //
    // Bent.
    //
    s[0][0] = -1;
    s[0][1] = 0;
    s[0][2] = 0;
    s[1][0] = 0;
    s[1][1] = 0;
    s[1][2] = 0;
    s[2][0] = 1;
    s[2][1] = 0;
    s[2][2] = 0;
    t[0][0] = -1;
    t[0][1] = 0.1;
    t[0][2] = 0;
    t[1][0] = 0;
    t[1][1] = 0;
    t[1][2] = 0;
    t[2][0] = 1;
    t[2][1] = 0.1;
    t[2][2] = 0;
    source = s;
    target = t;
    geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                     &r, &scale);
    assert(areEqual(sourceCentroid, Point{{0, 0, 0}}));
    //assert(areEqual(targetCentroid, Point{{0, 0, 0}}));
    rotate(source, r, &transformed);
    for (std::size_t i = 0; i != transformed.size(); ++i) {
      assert(areEqual(source[i], s[i] - sourceCentroid));
      assert(areEqual(target[i], t[i] - targetCentroid));
      //assert(areEqual(transformed[i], target[i]));
    }
#if 0
    std::cout << "\nBent:\n"
              << "source =\n" << source
              << "target =\n" << target
              << "transformed =\n" << transformed
              << "r =\n"
              << r(0, 0) << ' ' << r(0, 1) << ' ' << r(0, 2) << '\n'
              << r(1, 0) << ' ' << r(1, 1) << ' ' << r(1, 2) << '\n'
              << r(2, 0) << ' ' << r(2, 1) << ' ' << r(2, 2) << '\n'
              << "scale = " << scale << '\n';
#endif
  }

  return 0;
}

void
amosTest()
{
  using numerical::areEqual;

  typedef float Number;
  const std::size_t Dimension = 3;
  typedef std::array<Number, Dimension> Point;

  std::vector<Point> s(395), source(395), t(395), target(395), transformed(395);
  std::array<Number, Dimension> sourceCentroid, targetCentroid;
  container::EquilateralArray<Number, 2, Dimension> r;
  Number scale;

  //
  // Centered.
  //
  s[  0][0] =   -0.4655625561;
  s[  0][1] =   -0.6932919255;
  s[  0][2] =   11.1849784127;
  s[  1][0] =   -0.2528197823;
  s[  1][1] =    0.4969904405;
  s[  1][2] =   12.0279514573;
  s[  2][0] =    1.1061939520;
  s[  2][1] =    0.5196922765;
  s[  2][2] =   12.7531821046;
  s[  3][0] =    2.5194481043;
  s[  3][1] =    1.5129958489;
  s[  3][2] =    9.2310406712;
  s[  4][0] =    2.1724005568;
  s[  4][1] =    0.8227326779;
  s[  4][2] =   11.7002332997;
  s[  5][0] =   -0.3542555663;
  s[  5][1] =    1.6619762983;
  s[  5][2] =   11.2352324958;
  s[  6][0] =   -0.1285114568;
  s[  6][1] =   -0.6761190498;
  s[  6][2] =    9.9762887746;
  s[  7][0] =    1.5337485914;
  s[  7][1] =    2.1067054615;
  s[  7][2] =   10.6142679475;
  s[  8][0] =    6.8067924823;
  s[  8][1] =  -22.1433908475;
  s[  8][2] =   -3.3719241461;
  s[  9][0] =    8.1918339826;
  s[  9][1] =  -21.7848412551;
  s[  9][2] =   -3.1176664894;
  s[ 10][0] =    9.2702164008;
  s[ 10][1] =  -22.6403161567;
  s[ 10][2] =   -3.8375466291;
  s[ 11][0] =    8.5430512814;
  s[ 11][1] =  -22.3318258839;
  s[ 11][2] =   -1.7681792545;
  s[ 12][0] =    8.5455805044;
  s[ 12][1] =  -21.4088329773;
  s[ 12][2] =   -0.5531854768;
  s[ 13][0] =    9.8886253924;
  s[ 13][1] =  -22.6485097647;
  s[ 13][2] =   -2.4498965929;
  s[ 14][0] =    8.4769729859;
  s[ 14][1] =  -21.9759820034;
  s[ 14][2] =   -1.7320036987;
  s[ 15][0] =    7.1523541920;
  s[ 15][1] =  -21.0584909058;
  s[ 15][2] =   -0.3055927477;
  s[ 16][0] =    6.1745225768;
  s[ 16][1] =  -22.9274684521;
  s[ 16][2] =   -2.6321419453;
  s[ 17][0] =    3.8445256288;
  s[ 17][1] =  -20.9627833370;
  s[ 17][2] =   -4.1849712799;
  s[ 18][0] =    4.9226857483;
  s[ 18][1] =  -21.6988378264;
  s[ 18][2] =   -4.8631009964;
  s[ 19][0] =    5.1116569423;
  s[ 19][1] =  -21.0224957516;
  s[ 19][2] =   -6.2020008248;
  s[ 20][0] =    5.8362219554;
  s[ 20][1] =  -21.9774939518;
  s[ 20][2] =   -7.1533562987;
  s[ 21][0] =    6.2380884755;
  s[ 21][1] =  -21.5140754033;
  s[ 21][2] =   -4.3998085284;
  s[ 22][0] =    2.6621609771;
  s[ 22][1] =  -21.3461985283;
  s[ 22][2] =   -4.3368663015;
  s[ 23][0] =    5.9259913412;
  s[ 23][1] =  -19.8773148637;
  s[ 23][2] =   -5.9969979611;
  s[ 24][0] =    3.1245845303;
  s[ 24][1] =  -19.2429446555;
  s[ 24][2] =   -1.4317963495;
  s[ 25][0] =    3.0073385093;
  s[ 25][1] =  -19.0976157178;
  s[ 25][2] =   -2.9181220407;
  s[ 26][0] =    3.1895137245;
  s[ 26][1] =  -17.6211466023;
  s[ 26][2] =   -3.3830347153;
  s[ 27][0] =    4.7335552214;
  s[ 27][1] =  -18.5796097310;
  s[ 27][2] =   -1.8087673006;
  s[ 28][0] =    2.9800564268;
  s[ 28][1] =  -17.0532442465;
  s[ 28][2] =   -0.9940959989;
  s[ 29][0] =    3.9845119419;
  s[ 29][1] =  -17.2846790514;
  s[ 29][2] =   -2.1306008477;
  s[ 30][0] =    4.0807125318;
  s[ 30][1] =  -19.8542845806;
  s[ 30][2] =   -3.4614235439;
  s[ 31][0] =    4.2087555099;
  s[ 31][1] =  -18.9733762420;
  s[ 31][2] =   -0.8592596897;
  s[ 32][0] =    1.1483534150;
  s[ 32][1] =  -19.0500764314;
  s[ 32][2] =    1.4178010027;
  s[ 33][0] =    2.2711384939;
  s[ 33][1] =  -19.7807676763;
  s[ 33][2] =    0.7828790996;
  s[ 34][0] =    2.2513800243;
  s[ 34][1] =  -21.2367217915;
  s[ 34][2] =    1.3278457759;
  s[ 35][0] =    0.8069439097;
  s[ 35][1] =  -22.1944473748;
  s[ 35][2] =    3.1999682739;
  s[ 36][0] =   -0.6534426517;
  s[ 36][1] =  -22.4500101069;
  s[ 36][2] =    3.5969546786;
  s[ 37][0] =    0.8278524505;
  s[ 37][1] =  -21.4499695133;
  s[ 37][2] =    1.8604025353;
  s[ 38][0] =    2.1324850663;
  s[ 38][1] =  -19.7206266289;
  s[ 38][2] =   -0.6432977472;
  s[ 39][0] =   -0.7258250731;
  s[ 39][1] =  -23.3127454340;
  s[ 39][2] =    4.7779842686;
  s[ 40][0] =   -0.0092942232;
  s[ 40][1] =  -19.4357442562;
  s[ 40][2] =    1.1489475004;
  s[ 41][0] =    0.2302818614;
  s[ 41][1] =  -16.1840722983;
  s[ 41][2] =    3.6114164837;
  s[ 42][0] =    0.0891444525;
  s[ 42][1] =  -17.4590925300;
  s[ 42][2] =    2.8643393013;
  s[ 43][0] =    1.2752836307;
  s[ 43][1] =  -18.0142013873;
  s[ 43][2] =    2.2788472776;
  s[ 44][0] =    1.3388239126;
  s[ 44][1] =  -15.6128499903;
  s[ 44][2] =    3.6846237860;
  s[ 45][0] =   -1.2535179621;
  s[ 45][1] =  -13.2630401679;
  s[ 45][2] =    4.5848912071;
  s[ 46][0] =   -0.7492639938;
  s[ 46][1] =  -14.5730536553;
  s[ 46][2] =    5.1971804702;
  s[ 47][0] =   -1.3530061989;
  s[ 47][1] =  -15.2349137661;
  s[ 47][2] =    6.5142603695;
  s[ 48][0] =   -1.9886886505;
  s[ 48][1] =  -17.4435514272;
  s[ 48][2] =    5.7112504387;
  s[ 49][0] =   -1.0230079402;
  s[ 49][1] =  -16.7270436726;
  s[ 49][2] =    6.5379124438;
  s[ 50][0] =   -0.8389575096;
  s[ 50][1] =  -15.6440777231;
  s[ 50][2] =    4.2468824522;
  s[ 51][0] =   -1.9929075931;
  s[ 51][1] =  -13.3524847802;
  s[ 51][2] =    3.5813996637;
  s[ 52][0] =   -1.1558270757;
  s[ 52][1] =   -9.4923761320;
  s[ 52][2] =    5.0565184632;
  s[ 53][0] =   -0.8298570300;
  s[ 53][1] =  -10.7022258700;
  s[ 53][2] =    4.2024852939;
  s[ 54][0] =   -1.3647548111;
  s[ 54][1] =  -10.5546010856;
  s[ 54][2] =    2.7253003573;
  s[ 55][0] =   -1.5586503622;
  s[ 55][1] =  -11.9510456940;
  s[ 55][2] =    2.1373286007;
  s[ 56][0] =   -0.8136451256;
  s[ 56][1] =  -12.0023432814;
  s[ 56][2] =    4.9286537979;
  s[ 57][0] =   -0.3835657523;
  s[ 57][1] =   -9.2219361427;
  s[ 57][2] =    6.0011827057;
  s[ 58][0] =   -2.6093619873;
  s[ 58][1] =   -9.8783895685;
  s[ 58][2] =    2.7406286463;
  s[ 59][0] =   -1.8282424629;
  s[ 59][1] =   -6.0777934843;
  s[ 59][2] =    5.0484943779;
  s[ 60][0] =   -2.4967807436;
  s[ 60][1] =   -7.3474952285;
  s[ 60][2] =    5.4910757481;
  s[ 61][0] =   -2.9615265433;
  s[ 61][1] =   -7.3417552577;
  s[ 61][2] =    6.9829875067;
  s[ 62][0] =   -3.3919257727;
  s[ 62][1] =   -8.7545122633;
  s[ 62][2] =    7.3735516108;
  s[ 63][0] =   -2.1699669151;
  s[ 63][1] =   -8.6126337393;
  s[ 63][2] =    4.8207527463;
  s[ 64][0] =   -2.1999386403;
  s[ 64][1] =   -5.6576666817;
  s[ 64][2] =    3.9310716314;
  s[ 65][0] =   -4.0823608634;
  s[ 65][1] =   -6.4738580590;
  s[ 65][2] =    7.1175121044;
  s[ 66][0] =   -1.0231613503;
  s[ 66][1] =   -3.0143928567;
  s[ 66][2] =    6.6405584866;
  s[ 67][0] =   -0.9718956604;
  s[ 67][1] =   -3.8327718148;
  s[ 67][2] =    5.3564688307;
  s[ 68][0] =    0.0929348046;
  s[ 68][1] =   -3.5475077212;
  s[ 68][2] =    4.2057087438;
  s[ 69][0] =    1.3690339318;
  s[ 69][1] =   -4.3511254784;
  s[ 69][2] =    4.4595125715;
  s[ 70][0] =   -1.0049835827;
  s[ 70][1] =   -5.2496309922;
  s[ 70][2] =    5.7518944970;
  s[ 71][0] =   -0.7636899235;
  s[ 71][1] =   -3.7178546079;
  s[ 71][2] =    7.6542991351;
  s[ 72][0] =    0.4142546915;
  s[ 72][1] =   -2.1622490792;
  s[ 72][2] =    4.1323375305;
  s[ 73][0] =   -0.6310932041;
  s[ 73][1] =    0.4673861731;
  s[ 73][2] =    6.3292897416;
  s[ 74][0] =   -1.8111434736;
  s[ 74][1] =   -0.5054281156;
  s[ 74][2] =    6.2829645130;
  s[ 75][0] =   -3.2650175854;
  s[ 75][1] =    0.1383370944;
  s[ 75][2] =    6.6262821184;
  s[ 76][0] =   -3.8678716763;
  s[ 76][1] =   -2.2325263567;
  s[ 76][2] =    6.7994523331;
  s[ 77][0] =   -4.1795950494;
  s[ 77][1] =   -0.8949689563;
  s[ 77][2] =    7.2912110428;
  s[ 78][0] =   -1.3940592457;
  s[ 78][1] =   -1.7191159944;
  s[ 78][2] =    6.9965714603;
  s[ 79][0] =   -0.2805288077;
  s[ 79][1] =    0.9358423013;
  s[ 79][2] =    5.2251875485;
  s[ 80][0] =   -0.1760755056;
  s[ 80][1] =   -3.3712631948;
  s[ 80][2] =   10.2335651349;
  s[ 81][0] =   -1.3246013893;
  s[ 81][1] =   -3.0252853529;
  s[ 81][2] =   11.1407317447;
  s[ 82][0] =   -2.8885366085;
  s[ 82][1] =   -3.2566806646;
  s[ 82][2] =   11.0938823400;
  s[ 83][0] =   -3.5779235583;
  s[ 83][1] =   -3.0995010684;
  s[ 83][2] =    9.7421134767;
  s[ 84][0] =   -1.0668115770;
  s[ 84][1] =   -1.7647632931;
  s[ 84][2] =   11.7502000806;
  s[ 85][0] =    0.7982653592;
  s[ 85][1] =   -2.5955570604;
  s[ 85][2] =   10.3349495652;
  s[ 86][0] =   -3.4418517864;
  s[ 86][1] =   -2.2793898367;
  s[ 86][2] =   11.9701475673;
  s[ 87][0] =    1.2823095287;
  s[ 87][1] =    3.0747935772;
  s[ 87][2] =    7.1543670618;
  s[ 88][0] =    1.4003142195;
  s[ 88][1] =    1.6143596426;
  s[ 88][2] =    7.4390183180;
  s[ 89][0] =    2.4690180706;
  s[ 89][1] =    1.0815719440;
  s[ 89][2] =    6.4508772028;
  s[ 90][0] =    0.1510702790;
  s[ 90][1] =    0.8116541279;
  s[ 90][2] =    7.4123122557;
  s[ 91][0] =    0.4087297625;
  s[ 91][1] =    3.3270929833;
  s[ 91][2] =    6.2972351598;
  s[ 92][0] =    3.3940188429;
  s[ 92][1] =    3.7493191780;
  s[ 92][2] =    9.6803587387;
  s[ 93][0] =    2.8759098819;
  s[ 93][1] =    4.6787928724;
  s[ 93][2] =    8.6015745799;
  s[ 94][0] =    3.3915158262;
  s[ 94][1] =    6.1376900083;
  s[ 94][2] =    8.2229692552;
  s[ 95][0] =    2.1455839402;
  s[ 95][1] =    6.8290033181;
  s[ 95][2] =    7.6597954579;
  s[ 96][0] =    3.9581105964;
  s[ 96][1] =    6.9298841907;
  s[ 96][2] =    9.4052608167;
  s[ 97][0] =    1.9427472981;
  s[ 97][1] =    4.2165053281;
  s[ 97][2] =    7.5653035243;
  s[ 98][0] =    2.3903920756;
  s[ 98][1] =    3.3877650980;
  s[ 98][2] =   10.3320836613;
  s[ 99][0] =    6.2274386552;
  s[ 99][1] =    1.5776565479;
  s[ 99][2] =    9.6122335943;
  s[100][0] =    5.9126287309;
  s[100][1] =    2.8434955069;
  s[100][2] =   10.2527705395;
  s[101][0] =    7.1423159673;
  s[101][1] =    3.7612783056;
  s[101][2] =   10.0512695515;
  s[102][0] =    7.0831764978;
  s[102][1] =    4.8668575854;
  s[102][2] =   11.0047310039;
  s[103][0] =    4.4798355243;
  s[103][1] =    3.1399786880;
  s[103][2] =   10.2806410180;
  s[104][0] =    5.7236724012;
  s[104][1] =    1.3229764782;
  s[104][2] =    8.4984815728;
  s[105][0] =    8.1932133138;
  s[105][1] =    0.2067160132;
  s[105][2] =    8.3217240908;
  s[106][0] =    7.7209950744;
  s[106][1] =   -0.3321466907;
  s[106][2] =    9.6252032227;
  s[107][0] =    8.7354246282;
  s[107][1] =   -1.0247590154;
  s[107][2] =   10.6093666378;
  s[108][0] =    7.1099405928;
  s[108][1] =    0.7781041076;
  s[108][2] =   10.2505291104;
  s[109][0] =    7.5987389703;
  s[109][1] =   -0.1654202044;
  s[109][2] =    7.2879780211;
  s[110][0] =    9.1004723870;
  s[110][1] =    1.3657027666;
  s[110][2] =    5.6941910721;
  s[111][0] =    9.3275827787;
  s[111][1] =    1.9778724346;
  s[111][2] =    7.0010269010;
  s[112][0] =   10.8135619267;
  s[112][1] =    2.3442209298;
  s[112][2] =    6.8552722635;
  s[113][0] =    9.0861907447;
  s[113][1] =    1.2158741971;
  s[113][2] =    8.2047151541;
  s[114][0] =   10.0838237544;
  s[114][1] =    0.8129213122;
  s[114][2] =    5.1564024253;
  s[115][0] =    6.1837590065;
  s[115][1] =    0.5365709826;
  s[115][2] =    5.5901058170;
  s[116][0] =    6.7555842690;
  s[116][1] =    1.8863450313;
  s[116][2] =    5.5386657938;
  s[117][0] =    5.9839722553;
  s[117][1] =    2.9754046945;
  s[117][2] =    4.7587885604;
  s[118][0] =    4.9134898647;
  s[118][1] =    3.6248301405;
  s[118][2] =    5.6399403425;
  s[119][0] =    7.9835251346;
  s[119][1] =    1.5202705151;
  s[119][2] =    4.9621527286;
  s[120][0] =    5.6855082637;
  s[120][1] =   -0.0624453341;
  s[120][2] =    4.5934180587;
  s[121][0] =    6.8955781633;
  s[121][1] =    3.9702634679;
  s[121][2] =    4.3189568090;
  s[122][0] =    7.3034335339;
  s[122][1] =   -1.9027534213;
  s[122][2] =    5.7511413514;
  s[123][0] =    6.4815049806;
  s[123][1] =   -1.4961970662;
  s[123][2] =    6.8623722136;
  s[124][0] =    5.0606637415;
  s[124][1] =   -2.0213852427;
  s[124][2] =    6.7678535959;
  s[125][0] =    6.5726843837;
  s[125][1] =   -0.1070884135;
  s[125][2] =    6.7082960275;
  s[126][0] =    6.8262688906;
  s[126][1] =   -2.3748019926;
  s[126][2] =    4.6974236937;
  s[127][0] =    8.7028679567;
  s[127][1] =   -1.6022345153;
  s[127][2] =    3.5102011064;
  s[128][0] =    9.4389225819;
  s[128][1] =   -1.6545293979;
  s[128][2] =    4.7642731527;
  s[129][0] =   10.0812699785;
  s[129][1] =   -2.9936773752;
  s[129][2] =    4.8909788240;
  s[130][0] =   10.4687781491;
  s[130][1] =   -4.0784044691;
  s[130][2] =    2.7224537916;
  s[131][0] =    9.4868558212;
  s[131][1] =   -3.8664171782;
  s[131][2] =    3.7833184335;
  s[132][0] =    8.5996758153;
  s[132][1] =   -1.6145440951;
  s[132][2] =    5.8707005165;
  s[133][0] =    8.2216749703;
  s[133][1] =   -2.6259311096;
  s[133][2] =    2.9653903319;
  s[134][0] =    7.0651301853;
  s[134][1] =   -1.3019978853;
  s[134][2] =    1.3475226307;
  s[135][0] =    7.8661143537;
  s[135][1] =   -0.1808181865;
  s[135][2] =    1.7808538020;
  s[136][0] =    8.8043669690;
  s[136][1] =    0.1088449886;
  s[136][2] =    0.6307889847;
  s[137][0] =    9.2554625197;
  s[137][1] =   -2.2927539942;
  s[137][2] =   -0.0899887676;
  s[138][0] =   10.3959235756;
  s[138][1] =   -3.2935425500;
  s[138][2] =   -0.3303520571;
  s[139][0] =    9.8696014738;
  s[139][1] =   -0.9751534235;
  s[139][2] =    0.4088854179;
  s[140][0] =    8.6078533232;
  s[140][1] =   -0.3955527207;
  s[140][2] =    2.9478083148;
  s[141][0] =   10.0705624402;
  s[141][1] =   -4.1852843057;
  s[141][2] =   -1.4453732502;
  s[142][0] =    7.3124853993;
  s[142][1] =   -1.9148880595;
  s[142][2] =    0.2783864784;
  s[143][0] =    5.3753271403;
  s[143][1] =   -3.8756123221;
  s[143][2] =    1.4252638631;
  s[144][0] =    5.1710642082;
  s[144][1] =   -2.4450366812;
  s[144][2] =    1.1640711830;
  s[145][0] =    3.7634326167;
  s[145][1] =   -1.8828877264;
  s[145][2] =    1.0665627968;
  s[146][0] =    3.7968428781;
  s[146][1] =   -0.7548347993;
  s[146][2] =    0.0342931741;
  s[147][0] =    3.2886181648;
  s[147][1] =   -1.3408321524;
  s[147][2] =    2.4148571317;
  s[148][0] =    5.9511845284;
  s[148][1] =   -1.6203273690;
  s[148][2] =    1.9994388321;
  s[149][0] =    5.1874475694;
  s[149][1] =   -4.7316688593;
  s[149][2] =    0.5131719478;
  s[150][0] =    1.3137575413;
  s[150][1] =   -6.5297253466;
  s[150][2] =    8.7534995871;
  s[151][0] =    1.2725918350;
  s[151][1] =   -5.0412591243;
  s[151][2] =    8.9739399600;
  s[152][0] =    2.6557451348;
  s[152][1] =   -4.3073140430;
  s[152][2] =    9.0889736066;
  s[153][0] =    2.4641427756;
  s[153][1] =   -3.7331916576;
  s[153][2] =    7.7624154223;
  s[154][0] =    1.5010323360;
  s[154][1] =   -2.8230343621;
  s[154][2] =    5.3197356100;
  s[155][0] =    0.0135709020;
  s[155][1] =   -4.4974391357;
  s[155][2] =    9.4923614469;
  s[156][0] =    0.5941078280;
  s[156][1] =   -6.8479862617;
  s[156][2] =    7.7685755575;
  s[157][0] =    0.9599823841;
  s[157][1] =   -2.3908122107;
  s[157][2] =    4.1603561155;
  s[158][0] =    7.3975732867;
  s[158][1] =   -5.6461689212;
  s[158][2] =    1.5544282567;
  s[159][0] =    6.6082352057;
  s[159][1] =   -5.4124793718;
  s[159][2] =    2.7944283151;
  s[160][0] =    5.6692750778;
  s[160][1] =   -6.5877491300;
  s[160][2] =    2.6419534736;
  s[161][0] =    6.4473480538;
  s[161][1] =   -6.5813205856;
  s[161][2] =    1.4129622846;
  s[162][0] =    8.5479644751;
  s[162][1] =   -5.8749493917;
  s[162][2] =   -0.2714926757;
  s[163][0] =    5.9651982532;
  s[163][1] =   -4.1737661753;
  s[163][2] =    2.5972261574;
  s[164][0] =    7.1829006347;
  s[164][1] =   -6.7023661494;
  s[164][2] =    0.9122485383;
  s[165][0] =    7.7043226425;
  s[165][1] =   -5.0706251348;
  s[165][2] =   -1.3523769955;
  s[166][0] =    8.6790348933;
  s[166][1] =   -4.6785779069;
  s[166][2] =   -0.3342486941;
  s[167][0] =   10.0573123338;
  s[167][1] =   -5.2700139580;
  s[167][2] =   -0.7068568581;
  s[168][0] =   10.6919105842;
  s[168][1] =   -3.6916646991;
  s[168][2] =    1.0655812289;
  s[169][0] =    9.3965920004;
  s[169][1] =   -3.7488641177;
  s[169][2] =    1.8779513634;
  s[170][0] =   11.0215591284;
  s[170][1] =   -5.0535585670;
  s[170][2] =    0.4543712439;
  s[171][0] =    8.2099661958;
  s[171][1] =   -4.7057591961;
  s[171][2] =    1.0310602497;
  s[172][0] =    9.5881510826;
  s[172][1] =   -3.1254372069;
  s[172][2] =    3.1870108084;
  s[173][0] =    7.7536519683;
  s[173][1] =   -6.1520104350;
  s[173][2] =   -1.9880908445;
  s[174][0] =    5.7472971671;
  s[174][1] =   -5.7456905697;
  s[174][2] =   -3.0447742411;
  s[175][0] =    5.9509689547;
  s[175][1] =   -4.3400712649;
  s[175][2] =   -2.7928147922;
  s[176][0] =    6.3436870708;
  s[176][1] =   -3.5595464511;
  s[176][2] =   -4.0339413830;
  s[177][0] =    5.8609165922;
  s[177][1] =   -2.4192980905;
  s[177][2] =   -2.0751978287;
  s[178][0] =    5.8225276601;
  s[178][1] =   -2.2196424479;
  s[178][2] =   -3.5238945121;
  s[179][0] =    6.8443734711;
  s[179][1] =   -4.1145015624;
  s[179][2] =   -1.7138267721;
  s[180][0] =    5.1939981022;
  s[180][1] =   -1.6397665441;
  s[180][2] =   -1.2171913142;
  s[181][0] =    5.9698162329;
  s[181][1] =   -6.3095611951;
  s[181][2] =   -4.1441852179;
  s[182][0] =    6.5045270636;
  s[182][1] =   -3.3950245902;
  s[182][2] =   -1.6329028186;
  s[183][0] =    5.3778806213;
  s[183][1] =   -8.2734881521;
  s[183][2] =   -0.9967361950;
  s[184][0] =    4.7053985893;
  s[184][1] =   -7.6785636597;
  s[184][2] =   -2.1414911697;
  s[185][0] =    3.1936245521;
  s[185][1] =   -7.4692790323;
  s[185][2] =   -1.9137276665;
  s[186][0] =    3.3978230958;
  s[186][1] =   -6.3806624480;
  s[186][2] =   -0.9686221117;
  s[187][0] =    4.1393723028;
  s[187][1] =   -4.3591394814;
  s[187][2] =    0.7766135660;
  s[188][0] =    5.1936161997;
  s[188][1] =   -6.3648363466;
  s[188][2] =   -2.0046148415;
  s[189][0] =    4.7061212098;
  s[189][1] =   -8.3613980654;
  s[189][2] =    0.0531110466;
  s[190][0] =    4.5178977130;
  s[190][1] =   -3.3781307605;
  s[190][2] =    1.6225784773;
  s[191][0] =    7.5741814950;
  s[191][1] =   -8.9606230218;
  s[191][2] =   -3.3002232871;
  s[192][0] =    7.6607683857;
  s[192][1] =   -8.9889290134;
  s[192][2] =   -1.8495368974;
  s[193][0] =    7.5170913685;
  s[193][1] =  -10.4898619287;
  s[193][2] =   -1.7829368558;
  s[194][0] =    6.7015492564;
  s[194][1] =   -8.5180264344;
  s[194][2] =   -0.8935666861;
  s[195][0] =    7.1373588327;
  s[195][1] =  -10.0259316463;
  s[195][2] =   -3.8243782771;
  s[196][0] =    6.8972650891;
  s[196][1] =   -8.8428282468;
  s[196][2] =   -5.5585481629;
  s[197][0] =    8.0145999276;
  s[197][1] =   -7.9192121503;
  s[197][2] =   -5.3891799188;
  s[198][0] =    9.2390226846;
  s[198][1] =   -8.2133336190;
  s[198][2] =   -6.3278944808;
  s[199][0] =    9.8744221297;
  s[199][1] =   -7.0267668345;
  s[199][2] =   -6.8966193688;
  s[200][0] =    8.1557045042;
  s[200][1] =   -7.9530597917;
  s[200][2] =   -3.9849789802;
  s[201][0] =   10.4388611436;
  s[201][1] =   -6.0954449209;
  s[201][2] =   -6.1156858304;
  s[202][0] =    6.9030566163;
  s[202][1] =  -10.0306987101;
  s[202][2] =   -5.1343806849;
  s[203][0] =    9.9298503748;
  s[203][1] =   -6.9024239475;
  s[203][2] =   -8.1407639039;
  s[204][0] =    4.7084575484;
  s[204][1] =  -10.1414421535;
  s[204][2] =   -7.0401598980;
  s[205][0] =    5.1761377819;
  s[205][1] =   -8.7618341744;
  s[205][2] =   -7.2272049108;
  s[206][0] =    4.0899237146;
  s[206][1] =   -7.8058196969;
  s[206][2] =   -7.7584300511;
  s[207][0] =    3.5143185606;
  s[207][1] =   -8.4475186806;
  s[207][2] =   -8.9352831273;
  s[208][0] =    5.8235446246;
  s[208][1] =   -8.2339920973;
  s[208][2] =   -6.0849422668;
  s[209][0] =    4.1040358200;
  s[209][1] =  -10.7293654107;
  s[209][2] =   -7.9619873894;
  s[210][0] =    5.1397466212;
  s[210][1] =  -11.9683452848;
  s[210][2] =   -4.2248652525;
  s[211][0] =    4.7820418395;
  s[211][1] =  -12.1567224442;
  s[211][2] =   -5.6106431338;
  s[212][0] =    3.3901113059;
  s[212][1] =  -12.8134172663;
  s[212][2] =   -5.8382296609;
  s[213][0] =    3.6143349568;
  s[213][1] =  -13.8634901592;
  s[213][2] =   -6.8248673817;
  s[214][0] =    5.0274983722;
  s[214][1] =  -10.7878149378;
  s[214][2] =   -5.9101161726;
  s[215][0] =    4.8407607750;
  s[215][1] =  -13.9874008058;
  s[215][2] =   -7.3552256698;
  s[216][0] =    4.3784973848;
  s[216][1] =  -12.2505342895;
  s[216][2] =   -3.2743014134;
  s[217][0] =    2.6949326506;
  s[217][1] =  -14.6407600281;
  s[217][2] =   -7.1606149497;
  s[218][0] =    8.2284748092;
  s[218][1] =  -12.2332439600;
  s[218][2] =   -4.7946539081;
  s[219][0] =    7.1991853545;
  s[219][1] =  -11.2723121284;
  s[219][2] =   -5.1822426930;
  s[220][0] =    6.2957585550;
  s[220][1] =  -11.2975963137;
  s[220][2] =   -4.1074786273;
  s[221][0] =    8.9541134312;
  s[221][1] =  -12.8081454688;
  s[221][2] =   -5.6330636598;
  s[222][0] =   10.1325232363;
  s[222][1] =  -12.7412287452;
  s[222][2] =   -3.6630372800;
  s[223][0] =    9.1089160607;
  s[223][1] =  -13.5647285432;
  s[223][2] =   -2.9351003571;
  s[224][0] =    7.7785102745;
  s[224][1] =  -14.4395965362;
  s[224][2] =   -3.0662275897;
  s[225][0] =    8.2009293971;
  s[225][1] =  -15.8896529233;
  s[225][2] =   -2.8437004814;
  s[226][0] =    7.1660605026;
  s[226][1] =  -14.3319656816;
  s[226][2] =   -4.4666461358;
  s[227][0] =    8.3092756530;
  s[227][1] =  -12.4738096403;
  s[227][2] =   -3.4814772916;
  s[228][0] =    9.5687087341;
  s[228][1] =  -11.7697180584;
  s[228][2] =   -4.2126530918;
  s[229][0] =    2.7887457045;
  s[229][1] =   -9.6012602136;
  s[229][2] =    8.1979838907;
  s[230][0] =    2.2214259306;
  s[230][1] =   -8.8938591386;
  s[230][2] =    9.4310470842;
  s[231][0] =    0.9394244508;
  s[231][1] =   -9.6231368410;
  s[231][2] =    9.9368197103;
  s[232][0] =    1.6228699009;
  s[232][1] =   -8.8189614119;
  s[232][2] =    7.9304818542;
  s[233][0] =    2.0739619500;
  s[233][1] =   -9.2810735530;
  s[233][2] =    6.5429016119;
  s[234][0] =    0.3998703063;
  s[234][1] =   -9.5389934511;
  s[234][2] =    8.5354555620;
  s[235][0] =    2.0818387804;
  s[235][1] =   -7.4262730229;
  s[235][2] =    9.4760501611;
  s[236][0] =    3.3488038828;
  s[236][1] =   -8.5832975198;
  s[236][2] =    6.2922284908;
  s[237][0] =    3.9267454380;
  s[237][1] =  -10.1091695162;
  s[237][2] =    8.3133119728;
  s[238][0] =   13.7625284134;
  s[238][1] =  -13.7419294778;
  s[238][2] =   -2.6703827482;
  s[239][0] =   12.8140943541;
  s[239][1] =  -13.3271143217;
  s[239][2] =   -3.7746245005;
  s[240][0] =   13.3756681523;
  s[240][1] =  -13.9532995812;
  s[240][2] =   -5.1155347915;
  s[241][0] =   13.8748340308;
  s[241][1] =  -15.2926516067;
  s[241][2] =   -4.8252898437;
  s[242][0] =   11.4987462596;
  s[242][1] =  -12.6693611839;
  s[242][2] =   -3.8819998272;
  s[243][0] =   14.8285727260;
  s[243][1] =  -14.0853501033;
  s[243][2] =   -3.2233783322;
  s[244][0] =   14.1541332831;
  s[244][1] =  -13.4723452773;
  s[244][2] =    1.0968347652;
  s[245][0] =   13.2432210569;
  s[245][1] =  -13.6556329284;
  s[245][2] =   -0.0987797965;
  s[246][0] =   13.9856369955;
  s[246][1] =  -13.8112398854;
  s[246][2] =   -1.3153105939;
  s[247][0] =   14.8328737900;
  s[247][1] =  -12.4227925056;
  s[247][2] =    1.1356241247;
  s[248][0] =   13.8068450724;
  s[248][1] =  -13.7502087152;
  s[248][2] =    4.3474278540;
  s[249][0] =   14.9639760307;
  s[249][1] =  -14.0096829866;
  s[249][2] =    3.4189144255;
  s[250][0] =   16.0168579743;
  s[250][1] =  -15.1450192653;
  s[250][2] =    3.8870543202;
  s[251][0] =   17.7113760377;
  s[251][1] =  -13.4385316005;
  s[251][2] =    3.5879518324;
  s[252][0] =   17.3464117116;
  s[252][1] =  -14.8032265060;
  s[252][2] =    3.2135787498;
  s[253][0] =   14.2685142121;
  s[253][1] =  -14.3244854442;
  s[253][2] =    2.1665563171;
  s[254][0] =   13.0695718632;
  s[254][1] =  -14.7534388637;
  s[254][2] =    4.4627168214;
  s[255][0] =   13.4059980200;
  s[255][1] =  -11.1533676140;
  s[255][2] =    6.8615741352;
  s[256][0] =   13.7880896811;
  s[256][1] =  -11.3432363496;
  s[256][2] =    5.4051718453;
  s[257][0] =   13.3363631128;
  s[257][1] =  -10.0572690070;
  s[257][2] =    4.5859003723;
  s[258][0] =   12.4199146899;
  s[258][1] =   -9.1780189873;
  s[258][2] =    6.7731641807;
  s[259][0] =   13.5016238241;
  s[259][1] =   -7.6790395054;
  s[259][2] =    5.5400161246;
  s[260][0] =   13.0741499418;
  s[260][1] =   -7.0979828269;
  s[260][2] =    6.7163905150;
  s[261][0] =   14.1987346692;
  s[261][1] =   -6.9558259282;
  s[261][2] =    4.5833589147;
  s[262][0] =   13.0975051770;
  s[262][1] =   -9.0063008999;
  s[262][2] =    5.5702145247;
  s[263][0] =   14.0337490536;
  s[263][1] =   -5.0286595872;
  s[263][2] =    6.0519162098;
  s[264][0] =   13.3318752826;
  s[264][1] =   -5.7668477282;
  s[264][2] =    7.0010990645;
  s[265][0] =   14.4585088195;
  s[265][1] =   -5.6132017489;
  s[265][2] =    4.8565159339;
  s[266][0] =   13.2869312061;
  s[266][1] =  -12.6474199323;
  s[266][2] =    4.9976035068;
  s[267][0] =   12.4235361040;
  s[267][1] =   -8.0150528574;
  s[267][2] =    7.4471728214;
  s[268][0] =   14.1525490934;
  s[268][1] =  -10.4363134358;
  s[268][2] =    7.5624592359;
  s[269][0] =   11.1724361381;
  s[269][1] =  -10.2980892927;
  s[269][2] =    9.2336546376;
  s[270][0] =   11.8529733385;
  s[270][1] =  -11.5985152230;
  s[270][2] =    8.8584690792;
  s[271][0] =   12.6905790738;
  s[271][1] =  -12.3512314366;
  s[271][2] =    9.9600586802;
  s[272][0] =   12.8925198048;
  s[272][1] =  -13.8113933578;
  s[272][2] =    9.5476489403;
  s[273][0] =   12.3072145024;
  s[273][1] =  -11.7047587977;
  s[273][2] =    7.4631545206;
  s[274][0] =   10.1885774923;
  s[274][1] =   -9.9965941162;
  s[274][2] =    8.5226338790;
  s[275][0] =   13.9562693794;
  s[275][1] =  -11.7225574670;
  s[275][2] =   10.1104982970;
  s[276][0] =   10.2428138059;
  s[276][1] =   -8.5603686178;
  s[276][2] =   12.1804212099;
  s[277][0] =   10.5064480546;
  s[277][1] =   -8.3979931424;
  s[277][2] =   10.7168675134;
  s[278][0] =   11.0931738313;
  s[278][1] =   -6.9751197680;
  s[278][2] =   10.3973871413;
  s[279][0] =   12.4468580373;
  s[279][1] =   -7.4761157618;
  s[279][2] =   10.2301281409;
  s[280][0] =   14.7728192600;
  s[280][1] =   -8.9802608947;
  s[280][2] =   10.1245612818;
  s[281][0] =   11.4028423603;
  s[281][1] =   -9.4746758610;
  s[281][2] =   10.2985639711;
  s[282][0] =   11.1774607954;
  s[282][1] =   -8.3879327913;
  s[282][2] =   12.9922076383;
  s[283][0] =   15.8576308291;
  s[283][1] =   -9.7836952085;
  s[283][2] =   10.1096901541;
  s[284][0] =    8.4372156992;
  s[284][1] =   -7.7327920866;
  s[284][2] =   14.7113671025;
  s[285][0] =    8.5949532650;
  s[285][1] =   -9.0450252741;
  s[285][2] =   14.0213754591;
  s[286][0] =    9.1033575209;
  s[286][1] =  -10.2443908325;
  s[286][2] =   14.9358580561;
  s[287][0] =   10.3612001612;
  s[287][1] =  -10.7624894793;
  s[287][2] =   14.4081455459;
  s[288][0] =    9.0142956831;
  s[288][1] =   -8.9004167785;
  s[288][2] =   12.6401860054;
  s[289][0] =    7.2464759236;
  s[289][1] =   -7.4742131979;
  s[289][2] =   15.0026196230;
  s[290][0] =    8.2097006046;
  s[290][1] =   -6.6830722530;
  s[290][2] =   16.9511333687;
  s[291][0] =    9.2951345423;
  s[291][1] =   -6.0393125256;
  s[291][2] =   16.2355006932;
  s[292][0] =    9.2698533341;
  s[292][1] =   -4.4808319744;
  s[292][2] =   16.3227854453;
  s[293][0] =   10.5781864029;
  s[293][1] =   -3.9240341762;
  s[293][2] =   15.9825535121;
  s[294][0] =    9.4149750492;
  s[294][1] =   -6.8380333086;
  s[294][2] =   15.0409469517;
  s[295][0] =    7.0139330737;
  s[295][1] =   -6.7083845785;
  s[295][2] =   16.5631528852;
  s[296][0] =    7.5445060814;
  s[296][1] =   -9.4064855838;
  s[296][2] =   17.3874456774;
  s[297][0] =    8.1426015753;
  s[297][1] =   -8.5499282361;
  s[297][2] =   18.4688193756;
  s[298][0] =    7.6093471216;
  s[298][1] =   -8.2292411163;
  s[298][2] =   19.8776540386;
  s[299][0] =    8.7249244773;
  s[299][1] =   -7.3951232891;
  s[299][2] =   17.9513532859;
  s[300][0] =    8.3589446373;
  s[300][1] =  -10.3002254601;
  s[300][2] =   17.0820963744;
  s[301][0] =    4.7635252092;
  s[301][1] =   -8.2543869159;
  s[301][2] =   15.3113255745;
  s[302][0] =    5.1239146950;
  s[302][1] =   -8.9751338290;
  s[302][2] =   16.5702053452;
  s[303][0] =    4.2669508760;
  s[303][1] =  -10.2559947677;
  s[303][2] =   16.4856973604;
  s[304][0] =    4.7401516511;
  s[304][1] =  -11.3110789945;
  s[304][2] =   17.4858346749;
  s[305][0] =    6.4526987682;
  s[305][1] =   -9.5013646971;
  s[305][2] =   16.5668153583;
  s[306][0] =    4.3982369441;
  s[306][1] =   -8.9470243945;
  s[306][2] =   14.3364551955;
  s[307][0] =    2.9001657056;
  s[307][1] =   -9.9529075113;
  s[307][2] =   16.7036482955;
  s[308][0] =    2.6188647449;
  s[308][1] =  -12.1393773520;
  s[308][2] =    6.0121638397;
  s[309][0] =    2.5149853068;
  s[309][1] =  -10.6407547557;
  s[309][2] =    5.8336240693;
  s[310][0] =    3.3444828488;
  s[310][1] =   -9.9891332410;
  s[310][2] =    4.6464613002;
  s[311][0] =    2.0215129126;
  s[311][1] =  -10.8941005572;
  s[311][2] =    2.7129753017;
  s[312][0] =    1.2497238311;
  s[312][1] =   -8.8151907907;
  s[312][2] =    3.8802044607;
  s[313][0] =    2.4711003783;
  s[313][1] =   -9.6248998077;
  s[313][2] =    3.4381213496;
  s[314][0] =    2.1494532944;
  s[314][1] =   -9.8223115536;
  s[314][2] =    7.0077379277;
  s[315][0] =    1.5509945971;
  s[315][1] =  -12.7657386667;
  s[315][2] =    5.8379651067;
  s[316][0] =    4.4348264179;
  s[316][1] =   -7.3912828614;
  s[316][2] =   12.7835653974;
  s[317][0] =    4.3689462606;
  s[317][1] =   -6.3161481819;
  s[317][2] =   13.8641412482;
  s[318][0] =    3.1713825412;
  s[318][1] =   -5.2237147018;
  s[318][2] =   13.8366493986;
  s[319][0] =    3.4138601502;
  s[319][1] =   -5.5024508573;
  s[319][2] =   11.4079615451;
  s[320][0] =    4.7015948288;
  s[320][1] =   -5.9196863362;
  s[320][2] =   10.6688369796;
  s[321][0] =    3.6091216843;
  s[321][1] =   -4.4938372014;
  s[321][2] =   12.5581612685;
  s[322][0] =    4.7390196865;
  s[322][1] =   -6.9121610487;
  s[322][2] =   15.1314692848;
  s[323][0] =    4.7734295147;
  s[323][1] =   -7.3865030655;
  s[323][2] =   10.6025889121;
  s[324][0] =    3.5244903554;
  s[324][1] =   -7.4770683049;
  s[324][2] =   11.9250259972;
  s[325][0] =    6.3644644514;
  s[325][1] =   -9.2720856953;
  s[325][2] =   10.4250254929;
  s[326][0] =    5.8094232031;
  s[326][1] =   -9.4174869468;
  s[326][2] =   11.8558113254;
  s[327][0] =    4.8773881206;
  s[327][1] =  -10.6282072444;
  s[327][2] =   12.2155105591;
  s[328][0] =    5.7331452666;
  s[328][1] =  -11.8354568846;
  s[328][2] =   12.6029015279;
  s[329][0] =    5.5207135146;
  s[329][1] =   -8.2356073043;
  s[329][2] =   12.7050918840;
  s[330][0] =    7.3222337065;
  s[330][1] =   -8.4709514204;
  s[330][2] =   10.3461931892;
  s[331][0] =    4.0309310540;
  s[331][1] =  -10.9567294950;
  s[331][2] =   11.1175544466;
  s[332][0] =    7.1002339002;
  s[332][1] =  -11.2902309609;
  s[332][2] =    7.3816563180;
  s[333][0] =    6.5100941498;
  s[333][1] =   -9.9978382593;
  s[333][2] =    7.8725943686;
  s[334][0] =    7.1432847574;
  s[334][1] =   -8.7170079511;
  s[334][2] =    7.1856020477;
  s[335][0] =    7.9702558991;
  s[335][1] =   -8.8157279606;
  s[335][2] =    5.9903392420;
  s[336][0] =    9.6192503946;
  s[336][1] =   -8.7085202232;
  s[336][2] =    3.7543665672;
  s[337][0] =    6.0430781154;
  s[337][1] =  -10.0090562843;
  s[337][2] =    9.2952982471;
  s[338][0] =    6.4542330837;
  s[338][1] =  -11.9019490373;
  s[338][2] =    6.5024311195;
  s[339][0] =    9.2196174336;
  s[339][1] =  -13.2809334936;
  s[339][2] =    5.8955992449;
  s[340][0] =    8.9175635208;
  s[340][1] =  -13.0630333578;
  s[340][2] =    7.3637779799;
  s[341][0] =    8.5716801814;
  s[341][1] =  -14.3426602392;
  s[341][2] =    8.2084778947;
  s[342][0] =    9.7945117200;
  s[342][1] =  -15.2588654600;
  s[342][2] =    8.1376980238;
  s[343][0] =    8.2620436546;
  s[343][1] =  -11.8423235451;
  s[343][2] =    7.8112494100;
  s[344][0] =   10.4031253925;
  s[344][1] =  -13.0291966273;
  s[344][2] =    5.5693597549;
  s[345][0] =    7.4365598456;
  s[345][1] =  -15.0006803433;
  s[345][2] =    7.6636250472;
  s[346][0] =    9.3804240983;
  s[346][1] =  -15.5142295544;
  s[346][2] =    3.4536580585;
  s[347][0] =    8.7014104612;
  s[347][1] =  -14.1645225876;
  s[347][2] =    3.5549093135;
  s[348][0] =    9.2340402366;
  s[348][1] =  -13.0994084478;
  s[348][2] =    2.5084169512;
  s[349][0] =   10.3788447750;
  s[349][1] =  -13.8326524269;
  s[349][2] =    1.8131229505;
  s[350][0] =    8.1848176852;
  s[350][1] =  -12.6462214714;
  s[350][2] =    1.4909936986;
  s[351][0] =    8.3792144911;
  s[351][1] =  -13.7603399748;
  s[351][2] =    4.9282785934;
  s[352][0] =    8.6921991396;
  s[352][1] =  -16.4366581833;
  s[352][2] =    2.9539613495;
  s[353][0] =   11.6831949082;
  s[353][1] =  -17.3139625139;
  s[353][2] =    2.1515583605;
  s[354][0] =   11.5077386742;
  s[354][1] =  -16.9595410311;
  s[354][2] =    3.6155485940;
  s[355][0] =   11.7768546271;
  s[355][1] =  -18.0153828225;
  s[355][2] =    4.7763741374;
  s[356][0] =   12.4385519447;
  s[356][1] =  -19.2602164961;
  s[356][2] =    4.1813837767;
  s[357][0] =   10.6508391281;
  s[357][1] =  -15.7951185629;
  s[357][2] =    3.8763245013;
  s[358][0] =   12.5756234837;
  s[358][1] =  -16.6197032490;
  s[358][2] =    1.6148424050;
  s[359][0] =   10.5371272939;
  s[359][1] =  -18.3355755275;
  s[359][2] =    5.3908364131;
  s[360][0] =   10.2589267151;
  s[360][1] =  -17.3209856811;
  s[360][2] =   -0.7365850113;
  s[361][0] =   11.3763547628;
  s[361][1] =  -18.0418152292;
  s[361][2] =   -0.1372148401;
  s[362][0] =   11.6634865722;
  s[362][1] =  -19.2521720123;
  s[362][2] =   -1.1044054539;
  s[363][0] =    9.8976385292;
  s[363][1] =  -18.5651800348;
  s[363][2] =   -1.0532212399;
  s[364][0] =   10.3436852427;
  s[364][1] =  -19.8502551355;
  s[364][2] =   -1.5846652316;
  s[365][0] =   11.0483656664;
  s[365][1] =  -18.1663058036;
  s[365][2] =    1.2806093051;
  s[366][0] =   10.3090121654;
  s[366][1] =  -16.9600342084;
  s[366][2] =   -1.9322261880;
  s[367][0] =    4.3087346466;
  s[367][1] =  -14.9009862740;
  s[367][2] =    5.0331625850;
  s[368][0] =    3.5966687868;
  s[368][1] =  -14.4102956057;
  s[368][2] =    6.2505893482;
  s[369][0] =    4.1511377377;
  s[369][1] =  -15.1308765285;
  s[369][2] =    7.5484589075;
  s[370][0] =    5.7255742923;
  s[370][1] =  -13.1141790991;
  s[370][2] =    6.7555891129;
  s[371][0] =    5.2944548588;
  s[371][1] =  -14.1576451470;
  s[371][2] =    7.7935834780;
  s[372][0] =    4.7309867040;
  s[372][1] =  -16.5183648059;
  s[372][2] =    7.2625605683;
  s[373][0] =    3.6904563862;
  s[373][1] =  -12.9475852931;
  s[373][2] =    6.2662856151;
  s[374][0] =    5.4246993858;
  s[374][1] =  -14.4281169943;
  s[374][2] =    4.7247551646;
  s[375][0] =    4.8332125665;
  s[375][1] =  -17.6831280626;
  s[375][2] =    3.0079399407;
  s[376][0] =    4.1125766982;
  s[376][1] =  -16.3985547841;
  s[376][2] =    2.9702624169;
  s[377][0] =    4.0988447270;
  s[377][1] =  -15.6020996935;
  s[377][2] =    1.6257046501;
  s[378][0] =    1.9149418384;
  s[378][1] =  -16.2944641876;
  s[378][2] =    2.4572352075;
  s[379][0] =    2.8055622394;
  s[379][1] =  -17.3459186228;
  s[379][2] =    0.4017652075;
  s[380][0] =    2.7034448709;
  s[380][1] =  -16.0253813642;
  s[380][2] =    1.1683387883;
  s[381][0] =    3.7298922924;
  s[381][1] =  -15.8418787000;
  s[381][2] =    4.2512519423;
  s[382][0] =    4.0792358427;
  s[382][1] =  -18.6793365338;
  s[382][2] =    3.0640968962;
  s[383][0] =    7.1468410511;
  s[383][1] =  -19.6532544195;
  s[383][2] =    1.7011969733;
  s[384][0] =    6.5088177075;
  s[384][1] =  -19.3199561435;
  s[384][2] =    3.0094609064;
  s[385][0] =    7.3702057801;
  s[385][1] =  -19.6773260792;
  s[385][2] =    4.2618119375;
  s[386][0] =    8.7206358266;
  s[386][1] =  -19.3434319494;
  s[386][2] =    3.8401549076;
  s[387][0] =    6.1581606677;
  s[387][1] =  -17.9387989182;
  s[387][2] =    2.9946973121;
  s[388][0] =    8.8772996261;
  s[388][1] =  -18.7965526692;
  s[388][2] =    2.6245138496;
  s[389][0] =    8.3958431532;
  s[389][1] =  -19.6088490543;
  s[389][2] =    1.5918269525;
  s[390][0] =    9.7115032144;
  s[390][1] =  -19.5872114133;
  s[390][2] =    4.5627564464;
  s[391][0] =    7.9251941256;
  s[391][1] =  -21.3549473767;
  s[391][2] =   -0.6566861008;
  s[392][0] =    6.8990492402;
  s[392][1] =  -20.2998248564;
  s[392][2] =   -0.6873719013;
  s[393][0] =    6.3867612772;
  s[393][1] =  -19.9515501036;
  s[393][2] =    0.6100772748;
  s[394][0] =    8.3251186197;
  s[394][1] =  -21.7101714863;
  s[394][2] =    0.4735692213;
  t[  0][0] =   27.3610000000;
  t[  0][1] =   31.9540000000;
  t[  0][2] =   35.3130000000;
  t[  1][0] =   26.5840000000;
  t[  1][1] =   32.8040000000;
  t[  1][2] =   36.2440000000;
  t[  2][0] =   25.0780000000;
  t[  2][1] =   32.4150000000;
  t[  2][2] =   36.2380000000;
  t[  3][0] =   22.0560000000;
  t[  3][1] =   32.6730000000;
  t[  3][2] =   39.0370000000;
  t[  4][0] =   24.2880000000;
  t[  4][1] =   32.8820000000;
  t[  4][2] =   37.4820000000;
  t[  5][0] =   26.7780000000;
  t[  5][1] =   34.2130000000;
  t[  5][2] =   35.8800000000;
  t[  6][0] =   27.3830000000;
  t[  6][1] =   32.2350000000;
  t[  6][2] =   34.0900000000;
  t[  7][0] =   22.5630000000;
  t[  7][1] =   32.3540000000;
  t[  7][2] =   37.3340000000;
  t[  8][0] =   29.6070000000;
  t[  8][1] =   12.0430000000;
  t[  8][2] =   16.2630000000;
  t[  9][0] =   28.1250000000;
  t[  9][1] =   12.0460000000;
  t[  9][2] =   16.3160000000;
  t[ 10][0] =   27.5090000000;
  t[ 10][1] =   11.9900000000;
  t[ 10][2] =   14.8920000000;
  t[ 11][0] =   25.1950000000;
  t[ 11][1] =   12.2080000000;
  t[ 11][2] =   13.7060000000;
  t[ 12][0] =   25.6650000000;
  t[ 12][1] =   11.6660000000;
  t[ 12][2] =   12.3360000000;
  t[ 13][0] =   25.9890000000;
  t[ 13][1] =   11.6460000000;
  t[ 13][2] =   14.9120000000;
  t[ 14][0] =   27.6610000000;
  t[ 14][1] =   13.2220000000;
  t[ 14][2] =   16.9970000000;
  t[ 15][0] =   24.8700000000;
  t[ 15][1] =   12.2530000000;
  t[ 15][2] =   11.2540000000;
  t[ 16][0] =   30.2270000000;
  t[ 16][1] =   10.9900000000;
  t[ 16][2] =   16.5420000000;
  t[ 17][0] =   32.1680000000;
  t[ 17][1] =   13.9030000000;
  t[ 17][2] =   17.1810000000;
  t[ 18][0] =   31.6700000000;
  t[ 18][1] =   13.3770000000;
  t[ 18][2] =   15.8830000000;
  t[ 19][0] =   31.9820000000;
  t[ 19][1] =   14.3230000000;
  t[ 19][2] =   14.6840000000;
  t[ 20][0] =   33.4940000000;
  t[ 20][1] =   14.5530000000;
  t[ 20][2] =   14.4390000000;
  t[ 21][0] =   30.2550000000;
  t[ 21][1] =   13.1910000000;
  t[ 21][2] =   15.9580000000;
  t[ 22][0] =   33.0990000000;
  t[ 22][1] =   13.2870000000;
  t[ 22][2] =   17.7530000000;
  t[ 23][0] =   31.4220000000;
  t[ 23][1] =   13.7720000000;
  t[ 23][2] =   13.4890000000;
  t[ 24][0] =   31.5010000000;
  t[ 24][1] =   15.4630000000;
  t[ 24][2] =   20.0680000000;
  t[ 25][0] =   32.2070000000;
  t[ 25][1] =   15.7940000000;
  t[ 25][2] =   18.7990000000;
  t[ 26][0] =   32.1380000000;
  t[ 26][1] =   17.3200000000;
  t[ 26][2] =   18.4570000000;
  t[ 27][0] =   32.6180000000;
  t[ 27][1] =   19.6910000000;
  t[ 27][2] =   19.3510000000;
  t[ 28][0] =   33.9730000000;
  t[ 28][1] =   17.8000000000;
  t[ 28][2] =   20.3390000000;
  t[ 29][0] =   33.2210000000;
  t[ 29][1] =   18.2870000000;
  t[ 29][2] =   19.0670000000;
  t[ 30][0] =   31.6480000000;
  t[ 30][1] =   15.0400000000;
  t[ 30][2] =   17.7100000000;
  t[ 31][0] =   30.2890000000;
  t[ 31][1] =   15.7630000000;
  t[ 31][2] =   20.2060000000;
  t[ 32][0] =   32.5120000000;
  t[ 32][1] =   15.2360000000;
  t[ 32][2] =   23.3830000000;
  t[ 33][0] =   31.6700000000;
  t[ 33][1] =   14.5410000000;
  t[ 33][2] =   22.3790000000;
  t[ 34][0] =   31.6570000000;
  t[ 34][1] =   13.0070000000;
  t[ 34][2] =   22.6260000000;
  t[ 35][0] =   30.6350000000;
  t[ 35][1] =   10.7270000000;
  t[ 35][2] =   22.0150000000;
  t[ 36][0] =   29.6450000000;
  t[ 36][1] =    9.9070000000;
  t[ 36][2] =   21.1490000000;
  t[ 37][0] =   30.6700000000;
  t[ 37][1] =   12.2450000000;
  t[ 37][2] =   21.7010000000;
  t[ 38][0] =   32.1760000000;
  t[ 38][1] =   14.8540000000;
  t[ 38][2] =   21.0710000000;
  t[ 39][0] =   29.8180000000;
  t[ 39][1] =   10.1230000000;
  t[ 39][2] =   19.7080000000;
  t[ 40][0] =   33.7580000000;
  t[ 40][1] =   15.0940000000;
  t[ 40][2] =   23.3250000000;
  t[ 41][0] =   31.9210000000;
  t[ 41][1] =   17.5390000000;
  t[ 41][2] =   26.2840000000;
  t[ 42][0] =   32.7040000000;
  t[ 42][1] =   16.7440000000;
  t[ 42][2] =   25.3160000000;
  t[ 43][0] =   31.9420000000;
  t[ 43][1] =   15.9990000000;
  t[ 43][2] =   24.3510000000;
  t[ 44][0] =   30.6690000000;
  t[ 44][1] =   17.4890000000;
  t[ 44][2] =   26.2730000000;
  t[ 45][0] =   32.7550000000;
  t[ 45][1] =   20.3840000000;
  t[ 45][2] =   28.3470000000;
  t[ 46][0] =   32.0230000000;
  t[ 46][1] =   19.0880000000;
  t[ 46][2] =   28.2490000000;
  t[ 47][0] =   32.0300000000;
  t[ 47][1] =   18.2550000000;
  t[ 47][2] =   29.5700000000;
  t[ 48][0] =   31.0110000000;
  t[ 48][1] =   17.9570000000;
  t[ 48][2] =   31.8550000000;
  t[ 49][0] =   31.2450000000;
  t[ 49][1] =   18.9050000000;
  t[ 49][2] =   30.7440000000;
  t[ 50][0] =   32.5950000000;
  t[ 50][1] =   18.2890000000;
  t[ 50][2] =   27.1960000000;
  t[ 51][0] =   33.9720000000;
  t[ 51][1] =   20.3590000000;
  t[ 51][2] =   28.6510000000;
  t[ 52][0] =   31.8680000000;
  t[ 52][1] =   23.8020000000;
  t[ 52][2] =   29.0350000000;
  t[ 53][0] =   32.6850000000;
  t[ 53][1] =   22.8970000000;
  t[ 53][2] =   28.1770000000;
  t[ 54][0] =   32.8590000000;
  t[ 54][1] =   23.5260000000;
  t[ 54][2] =   26.7590000000;
  t[ 55][0] =   33.9890000000;
  t[ 55][1] =   22.8670000000;
  t[ 55][2] =   25.9390000000;
  t[ 56][0] =   32.1150000000;
  t[ 56][1] =   21.5620000000;
  t[ 56][2] =   28.1080000000;
  t[ 57][0] =   30.7320000000;
  t[ 57][1] =   23.4250000000;
  t[ 57][2] =   29.4020000000;
  t[ 58][0] =   31.6370000000;
  t[ 58][1] =   23.4670000000;
  t[ 58][2] =   26.0230000000;
  t[ 59][0] =   31.6970000000;
  t[ 59][1] =   27.3710000000;
  t[ 59][2] =   29.6250000000;
  t[ 60][0] =   31.6420000000;
  t[ 60][1] =   26.0010000000;
  t[ 60][2] =   30.2130000000;
  t[ 61][0] =   32.1730000000;
  t[ 61][1] =   26.0450000000;
  t[ 61][2] =   31.6810000000;
  t[ 62][0] =   32.0170000000;
  t[ 62][1] =   24.7120000000;
  t[ 62][2] =   32.4440000000;
  t[ 63][0] =   32.3450000000;
  t[ 63][1] =   25.0210000000;
  t[ 63][2] =   29.4150000000;
  t[ 64][0] =   32.6370000000;
  t[ 64][1] =   27.6860000000;
  t[ 64][2] =   28.8560000000;
  t[ 65][0] =   33.5420000000;
  t[ 65][1] =   26.4450000000;
  t[ 65][2] =   31.7120000000;
  t[ 66][0] =   30.3140000000;
  t[ 66][1] =   30.5020000000;
  t[ 66][2] =   30.8570000000;
  t[ 67][0] =   30.7640000000;
  t[ 67][1] =   29.6920000000;
  t[ 67][2] =   29.6940000000;
  t[ 68][0] =   29.9440000000;
  t[ 68][1] =   30.0210000000;
  t[ 68][2] =   28.4080000000;
  t[ 69][0] =   28.4070000000;
  t[ 69][1] =   30.0550000000;
  t[ 69][2] =   28.5910000000;
  t[ 70][0] =   30.7490000000;
  t[ 70][1] =   28.2790000000;
  t[ 70][2] =   29.9860000000;
  t[ 71][0] =   29.6380000000;
  t[ 71][1] =   29.9640000000;
  t[ 71][2] =   31.7600000000;
  t[ 72][0] =   30.3450000000;
  t[ 72][1] =   31.2860000000;
  t[ 72][2] =   27.8860000000;
  t[ 73][0] =   29.2950000000;
  t[ 73][1] =   33.7460000000;
  t[ 73][2] =   31.2630000000;
  t[ 74][0] =   30.2280000000;
  t[ 74][1] =   32.7810000000;
  t[ 74][2] =   31.9030000000;
  t[ 75][0] =   31.5180000000;
  t[ 75][1] =   33.4690000000;
  t[ 75][2] =   32.4570000000;
  t[ 76][0] =   30.3400000000;
  t[ 76][1] =   34.9380000000;
  t[ 76][2] =   34.2250000000;
  t[ 77][0] =   31.4590000000;
  t[ 77][1] =   34.0280000000;
  t[ 77][2] =   33.9110000000;
  t[ 78][0] =   30.5960000000;
  t[ 78][1] =   31.8280000000;
  t[ 78][2] =   30.8860000000;
  t[ 79][0] =   29.6960000000;
  t[ 79][1] =   34.3810000000;
  t[ 79][2] =   30.2580000000;
  t[ 80][0] =   27.7480000000;
  t[ 80][1] =   28.8370000000;
  t[ 80][2] =   34.6350000000;
  t[ 81][0] =   28.7160000000;
  t[ 81][1] =   29.8860000000;
  t[ 81][2] =   35.0600000000;
  t[ 82][0] =   29.9070000000;
  t[ 82][1] =   29.3170000000;
  t[ 82][2] =   35.8910000000;
  t[ 83][0] =   30.8000000000;
  t[ 83][1] =   28.3260000000;
  t[ 83][2] =   35.0990000000;
  t[ 84][0] =   28.0020000000;
  t[ 84][1] =   30.8690000000;
  t[ 84][2] =   35.8130000000;
  t[ 85][0] =   27.0670000000;
  t[ 85][1] =   28.2380000000;
  t[ 85][2] =   35.5030000000;
  t[ 86][0] =   30.7320000000;
  t[ 86][1] =   30.3950000000;
  t[ 86][2] =   36.3380000000;
  t[ 87][0] =   26.1200000000;
  t[ 87][1] =   35.3980000000;
  t[ 87][2] =   32.0800000000;
  t[ 88][0] =   27.0500000000;
  t[ 88][1] =   34.7800000000;
  t[ 88][2] =   31.1070000000;
  t[ 89][0] =   26.2370000000;
  t[ 89][1] =   33.9760000000;
  t[ 89][2] =   30.0630000000;
  t[ 90][0] =   28.0280000000;
  t[ 90][1] =   33.9180000000;
  t[ 90][2] =   31.7250000000;
  t[ 91][0] =   25.9040000000;
  t[ 91][1] =   34.8440000000;
  t[ 91][2] =   33.1820000000;
  t[ 92][0] =   23.2990000000;
  t[ 92][1] =   36.2550000000;
  t[ 92][2] =   32.8020000000;
  t[ 93][0] =   24.4430000000;
  t[ 93][1] =   37.1730000000;
  t[ 93][2] =   32.5390000000;
  t[ 94][0] =   23.9790000000;
  t[ 94][1] =   38.5540000000;
  t[ 94][2] =   31.9420000000;
  t[ 95][0] =   24.9960000000;
  t[ 95][1] =   39.6840000000;
  t[ 95][2] =   32.2670000000;
  t[ 96][0] =   23.6870000000;
  t[ 96][1] =   38.5430000000;
  t[ 96][2] =   30.4140000000;
  t[ 97][0] =   25.4510000000;
  t[ 97][1] =   36.5280000000;
  t[ 97][2] =   31.7400000000;
  t[ 98][0] =   22.9670000000;
  t[ 98][1] =   36.0530000000;
  t[ 98][2] =   33.9960000000;
  t[ 99][0] =   21.5650000000;
  t[ 99][1] =   33.5880000000;
  t[ 99][2] =   31.0290000000;
  t[100][0] =   21.5030000000;
  t[100][1] =   34.7770000000;
  t[100][2] =   31.9220000000;
  t[101][0] =   20.1870000000;
  t[101][1] =   35.5970000000;
  t[101][2] =   31.7400000000;
  t[102][0] =   19.9340000000;
  t[102][1] =   36.1660000000;
  t[102][2] =   30.4000000000;
  t[103][0] =   22.6370000000;
  t[103][1] =   35.6480000000;
  t[103][2] =   31.7840000000;
  t[104][0] =   22.4480000000;
  t[104][1] =   33.4670000000;
  t[104][2] =   30.1420000000;
  t[105][0] =   20.1870000000;
  t[105][1] =   31.7860000000;
  t[105][2] =   28.9760000000;
  t[106][0] =   20.4120000000;
  t[106][1] =   31.4730000000;
  t[106][2] =   30.4070000000;
  t[107][0] =   19.2140000000;
  t[107][1] =   30.6760000000;
  t[107][2] =   30.9690000000;
  t[108][0] =   20.6020000000;
  t[108][1] =   32.6510000000;
  t[108][2] =   31.1930000000;
  t[109][0] =   20.6840000000;
  t[109][1] =   31.0260000000;
  t[109][2] =   28.1180000000;
  t[110][0] =   20.5730000000;
  t[110][1] =   33.5570000000;
  t[110][2] =   26.5330000000;
  t[111][0] =   19.2800000000;
  t[111][1] =   33.2730000000;
  t[111][2] =   27.2160000000;
  t[112][0] =   18.3190000000;
  t[112][1] =   34.4870000000;
  t[112][2] =   27.1620000000;
  t[113][0] =   19.4810000000;
  t[113][1] =   32.8740000000;
  t[113][2] =   28.5830000000;
  t[114][0] =   20.8590000000;
  t[114][1] =   32.9540000000;
  t[114][2] =   25.4700000000;
  t[115][0] =   23.6130000000;
  t[115][1] =   33.4850000000;
  t[115][2] =   26.4640000000;
  t[116][0] =   22.7690000000;
  t[116][1] =   34.7130000000;
  t[116][2] =   26.5270000000;
  t[117][0] =   23.4390000000;
  t[117][1] =   35.8580000000;
  t[117][2] =   27.3400000000;
  t[118][0] =   24.7570000000;
  t[118][1] =   36.3720000000;
  t[118][2] =   26.7160000000;
  t[119][0] =   21.4670000000;
  t[119][1] =   34.4240000000;
  t[119][2] =   27.0660000000;
  t[120][0] =   24.2060000000;
  t[120][1] =   33.2160000000;
  t[120][2] =   25.3880000000;
  t[121][0] =   22.5490000000;
  t[121][1] =   36.9660000000;
  t[121][2] =   27.4450000000;
  t[122][0] =   23.9590000000;
  t[122][1] =   30.4940000000;
  t[122][2] =   26.4120000000;
  t[123][0] =   24.3950000000;
  t[123][1] =   31.3820000000;
  t[123][2] =   27.5250000000;
  t[124][0] =   24.2050000000;
  t[124][1] =   30.6710000000;
  t[124][2] =   28.8810000000;
  t[125][0] =   23.7060000000;
  t[125][1] =   32.6460000000;
  t[125][2] =   27.5310000000;
  t[126][0] =   24.8380000000;
  t[126][1] =   29.9580000000;
  t[126][2] =   25.6950000000;
  t[127][0] =   22.6350000000;
  t[127][1] =   30.1370000000;
  t[127][2] =   23.7500000000;
  t[128][0] =   22.0850000000;
  t[128][1] =   29.6030000000;
  t[128][2] =   25.0240000000;
  t[129][0] =   20.5250000000;
  t[129][1] =   29.6570000000;
  t[129][2] =   24.9990000000;
  t[130][0] =   18.4870000000;
  t[130][1] =   29.0050000000;
  t[130][2] =   23.5540000000;
  t[131][0] =   19.8800000000;
  t[131][1] =   28.7050000000;
  t[131][2] =   23.9450000000;
  t[132][0] =   22.6390000000;
  t[132][1] =   30.3070000000;
  t[132][2] =   26.1460000000;
  t[133][0] =   23.1890000000;
  t[133][1] =   29.3110000000;
  t[133][2] =   22.9940000000;
  t[134][0] =   24.5630000000;
  t[134][1] =   31.6670000000;
  t[134][2] =   22.0140000000;
  t[135][0] =   23.1180000000;
  t[135][1] =   32.0070000000;
  t[135][2] =   22.1880000000;
  t[136][0] =   22.9120000000;
  t[136][1] =   33.5550000000;
  t[136][2] =   22.0830000000;
  t[137][0] =   20.3150000000;
  t[137][1] =   34.0320000000;
  t[137][2] =   22.2310000000;
  t[138][0] =   19.5140000000;
  t[138][1] =   32.7030000000;
  t[138][2] =   22.2080000000;
  t[139][0] =   21.6090000000;
  t[139][1] =   34.0410000000;
  t[139][2] =   21.3710000000;
  t[140][0] =   22.5610000000;
  t[140][1] =   31.4510000000;
  t[140][2] =   23.4040000000;
  t[141][0] =   18.7570000000;
  t[141][1] =   32.5560000000;
  t[141][2] =   23.4560000000;
  t[142][0] =   24.9430000000;
  t[142][1] =   31.1810000000;
  t[142][2] =   20.9170000000;
  t[143][0] =   27.0180000000;
  t[143][1] =   30.0580000000;
  t[143][2] =   22.6460000000;
  t[144][0] =   26.8470000000;
  t[144][1] =   31.5040000000;
  t[144][2] =   22.9760000000;
  t[145][0] =   27.6030000000;
  t[145][1] =   31.9210000000;
  t[145][2] =   24.2870000000;
  t[146][0] =   29.0820000000;
  t[146][1] =   31.4310000000;
  t[146][2] =   24.2980000000;
  t[147][0] =   27.5800000000;
  t[147][1] =   33.4630000000;
  t[147][2] =   24.4890000000;
  t[148][0] =   25.4500000000;
  t[148][1] =   31.8550000000;
  t[148][2] =   23.0270000000;
  t[149][0] =   27.7490000000;
  t[149][1] =   29.7440000000;
  t[149][2] =   21.6740000000;
  t[150][0] =   27.5660000000;
  t[150][1] =   26.5220000000;
  t[150][2] =   32.0590000000;
  t[151][0] =   26.7500000000;
  t[151][1] =   27.5700000000;
  t[151][2] =   32.7230000000;
  t[152][0] =   25.8030000000;
  t[152][1] =   28.2070000000;
  t[152][2] =   31.6650000000;
  t[153][0] =   24.8850000000;
  t[153][1] =   29.2360000000;
  t[153][2] =   32.2180000000;
  t[154][0] =   23.1640000000;
  t[154][1] =   31.2080000000;
  t[154][2] =   33.2960000000;
  t[155][0] =   27.6280000000;
  t[155][1] =   28.5430000000;
  t[155][2] =   33.3180000000;
  t[156][0] =   28.6660000000;
  t[156][1] =   26.8500000000;
  t[156][2] =   31.5520000000;
  t[157][0] =   22.3610000000;
  t[157][1] =   32.1360000000;
  t[157][2] =   33.8630000000;
  t[158][0] =   25.9670000000;
  t[158][1] =   27.2340000000;
  t[158][2] =   21.8160000000;
  t[159][0] =   26.5230000000;
  t[159][1] =   27.6850000000;
  t[159][2] =   23.1260000000;
  t[160][0] =   25.9320000000;
  t[160][1] =   26.8750000000;
  t[160][2] =   24.3150000000;
  t[161][0] =   26.6920000000;
  t[161][1] =   26.9300000000;
  t[161][2] =   25.6060000000;
  t[162][0] =   28.0300000000;
  t[162][1] =   26.6950000000;
  t[162][2] =   28.0980000000;
  t[163][0] =   26.4070000000;
  t[163][1] =   29.0940000000;
  t[163][2] =   23.3760000000;
  t[164][0] =   26.5940000000;
  t[164][1] =   26.3510000000;
  t[164][2] =   21.1820000000;
  t[165][0] =   25.4110000000;
  t[165][1] =   27.9000000000;
  t[165][2] =   18.9670000000;
  t[166][0] =   24.3460000000;
  t[166][1] =   27.5580000000;
  t[166][2] =   19.9460000000;
  t[167][0] =   23.0730000000;
  t[167][1] =   28.4180000000;
  t[167][2] =   19.6600000000;
  t[168][0] =   21.0480000000;
  t[168][1] =   26.8020000000;
  t[168][2] =   19.3560000000;
  t[169][0] =   19.5570000000;
  t[169][1] =   26.5100000000;
  t[169][2] =   19.6800000000;
  t[170][0] =   21.7250000000;
  t[170][1] =   27.8900000000;
  t[170][2] =   20.2350000000;
  t[171][0] =   24.8390000000;
  t[171][1] =   27.7620000000;
  t[171][2] =   21.2830000000;
  t[172][0] =   19.2850000000;
  t[172][1] =   26.1680000000;
  t[172][2] =   21.0810000000;
  t[173][0] =   25.7700000000;
  t[173][1] =   27.0250000000;
  t[173][2] =   18.1420000000;
  t[174][0] =   28.2410000000;
  t[174][1] =   28.5590000000;
  t[174][2] =   18.2150000000;
  t[175][0] =   27.0930000000;
  t[175][1] =   29.5050000000;
  t[175][2] =   18.1280000000;
  t[176][0] =   27.5070000000;
  t[176][1] =   30.9760000000;
  t[176][2] =   18.4060000000;
  t[177][0] =   28.9220000000;
  t[177][1] =   32.9250000000;
  t[177][2] =   17.6220000000;
  t[178][0] =   28.6640000000;
  t[178][1] =   31.4780000000;
  t[178][2] =   17.4970000000;
  t[179][0] =   26.0130000000;
  t[179][1] =   29.1180000000;
  t[179][2] =   18.9930000000;
  t[180][0] =   29.0870000000;
  t[180][1] =   33.6920000000;
  t[180][2] =   16.5300000000;
  t[181][0] =   28.6620000000;
  t[181][1] =   28.0590000000;
  t[181][2] =   17.1460000000;
  t[182][0] =   29.0270000000;
  t[182][1] =   33.4630000000;
  t[182][2] =   18.7480000000;
  t[183][0] =   29.5410000000;
  t[183][1] =   25.9420000000;
  t[183][2] =   18.9040000000;
  t[184][0] =   29.8400000000;
  t[184][1] =   27.2320000000;
  t[184][2] =   19.5880000000;
  t[185][0] =   30.0990000000;
  t[185][1] =   27.0130000000;
  t[185][2] =   21.1060000000;
  t[186][0] =   31.1880000000;
  t[186][1] =   26.0380000000;
  t[186][2] =   21.3970000000;
  t[187][0] =   33.2560000000;
  t[187][1] =   24.1870000000;
  t[187][2] =   21.9770000000;
  t[188][0] =   28.7900000000;
  t[188][1] =   28.2040000000;
  t[188][2] =   19.4050000000;
  t[189][0] =   30.3740000000;
  t[189][1] =   25.4980000000;
  t[189][2] =   18.0780000000;
  t[190][0] =   34.2430000000;
  t[190][1] =   23.3140000000;
  t[190][2] =   22.2760000000;
  t[191][0] =   27.9530000000;
  t[191][1] =   24.1600000000;
  t[191][2] =   17.0310000000;
  t[192][0] =   27.9560000000;
  t[192][1] =   24.0610000000;
  t[192][2] =   18.5160000000;
  t[193][0] =   26.5530000000;
  t[193][1] =   23.6880000000;
  t[193][2] =   19.0530000000;
  t[194][0] =   28.3850000000;
  t[194][1] =   25.2760000000;
  t[194][2] =   19.1510000000;
  t[195][0] =   28.6150000000;
  t[195][1] =   23.3160000000;
  t[195][2] =   16.3740000000;
  t[196][0] =   28.6950000000;
  t[196][1] =   25.5010000000;
  t[196][2] =   14.4390000000;
  t[197][0] =   27.3090000000;
  t[197][1] =   25.3710000000;
  t[197][2] =   14.9700000000;
  t[198][0] =   26.4440000000;
  t[198][1] =   26.5910000000;
  t[198][2] =   14.5400000000;
  t[199][0] =   24.9980000000;
  t[199][1] =   26.2970000000;
  t[199][2] =   14.6010000000;
  t[200][0] =   27.2950000000;
  t[200][1] =   25.1660000000;
  t[200][2] =   16.3960000000;
  t[201][0] =   24.3150000000;
  t[201][1] =   26.3890000000;
  t[201][2] =   15.7510000000;
  t[202][0] =   29.0320000000;
  t[202][1] =   24.7670000000;
  t[202][2] =   13.4800000000;
  t[203][0] =   24.3840000000;
  t[203][1] =   25.9270000000;
  t[203][2] =   13.5750000000;
  t[204][0] =   31.7770000000;
  t[204][1] =   25.2520000000;
  t[204][2] =   14.6920000000;
  t[205][0] =   30.9630000000;
  t[205][1] =   26.4990000000;
  t[205][2] =   14.6000000000;
  t[206][0] =   31.6250000000;
  t[206][1] =   27.6500000000;
  t[206][2] =   15.4180000000;
  t[207][0] =   30.9980000000;
  t[207][1] =   28.9820000000;
  t[207][2] =   15.2600000000;
  t[208][0] =   29.5920000000;
  t[208][1] =   26.3410000000;
  t[208][2] =   15.0050000000;
  t[209][0] =   32.9220000000;
  t[209][1] =   25.2840000000;
  t[209][2] =   14.1790000000;
  t[210][0] =   31.2460000000;
  t[210][1] =   21.8000000000;
  t[210][2] =   14.5480000000;
  t[211][0] =   31.9970000000;
  t[211][1] =   22.8440000000;
  t[211][2] =   15.3010000000;
  t[212][0] =   32.2300000000;
  t[212][1] =   22.4490000000;
  t[212][2] =   16.7890000000;
  t[213][0] =   33.3080000000;
  t[213][1] =   23.2320000000;
  t[213][2] =   17.4270000000;
  t[214][0] =   31.3140000000;
  t[214][1] =   24.1070000000;
  t[214][2] =   15.2600000000;
  t[215][0] =   33.0660000000;
  t[215][1] =   24.4150000000;
  t[215][2] =   18.0160000000;
  t[216][0] =   31.7260000000;
  t[216][1] =   20.6400000000;
  t[216][2] =   14.5300000000;
  t[217][0] =   34.4860000000;
  t[217][1] =   22.8080000000;
  t[217][2] =   17.4210000000;
  t[218][0] =   28.3500000000;
  t[218][1] =   20.3640000000;
  t[218][2] =   13.7850000000;
  t[219][0] =   29.3810000000;
  t[219][1] =   21.1300000000;
  t[219][2] =   13.0580000000;
  t[220][0] =   30.1130000000;
  t[220][1] =   22.0690000000;
  t[220][2] =   13.8490000000;
  t[221][0] =   27.9210000000;
  t[221][1] =   19.3230000000;
  t[221][2] =   13.2300000000;
  t[222][0] =   25.5660000000;
  t[222][1] =   20.6140000000;
  t[222][2] =   15.5630000000;
  t[223][0] =   26.9170000000;
  t[223][1] =   20.0250000000;
  t[223][2] =   15.7830000000;
  t[224][0] =   27.3930000000;
  t[224][1] =   20.0160000000;
  t[224][2] =   17.2780000000;
  t[225][0] =   26.4270000000;
  t[225][1] =   19.2310000000;
  t[225][2] =   18.2070000000;
  t[226][0] =   28.8270000000;
  t[226][1] =   19.4320000000;
  t[226][2] =   17.4410000000;
  t[227][0] =   27.8540000000;
  t[227][1] =   20.7680000000;
  t[227][2] =   14.9840000000;
  t[228][0] =   25.3750000000;
  t[228][1] =   21.8170000000;
  t[228][2] =   15.8660000000;
  t[229][0] =   27.1260000000;
  t[229][1] =   23.6840000000;
  t[229][2] =   30.1260000000;
  t[230][0] =   27.7650000000;
  t[230][1] =   24.1250000000;
  t[230][2] =   31.3930000000;
  t[231][0] =   27.7990000000;
  t[231][1] =   22.9750000000;
  t[231][2] =   32.4460000000;
  t[232][0] =   29.1270000000;
  t[232][1] =   20.8850000000;
  t[232][2] =   33.2900000000;
  t[233][0] =   28.1880000000;
  t[233][1] =   19.6590000000;
  t[233][2] =   33.4120000000;
  t[234][0] =   28.8060000000;
  t[234][1] =   21.8400000000;
  t[234][2] =   32.1050000000;
  t[235][0] =   27.0890000000;
  t[235][1] =   25.2530000000;
  t[235][2] =   31.9820000000;
  t[236][0] =   28.7930000000;
  t[236][1] =   18.6340000000;
  t[236][2] =   34.2700000000;
  t[237][0] =   25.8800000000;
  t[237][1] =   23.7250000000;
  t[237][2] =   30.0340000000;
  t[238][0] =   22.3130000000;
  t[238][1] =   19.2820000000;
  t[238][2] =   15.3540000000;
  t[239][0] =   23.2270000000;
  t[239][1] =   20.2770000000;
  t[239][2] =   14.7330000000;
  t[240][0] =   23.0360000000;
  t[240][1] =   20.4200000000;
  t[240][2] =   13.1880000000;
  t[241][0] =   21.7100000000;
  t[241][1] =   20.8980000000;
  t[241][2] =   12.7340000000;
  t[242][0] =   24.5720000000;
  t[242][1] =   19.8550000000;
  t[242][2] =   15.0310000000;
  t[243][0] =   21.9850000000;
  t[243][1] =   18.2600000000;
  t[243][2] =   14.7040000000;
  t[244][0] =   19.9160000000;
  t[244][1] =   19.0890000000;
  t[244][2] =   18.0470000000;
  t[245][0] =   21.1230000000;
  t[245][1] =   18.5290000000;
  t[245][2] =   17.4020000000;
  t[246][0] =   21.8840000000;
  t[246][1] =   19.4660000000;
  t[246][2] =   16.6280000000;
  t[247][0] =   19.4840000000;
  t[247][1] =   20.2230000000;
  t[247][2] =   17.7340000000;
  t[248][0] =   18.6920000000;
  t[248][1] =   19.1860000000;
  t[248][2] =   21.1090000000;
  t[249][0] =   18.1880000000;
  t[249][1] =   18.6720000000;
  t[249][2] =   19.8120000000;
  t[250][0] =   17.2730000000;
  t[250][1] =   17.4410000000;
  t[250][2] =   20.0760000000;
  t[251][0] =   15.5440000000;
  t[251][1] =   15.8780000000;
  t[251][2] =   19.1810000000;
  t[252][0] =   16.4610000000;
  t[252][1] =   16.9840000000;
  t[252][2] =   18.8370000000;
  t[253][0] =   19.3180000000;
  t[253][1] =   18.3280000000;
  t[253][2] =   18.9960000000;
  t[254][0] =   19.6260000000;
  t[254][1] =   18.5730000000;
  t[254][2] =   21.6820000000;
  t[255][0] =   17.7010000000;
  t[255][1] =   20.8390000000;
  t[255][2] =   23.9770000000;
  t[256][0] =   18.5910000000;
  t[256][1] =   21.0110000000;
  t[256][2] =   22.8020000000;
  t[257][0] =   18.7030000000;
  t[257][1] =   22.5280000000;
  t[257][2] =   22.4640000000;
  t[258][0] =   19.9420000000;
  t[258][1] =   22.5960000000;
  t[258][2] =   20.2210000000;
  t[259][0] =   21.0930000000;
  t[259][1] =   23.3660000000;
  t[259][2] =   21.9670000000;
  t[260][0] =   21.8770000000;
  t[260][1] =   23.4020000000;
  t[260][2] =   20.8160000000;
  t[261][0] =   21.5980000000;
  t[261][1] =   23.8190000000;
  t[261][2] =   23.2010000000;
  t[262][0] =   19.8590000000;
  t[262][1] =   22.8260000000;
  t[262][2] =   21.5960000000;
  t[263][0] =   23.7310000000;
  t[263][1] =   24.2850000000;
  t[263][2] =   22.0930000000;
  t[264][0] =   23.2070000000;
  t[264][1] =   23.8490000000;
  t[264][2] =   20.8610000000;
  t[265][0] =   22.9270000000;
  t[265][1] =   24.2930000000;
  t[265][2] =   23.2510000000;
  t[266][0] =   18.1180000000;
  t[266][1] =   20.2860000000;
  t[266][2] =   21.6570000000;
  t[267][0] =   21.1590000000;
  t[267][1] =   22.9540000000;
  t[267][2] =   19.7790000000;
  t[268][0] =   16.4750000000;
  t[268][1] =   21.0690000000;
  t[268][2] =   23.8450000000;
  t[269][0] =   18.3430000000;
  t[269][1] =   21.6940000000;
  t[269][2] =   27.2910000000;
  t[270][0] =   17.6120000000;
  t[270][1] =   20.6950000000;
  t[270][2] =   26.4710000000;
  t[271][0] =   17.5100000000;
  t[271][1] =   19.3680000000;
  t[271][2] =   27.2850000000;
  t[272][0] =   16.8970000000;
  t[272][1] =   18.1910000000;
  t[272][2] =   26.4920000000;
  t[273][0] =   18.2550000000;
  t[273][1] =   20.5520000000;
  t[273][2] =   25.1850000000;
  t[274][0] =   19.5830000000;
  t[274][1] =   21.8160000000;
  t[274][2] =   27.1390000000;
  t[275][0] =   18.7770000000;
  t[275][1] =   18.9820000000;
  t[275][2] =   27.8160000000;
  t[276][0] =   17.6530000000;
  t[276][1] =   22.7700000000;
  t[276][2] =   30.6010000000;
  t[277][0] =   18.2440000000;
  t[277][1] =   23.1780000000;
  t[277][2] =   29.2990000000;
  t[278][0] =   18.0140000000;
  t[278][1] =   24.6970000000;
  t[278][2] =   29.0320000000;
  t[279][0] =   18.5150000000;
  t[279][1] =   25.5750000000;
  t[279][2] =   30.1320000000;
  t[280][0] =   19.4630000000;
  t[280][1] =   27.1430000000;
  t[280][2] =   32.2990000000;
  t[281][0] =   17.6740000000;
  t[281][1] =   22.3720000000;
  t[281][2] =   28.2590000000;
  t[282][0] =   16.4070000000;
  t[282][1] =   22.8110000000;
  t[282][2] =   30.7370000000;
  t[283][0] =   19.9100000000;
  t[283][1] =   27.8270000000;
  t[283][2] =   33.3750000000;
  t[284][0] =   18.6660000000;
  t[284][1] =   23.3300000000;
  t[284][2] =   33.8010000000;
  t[285][0] =   18.0770000000;
  t[285][1] =   22.2330000000;
  t[285][2] =   32.9960000000;
  t[286][0] =   18.5120000000;
  t[286][1] =   20.8340000000;
  t[286][2] =   33.5230000000;
  t[287][0] =   18.0440000000;
  t[287][1] =   20.5160000000;
  t[287][2] =   34.8950000000;
  t[288][0] =   18.4720000000;
  t[288][1] =   22.4430000000;
  t[288][2] =   31.6340000000;
  t[289][0] =   19.9100000000;
  t[289][1] =   23.4060000000;
  t[289][2] =   33.9520000000;
  t[290][0] =   18.7020000000;
  t[290][1] =   24.8020000000;
  t[290][2] =   36.5600000000;
  t[291][0] =   18.1990000000;
  t[291][1] =   25.2880000000;
  t[291][2] =   35.2540000000;
  t[292][0] =   16.9800000000;
  t[292][1] =   26.2350000000;
  t[292][2] =   35.4520000000;
  t[293][0] =   17.3240000000;
  t[293][1] =   27.5090000000;
  t[293][2] =   36.1200000000;
  t[294][0] =   17.8250000000;
  t[294][1] =   24.2120000000;
  t[294][2] =   34.3920000000;
  t[295][0] =   19.6090000000;
  t[295][1] =   25.4680000000;
  t[295][2] =   37.1140000000;
  t[296][0] =   19.9930000000;
  t[296][1] =   22.8370000000;
  t[296][2] =   38.5900000000;
  t[297][0] =   18.5660000000;
  t[297][1] =   23.2270000000;
  t[297][2] =   38.4610000000;
  t[298][0] =   17.6280000000;
  t[298][1] =   22.0590000000;
  t[298][2] =   38.8520000000;
  t[299][0] =   18.2320000000;
  t[299][1] =   23.6650000000;
  t[299][2] =   37.1350000000;
  t[300][0] =   20.4650000000;
  t[300][1] =   22.6430000000;
  t[300][2] =   39.7350000000;
  t[301][0] =   22.8430000000;
  t[301][1] =   23.3770000000;
  t[301][2] =   36.5180000000;
  t[302][0] =   22.1490000000;
  t[302][1] =   22.4020000000;
  t[302][2] =   37.4110000000;
  t[303][0] =   22.2950000000;
  t[303][1] =   20.9330000000;
  t[303][2] =   36.9050000000;
  t[304][0] =   21.6320000000;
  t[304][1] =   19.8820000000;
  t[304][2] =   37.8320000000;
  t[305][0] =   20.7550000000;
  t[305][1] =   22.7030000000;
  t[305][2] =   37.4790000000;
  t[306][0] =   24.0730000000;
  t[306][1] =   23.2180000000;
  t[306][2] =   36.3260000000;
  t[307][0] =   21.7270000000;
  t[307][1] =   20.8140000000;
  t[307][2] =   35.6000000000;
  t[308][0] =   27.9440000000;
  t[308][1] =   21.1860000000;
  t[308][2] =   27.8290000000;
  t[309][0] =   27.3920000000;
  t[309][1] =   22.5590000000;
  t[309][2] =   27.9110000000;
  t[310][0] =   27.7460000000;
  t[310][1] =   23.3680000000;
  t[310][2] =   26.6320000000;
  t[311][0] =   25.7720000000;
  t[311][1] =   22.7280000000;
  t[311][2] =   25.0770000000;
  t[312][0] =   28.0200000000;
  t[312][1] =   23.4240000000;
  t[312][2] =   24.0730000000;
  t[313][0] =   27.3120000000;
  t[313][1] =   22.7300000000;
  t[313][2] =   25.2680000000;
  t[314][0] =   27.8760000000;
  t[314][1] =   23.1980000000;
  t[314][2] =   29.1030000000;
  t[315][0] =   29.1900000000;
  t[315][1] =   21.0430000000;
  t[315][2] =   27.7670000000;
  t[316][0] =   23.3970000000;
  t[316][1] =   24.6090000000;
  t[316][2] =   33.8100000000;
  t[317][0] =   22.7320000000;
  t[317][1] =   25.3280000000;
  t[317][2] =   34.9410000000;
  t[318][0] =   23.6120000000;
  t[318][1] =   26.4300000000;
  t[318][2] =   35.6210000000;
  t[319][0] =   21.7780000000;
  t[319][1] =   28.1370000000;
  t[319][2] =   36.3980000000;
  t[320][0] =   21.1290000000;
  t[320][1] =   28.8310000000;
  t[320][2] =   37.6180000000;
  t[321][0] =   22.9570000000;
  t[321][1] =   27.2110000000;
  t[321][2] =   36.7990000000;
  t[322][0] =   22.1970000000;
  t[322][1] =   24.4200000000;
  t[322][2] =   35.9240000000;
  t[323][0] =   20.0670000000;
  t[323][1] =   29.7570000000;
  t[323][2] =   37.2080000000;
  t[324][0] =   24.5700000000;
  t[324][1] =   24.9090000000;
  t[324][2] =   33.4750000000;
  t[325][0] =   22.5090000000;
  t[325][1] =   22.6870000000;
  t[325][2] =   30.9430000000;
  t[326][0] =   23.3130000000;
  t[326][1] =   22.7180000000;
  t[326][2] =   32.1870000000;
  t[327][0] =   23.4720000000;
  t[327][1] =   21.2750000000;
  t[327][2] =   32.7530000000;
  t[328][0] =   24.2920000000;
  t[328][1] =   20.3270000000;
  t[328][2] =   31.8420000000;
  t[329][0] =   22.7400000000;
  t[329][1] =   23.6030000000;
  t[329][2] =   33.1700000000;
  t[330][0] =   21.3010000000;
  t[330][1] =   22.3620000000;
  t[330][2] =   30.9900000000;
  t[331][0] =   24.1200000000;
  t[331][1] =   21.3100000000;
  t[331][2] =   34.0230000000;
  t[332][0] =   23.1140000000;
  t[332][1] =   21.4030000000;
  t[332][2] =   27.9340000000;
  t[333][0] =   22.6350000000;
  t[333][1] =   22.7110000000;
  t[333][2] =   28.4500000000;
  t[334][0] =   23.1140000000;
  t[334][1] =   23.8670000000;
  t[334][2] =   27.5270000000;
  t[335][0] =   22.5050000000;
  t[335][1] =   25.1870000000;
  t[335][2] =   27.8580000000;
  t[336][0] =   21.3610000000;
  t[336][1] =   27.7030000000;
  t[336][2] =   28.4620000000;
  t[337][0] =   23.1440000000;
  t[337][1] =   22.8950000000;
  t[337][2] =   29.7690000000;
  t[338][0] =   24.3010000000;
  t[338][1] =   21.0560000000;
  t[338][2] =   28.1520000000;
  t[339][0] =   22.2960000000;
  t[339][1] =   19.5070000000;
  t[339][2] =   25.0970000000;
  t[340][0] =   22.6940000000;
  t[340][1] =   19.4090000000;
  t[340][2] =   26.5170000000;
  t[341][0] =   22.2250000000;
  t[341][1] =   18.0510000000;
  t[341][2] =   27.1370000000;
  t[342][0] =   22.2030000000;
  t[342][1] =   18.0470000000;
  t[342][2] =   28.6800000000;
  t[343][0] =   22.2870000000;
  t[343][1] =   20.6070000000;
  t[343][2] =   27.2080000000;
  t[344][0] =   21.0950000000;
  t[344][1] =   19.7690000000;
  t[344][2] =   24.8470000000;
  t[345][0] =   20.9780000000;
  t[345][1] =   17.5720000000;
  t[345][2] =   26.6330000000;
  t[346][0] =   23.1920000000;
  t[346][1] =   17.7370000000;
  t[346][2] =   22.2910000000;
  t[347][0] =   22.9410000000;
  t[347][1] =   19.1430000000;
  t[347][2] =   22.7160000000;
  t[348][0] =   23.6840000000;
  t[348][1] =   20.2340000000;
  t[348][2] =   21.8780000000;
  t[349][0] =   25.2300000000;
  t[349][1] =   20.2000000000;
  t[349][2] =   21.9740000000;
  t[350][0] =   23.2410000000;
  t[350][1] =   20.2210000000;
  t[350][2] =   20.3890000000;
  t[351][0] =   23.2140000000;
  t[351][1] =   19.2510000000;
  t[351][2] =   24.1250000000;
  t[352][0] =   24.3350000000;
  t[352][1] =   17.2410000000;
  t[352][2] =   22.4450000000;
  t[353][0] =   21.9030000000;
  t[353][1] =   15.5000000000;
  t[353][2] =   19.9360000000;
  t[354][0] =   22.1980000000;
  t[354][1] =   15.6250000000;
  t[354][2] =   21.3850000000;
  t[355][0] =   21.1870000000;
  t[355][1] =   14.8040000000;
  t[355][2] =   22.2470000000;
  t[356][0] =   21.1130000000;
  t[356][1] =   13.3000000000;
  t[356][2] =   21.8750000000;
  t[357][0] =   22.1630000000;
  t[357][1] =   17.0100000000;
  t[357][2] =   21.7760000000;
  t[358][0] =   20.7820000000;
  t[358][1] =   15.8810000000;
  t[358][2] =   19.5220000000;
  t[359][0] =   21.5560000000;
  t[359][1] =   14.9100000000;
  t[359][2] =   23.6210000000;
  t[360][0] =   21.9900000000;
  t[360][1] =   13.1640000000;
  t[360][2] =   17.6970000000;
  t[361][0] =   22.5290000000;
  t[361][1] =   14.5450000000;
  t[361][2] =   17.7350000000;
  t[362][0] =   23.7800000000;
  t[362][1] =   14.6470000000;
  t[362][2] =   16.8200000000;
  t[363][0] =   25.3150000000;
  t[363][1] =   16.2020000000;
  t[363][2] =   15.5610000000;
  t[364][0] =   24.2750000000;
  t[364][1] =   16.1010000000;
  t[364][2] =   16.6080000000;
  t[365][0] =   22.7970000000;
  t[365][1] =   14.9060000000;
  t[365][2] =   19.1050000000;
  t[366][0] =   22.4370000000;
  t[366][1] =   12.2670000000;
  t[366][2] =   18.4470000000;
  t[367][0] =   27.1090000000;
  t[367][1] =   18.6170000000;
  t[367][2] =   25.9300000000;
  t[368][0] =   27.3930000000;
  t[368][1] =   18.7890000000;
  t[368][2] =   27.3840000000;
  t[369][0] =   26.6110000000;
  t[369][1] =   17.7420000000;
  t[369][2] =   28.2550000000;
  t[370][0] =   26.0650000000;
  t[370][1] =   17.0640000000;
  t[370][2] =   30.7410000000;
  t[371][0] =   26.8460000000;
  t[371][1] =   17.9920000000;
  t[371][2] =   29.7810000000;
  t[372][0] =   26.9800000000;
  t[372][1] =   16.2770000000;
  t[372][2] =   27.8560000000;
  t[373][0] =   27.0730000000;
  t[373][1] =   20.1500000000;
  t[373][2] =   27.7230000000;
  t[374][0] =   25.9750000000;
  t[374][1] =   18.8990000000;
  t[374][2] =   25.4680000000;
  t[375][0] =   27.8640000000;
  t[375][1] =   16.2780000000;
  t[375][2] =   23.6010000000;
  t[376][0] =   27.9090000000;
  t[376][1] =   17.7510000000;
  t[376][2] =   23.7380000000;
  t[377][0] =   29.0550000000;
  t[377][1] =   18.3380000000;
  t[377][2] =   22.8660000000;
  t[378][0] =   30.4690000000;
  t[378][1] =   20.2820000000;
  t[378][2] =   23.8180000000;
  t[379][0] =   29.1940000000;
  t[379][1] =   20.6110000000;
  t[379][2] =   21.6200000000;
  t[380][0] =   29.2090000000;
  t[380][1] =   19.8930000000;
  t[380][2] =   22.9950000000;
  t[381][0] =   28.0660000000;
  t[381][1] =   18.0910000000;
  t[381][2] =   25.1210000000;
  t[382][0] =   28.8310000000;
  t[382][1] =   15.6090000000;
  t[382][2] =   24.0420000000;
  t[383][0] =   26.4430000000;
  t[383][1] =   14.4060000000;
  t[383][2] =   21.0620000000;
  t[384][0] =   26.7060000000;
  t[384][1] =   14.3530000000;
  t[384][2] =   22.5250000000;
  t[385][0] =   25.6320000000;
  t[385][1] =   13.6130000000;
  t[385][2] =   23.3760000000;
  t[386][0] =   25.5040000000;
  t[386][1] =   12.1710000000;
  t[386][2] =   23.0590000000;
  t[387][0] =   26.8030000000;
  t[387][1] =   15.7080000000;
  t[387][2] =   22.9740000000;
  t[388][0] =   26.5460000000;
  t[388][1] =   11.3350000000;
  t[388][2] =   23.1890000000;
  t[389][0] =   25.3340000000;
  t[389][1] =   14.0570000000;
  t[389][2] =   20.5780000000;
  t[390][0] =   24.4120000000;
  t[390][1] =   11.6950000000;
  t[390][2] =   22.6700000000;
  t[391][0] =   27.8740000000;
  t[391][1] =   13.5000000000;
  t[391][2] =   18.3030000000;
  t[392][0] =   27.3930000000;
  t[392][1] =   14.7940000000;
  t[392][2] =   18.8220000000;
  t[393][0] =   27.4380000000;
  t[393][1] =   14.8290000000;
  t[393][2] =   20.2450000000;
  t[394][0] =   28.4930000000;
  t[394][1] =   12.6920000000;
  t[394][2] =   19.0400000000;

  source = s;
  target = t;
  geom::procrustes(&source, &target, &sourceCentroid, &targetCentroid,
                   &r, &scale);
  rotate(source, r, &transformed);
  std::cout << sourceCentroid << std::endl;
  std::cout << targetCentroid << std::endl;
  std::cout << r << std::endl;
  double det =
    r(0, 0) * (r(1, 1) * r(2, 2) - r(2, 1) * r(1, 2))
    + r(0, 1) * (r(1, 2) * r(2, 0) - r(2, 2) * r(1, 0))
    + r(0, 2) * (r(1, 0) * r(2, 1) - r(2, 0) * r(1, 1));
  std::cout << "det = " << det << std::endl;
  assert(std::abs(det - 1) < 1e-5);
  /*
  for (std::size_t i = 1; i != transformed.size(); ++i) {
     assert(areEqual(euclideanDistance(source[i],source[i-1]),
                     euclideanDistance(s[i],s[i-1])));
  }
  */
}
