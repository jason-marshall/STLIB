#include <cstdlib>
#include <cassert>
#include <iostream>

#include "stlib/geom/kernel/SmallestEnclosingBall.h"


USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

template<typename _Float>
void
testComputeSmallestEnclosingBallSquaredRadius()
{
  typedef std::array<std::array<_Float, 2>, 3> Triangle2;
  typedef std::pair<Triangle2, _Float> Pair;
  _Float const Eps = std::numeric_limits<_Float>::epsilon();
  _Float const Tolerance = 10 * Eps;
  std::array<std::array<std::size_t, 3>, 6> const vi =
    {{{{0, 1, 2}},
      {{1, 2, 0}},
      {{2, 0, 1}},
      {{0, 2, 1}},
      {{2, 1, 0}},
      {{1, 0, 2}}}};
  Pair Data[] = {
    // Singular. Three vertices are the same.
    {Triangle2{{{{0, 0}}, {{0, 0}}, {{0, 0}}}}, _Float(0)},
    // Singular. Two vertices are the same.
    {Triangle2{{{{0, 0}}, {{0, 0}}, {{1, 0}}}}, _Float(0.25)},
    // Singular. Collinear.
    {Triangle2{{{{0, 0}}, {{0.5, 0}}, {{1, 0}}}}, _Float(0.25)},
    // Nearly singular. Three vertices are nearly the same.
    {Triangle2{{{{0, 0}}, {{Eps, 0}}, {{0, Eps}}}}, _Float(0.5) * Eps * Eps},
    // Nearly singular. Two vertices are nearly the same.
    {Triangle2{{{{0, 0}}, {{Eps, 0}}, {{1, 0}}}}, _Float(0.25)},
    // Nearly singular. Two vertices are nearly the same.
    {Triangle2{{{{0, 0}}, {{0, Eps}}, {{1, 0}}}}, _Float(0.25)},
    // Nearly singular. Nearly collinear.
    {Triangle2{{{{0, 0}}, {{0.5, Eps}}, {{1, 0}}}}, _Float(0.25)},
    // Transition case. Circumscribed ball and midpoint ball are the same.
    {Triangle2{{{{0, 0}}, {{1, 0}}, {{0, 1}}}}, _Float(0.5)},
    // Transition case. Circumscribed ball and midpoint ball are the same.
    // 1 * 2 * sqrt(3) / (4 * 0.5 * sqrt(3)) = 1
    {Triangle2{{{{0, 0}}, {{1, 0}}, {{0, _Float(std::sqrt(3.))}}}}, _Float(1)},
    // Circumscribed ball is the SEB.
    // 1 * 1 * 1 / (4 * 0.25 * sqrt(3)) = 1 / sqrt(3)
    {Triangle2{{{{0, 0}}, {{1, 0}}, {{0.5, _Float(0.5 * std::sqrt(3.))}}}},
     _Float(1) / 3},
    // Circumscribed ball is the SEB.
    // 1 * (sqrt(5)/2) * (sqrt(5)/2) / (4 * 1) = 5/8
    {Triangle2{{{{0, 0}}, {{1, 0}}, {{0.5, 1}}}}, _Float((5. / 8) * (5. / 8))}
  };

  // 2-D.
  {
    std::size_t const Dimension = 2;
    typedef std::array<_Float, Dimension> Point;
    for (std::size_t i = 0; i != sizeof(Data) / sizeof(Pair); ++i) {
      Triangle2 const& t = Data[i].first;
      std::array<Point, 3> const v = {{{{t[0][0], t[0][1]}},
                                       {{t[1][0], t[1][1]}},
                                       {{t[2][0], t[2][1]}}}};
      for (std::size_t j = 0; j != vi.size(); ++j) {
        _Float const r = geom::computeSmallestEnclosingBallSquaredRadius
          (v[vi[j][0]], v[vi[j][1]], v[vi[j][2]]);
        assert(std::abs(r - Data[i].second) < Tolerance);
      }
    }
  }

  // 3-D.
  {
    std::size_t const Dimension = 3;
    typedef std::array<_Float, Dimension> Point;
    for (std::size_t i = 0; i != sizeof(Data) / sizeof(Pair); ++i) {
      Triangle2 const& t = Data[i].first;
      std::array<Point, 3> const v = {{{{t[0][0], t[0][1], 0}},
                                       {{t[1][0], t[1][1], 0}},
                                       {{t[2][0], t[2][1], 0}}}};
      for (std::size_t j = 0; j != vi.size(); ++j) {
        _Float const r = geom::computeSmallestEnclosingBallSquaredRadius
          (v[vi[j][0]], v[vi[j][1]], v[vi[j][2]]);
        assert(std::abs(r - Data[i].second) < Tolerance);
      }
    }
  }
}

int
main()
{
  testComputeSmallestEnclosingBallSquaredRadius<float>();
  testComputeSmallestEnclosingBallSquaredRadius<double>();

  using std::cout;
  using std::endl;

  const std::size_t Dimension = 5;
  const std::size_t n = 100000;
  geom::SmallestEnclosingBall<Dimension> mb;

  typedef geom::SmallestEnclosingBall<Dimension>::Point Point;

  // generate random points and check them in
  // ----------------------------------------
  Point p;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < Dimension; ++j) {
      p[j] = rand();
    }
    mb.checkIn(p);
  }

  // construct ball
  // --------------
  cout << "Constructing miniball...";
  cout.flush();
  mb.build();
  cout << "done." << endl << endl;

  // output center and squared radius
  // --------------------------------
  cout << "Center:         " << mb.center() << endl;
  cout << "Squared radius: " << mb.squaredRadius() << endl << endl;

  // output number of support points
  // -------------------------------
  cout << mb.numSupportPoints() << " support points: " << endl << endl;

  // output support points
  // ---------------------
  geom::SmallestEnclosingBall<Dimension>::const_iterator it;
  for (it = mb.supportPointsBegin(); it != mb.supportPointsEnd(); ++it) {
    cout << *it << endl;
  }
  cout << endl;

  // output accuracy
  // ---------------
  double slack;
  cout << "Relative accuracy: " << mb.accuracy(&slack) << endl;
  cout << "Optimality slack:  " << slack << endl;

  // check validity (even if this fails, the ball may be acceptable,
  // see the interface of class SmallestEnclosingBall)
  // ------------------------------------
  cout << "Validity: " << (mb.isValid() ? "ok" : "possibly invalid") << endl;

  return 0;
}



