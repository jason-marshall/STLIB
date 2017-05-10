// -*- C++ -*-

#include "stlib/geom/spatialIndexing/SpecialEuclideanCode.h"
#include "stlib/geom/kernel/Point.h"
#include "stlib/numerical/equality.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/ext/vector.h"

using namespace stlib;

template<std::size_t _D, std::size_t _SubdivisionLevels,
         typename _Key = std::size_t>
class SpecialEuclideanCodeTester;

template<std::size_t _SubdivisionLevels, typename _Key>
class SpecialEuclideanCodeTester<3, _SubdivisionLevels, _Key> :
  public geom::SpecialEuclideanCode<3, _SubdivisionLevels, _Key>
{
public:
  typedef geom::SpecialEuclideanCode<3, _SubdivisionLevels, _Key> Base;
  typedef typename Base::Key Key;
  typedef typename Base::BBox BBox;
  typedef typename Base::Point Point;

  using Base::_lower;
  using Base::_spacing;
  using Base::_bitsPerTranslation;
  using Base::_centroids;
  using Base::_adjacencies;

  SpecialEuclideanCodeTester(const BBox& domain, const double spacing) :
    Base(domain, spacing)
  {
  }

  using Base::reportIncident;
  using Base::reportNeighbors;
};




// Euclidean distance for unit quaternions.
double
euclideanDistance(const boost::math::quaternion<double>& a,
                  const boost::math::quaternion<double>& b)
{
  return std::min(abs(a - b), abs(a + b));
}


template<std::size_t _SubdivisionLevels>
void
test()
{
  using numerical::areEqual;

  typedef SpecialEuclideanCodeTester<3, _SubdivisionLevels> SE;
  typedef typename SE::Quaternion Quaternion;
  typedef typename SE::Key Key;
  typedef typename SE::Point Point;
  typedef typename SE::BBox BBox;

  // Make the random number generator.
  typedef numerical::ContinuousUniformGeneratorOpen<> Continuous;
  typedef Continuous::DiscreteUniformGenerator Discrete;
  Discrete discrete;
  Continuous random(&discrete);

  // Test set.
  const std::array<BBox, 2> domains = {{
      {{{0, 0, 0}}, {{1, 1, 1}}},
      {{{ -1, -2, -3}}, {{2, 3, 5}}}
    }
  };
  const std::array<double, 5> spacings = {{2, 1, 0.5, 0.3, 0.2}};
  // Randomly choose rotations and translations.
  const std::size_t Size = 100;
  std::vector<Point> x(Size), y(Size), t(Size);
  std::vector<Quaternion> q(Size);
  for (std::size_t i = 0; i != x.size(); ++i) {
    for (std::size_t j = 0; j != 3; ++j) {
      x[i][j] = random();
      t[i][j] = random();
    }
    stlib::ext::normalize(&x[i]);
    geom::computeAnOrthogonalVector(x[i], &y[i]);
    stlib::ext::normalize(&y[i]);
    q[i] = Quaternion(random(), random(), random(), random());
    q[i] /= abs(q[i]);
  }
  // A couple of special cases.
  x[0][0] = 1;
  x[0][1] = 0;
  x[0][2] = 0;
  y[0][0] = 0;
  y[0][1] = 1;
  y[0][2] = 0;
  x[1][0] = -1;
  x[1][1] = 0;
  x[1][2] = 0;
  y[1][0] = 0;
  y[1][1] = -1;
  y[1][2] = 0;
  t[0][0] = 0;
  t[0][1] = 0;
  t[0][2] = 0;
  t[1][0] = 1;
  t[1][1] = 1;
  t[1][2] = 1;
  q[0] = Quaternion(1, 0, 0, 0);
  q[1] = Quaternion(-1, 0, 0, 0);
  q[2] = Quaternion(0, 1, 0, 0);

  Point x2, y2, t2;
  Quaternion q2;
  for (std::size_t a = 0; a != domains.size(); ++a) {
    for (std::size_t b = 0; b != spacings.size(); ++b) {
      SE sec(domains[a], spacings[b]);
      for (std::size_t i = 0; i != sec._lower.size(); ++i) {
        assert(sec._lower[i] <= domains[a].lower[i]);
      }
      assert(sec._spacing == spacings[b]);

      // The length a quarter circle, subdivided. The leading factor of
      // 2 is the fudge factor. (The spherical triangles are not the same
      // size.)
      const double Dx = 2 * 0.25 * 2 * 3.14 / (1 << _SubdivisionLevels);
      const double Dt = 1.1 * std::sqrt(3.) * 0.5 * sec._spacing;

      for (std::size_t i = 0; i != x.size(); ++i) {
        const Key key = sec.encode(x[i], y[i], t[i]);
        sec.decode(key, &x2, &y2, &t2);
#if 0
        std::cerr << "Dx = " << Dx << '\n'
                  << "Encoded: " << x[i] << ", " << y[i] << ", "
                  << t[i] << '\n'
                  << "Decoded: " << x2 << ", " << y2 << ", "
                  << t2 << '\n';
#endif
        //std::cerr << euclideanDistance(x[i], x2) << '\n';
        assert(stlib::ext::euclideanDistance(x[i], x2) < Dx);
        assert(stlib::ext::euclideanDistance(y[i], y2) < Dx);
        assert(stlib::ext::euclideanDistance(t[i], t2) < Dt);
      }
      for (std::size_t i = 0; i != q.size(); ++i) {
        const Key key = sec.encode(q[i], t[0]);
        sec.decode(key, &q2, &t2);
        // I don't really know the appropriate threshold. I'll just use
        // the one for rotated axes.
#if 0
        std::cerr << "Dx = " << Dx << '\n'
                  << "Encoded: " << q[i] << '\n'
                  << "Decoded: " << q2 << '\n';
#endif
        assert(euclideanDistance(q[i], q2) < Dx);
        assert(stlib::ext::euclideanDistance(t[0], t2) < Dt);
      }
    }
  }
}


template<std::size_t _SubdivisionLevels>
void
testIncident()
{
  typedef SpecialEuclideanCodeTester<3, _SubdivisionLevels> SE;
  typedef typename SE::BBox BBox;
  typedef typename SE::Key Key;
  const BBox Domain = {{{0, 0, 0}}, {{1, 1, 1}}};
  const double Spacing = 1.;
  SE sec(Domain, Spacing);

  std::set<Key> incident;
  for (std::size_t i = 0; i != sec._adjacencies.size(); ++i) {
    for (std::size_t j = 0; j != 3; ++j) {
      incident.clear();
      sec.reportIncident(i, j, &incident);
      assert(incident.size() == 4 || incident.size() == 6);
    }
  }

  std::vector<Key> neighbors;
  for (std::size_t i = 0; i != sec._adjacencies.size(); ++i) {
    sec.reportNeighbors(i, &neighbors);
    // Vertices have either 4 or six incident triangles.
    // The only possibilities for the vertices or a trinagle are (4, 6, 6)
    // or (6, 6, 6).
    assert(neighbors.size() == 11 || neighbors.size() == 13);
  }
}

template<std::size_t _SubdivisionLevels>
void
testTranslationNeighbors()
{
  typedef SpecialEuclideanCodeTester<3, _SubdivisionLevels> SE;
  typedef typename SE::BBox BBox;
  typedef typename SE::Key Key;
  typedef std::array<std::size_t, 3> IndexList;
  const BBox Domain = {{{0, 0, 0}}, {{1, 1, 1}}};
  const double Spacing = 0.1;
  SE sec(Domain, Spacing);

  std::vector<Key> neighbors;
  {
    const IndexList Cell = {{0, 0, 0}};
    sec.reportNeighbors(Cell, &neighbors);
    assert(neighbors.size() == 8);
  }

  {
    const IndexList Cell = {{0, 0, 1}};
    sec.reportNeighbors(Cell, &neighbors);
    assert(neighbors.size() == 2 * 2 * 3);
  }
  {
    const IndexList Cell = {{0, 1, 0}};
    sec.reportNeighbors(Cell, &neighbors);
    assert(neighbors.size() == 2 * 2 * 3);
  }
  {
    const IndexList Cell = {{1, 0, 0}};
    sec.reportNeighbors(Cell, &neighbors);
    assert(neighbors.size() == 2 * 2 * 3);
  }

  {
    const IndexList Cell = {{0, 1, 1}};
    sec.reportNeighbors(Cell, &neighbors);
    assert(neighbors.size() == 2 * 3 * 3);
  }
  {
    const IndexList Cell = {{1, 0, 1}};
    sec.reportNeighbors(Cell, &neighbors);
    assert(neighbors.size() == 2 * 3 * 3);
  }
  {
    const IndexList Cell = {{1, 1, 0}};
    sec.reportNeighbors(Cell, &neighbors);
    assert(neighbors.size() == 2 * 3 * 3);
  }

  {
    const IndexList Cell = {{1, 1, 1}};
    sec.reportNeighbors(Cell, &neighbors);
    assert(neighbors.size() == 27);
  }

  {
    const IndexList Cell = {
      {
        std::size_t((1 << sec._bitsPerTranslation[0]) - 1),
        std::size_t((1 << sec._bitsPerTranslation[1]) - 1),
        std::size_t((1 << sec._bitsPerTranslation[2]) - 1)
      }
    };
    sec.reportNeighbors(Cell, &neighbors);
    assert(neighbors.size() == 8);
  }
}


template<std::size_t _SubdivisionLevels>
void
testNeighbors()
{
  typedef SpecialEuclideanCodeTester<3, _SubdivisionLevels> SE;
  typedef typename SE::Point Point;
  typedef typename SE::BBox BBox;
  typedef typename SE::Key Key;
  const BBox Domain = {{{0, 0, 0}}, {{1, 1, 1}}};
  const double Spacing = 0.251;
  SE sec(Domain, Spacing);

  std::vector<Key> neighbors;
  {
    const Key key = sec.encode(Point{{1., 0., 0.}},
                               Point{{0., 1., 0.}},
                               Point{{0., 0., 0.}});
    sec.neighbors(key, &neighbors);
    assert(neighbors.size() == 11 * 11 * 8);
  }
  {
    const Key key = sec.encode(Point{{1., 0., 0.}},
                               Point{{0., 1., 0.}},
                               Point{{1., 1., 1.}});
    sec.neighbors(key, &neighbors);
    assert(neighbors.size() == 11 * 11 * 8);
  }

  {
    const Key key = sec.encode(Point{{1., 0., 0.}},
                               Point{{0., 1., 0.}},
                               Point{{0.5, 0., 0.}});
    sec.neighbors(key, &neighbors);
    assert(neighbors.size() == 11 * 11 * (3 * 2 * 2));
  }
  {
    const Key key = sec.encode(Point{{1., 0., 0.}},
                               Point{{0., 1., 0.}},
                               Point{{0.5, 0.5, 0.}});
    neighbors.clear();
    sec.neighbors(key, &neighbors);
    assert(neighbors.size() == 11 * 11 * (3 * 3 * 2));
  }
  {
    const Key key = sec.encode(Point{{1., 0., 0.}},
                               Point{{0., 1., 0.}},
                               Point{{0.5, 0.5, 0.5}});
    neighbors.clear();
    sec.neighbors(key, &neighbors);
    assert(neighbors.size() == 11 * 11 * (3 * 3 * 3));
  }

  {
    const double X = 1. / std::sqrt(3.);
    const double Y = 1. / std::sqrt(2.);
    const Key key = sec.encode(Point{{X, X, X}},
                               Point{{-Y, Y, 0.}},
                               Point{{0.5, 0.5, 0.5}});
    neighbors.clear();
    sec.neighbors(key, &neighbors);
    assert(neighbors.size() == 13 * 11 * (3 * 3 * 3) ||
           neighbors.size() == 13 * 13 * (3 * 3 * 3));
  }
}


int
main()
{
  test<1>();
  test<2>();
  // These tests require 64-bit executables.
#if (__GNUC__ && (__x86_64__ || __ppc64__)) || _WIN64
  test<3>();
  test<4>();
  test<5>();
#endif
  testIncident<1>();
  testIncident<2>();
  testTranslationNeighbors<1>();
  testTranslationNeighbors<2>();
  testNeighbors<1>();
  testNeighbors<2>();

  using numerical::areEqual;

  {
    typedef SpecialEuclideanCodeTester<3, 1> SE;
    typedef SE::Point Point;
    typedef SE::BBox BBox;
    typedef std::array<std::size_t, 3> SizeList;

    const BBox Domain = {{{0, 0, 0}}, {{1, 1, 1}}};
    {
      // Extents = 1.
      const double Spacing = 2;
      SE sec(Domain, Spacing);
      assert(areEqual(sec._lower, Point{{-0.5, -0.5, -0.5}}));
      assert(sec._bitsPerTranslation == (SizeList{{0, 0, 0}}));
    }
    {
      // Extents = 2.
      const double Spacing = 1;
      SE sec(Domain, Spacing);
      assert(areEqual(sec._lower, Point{{-0.5, -0.5, -0.5}}));
      assert(sec._bitsPerTranslation == (SizeList{{1, 1, 1}}));
    }
    {
      // Extents = 3.
      const double Spacing = 0.5;
      SE sec(Domain, Spacing);
      assert(areEqual(sec._lower, Point{{-0.5, -0.5, -0.5}}));
      assert(sec._bitsPerTranslation == (SizeList{{2, 2, 2}}));
    }
    {
      // Extents = 4.
      const double Spacing = 0.3;
      SE sec(Domain, Spacing);
      assert(areEqual(sec._lower, Point{{-0.1, -0.1, -0.1}}));
      assert(sec._bitsPerTranslation == (SizeList{{2, 2, 2}}));
    }
    {
      // Extents = 5.
      const double Spacing = 0.2;
      SE sec(Domain, Spacing);
      assert(areEqual(sec._lower, Point{{-0.3, -0.3, -0.3}}));
      assert(sec._bitsPerTranslation == (SizeList{{3, 3, 3}}));
    }
  }
  // These tests require 64-bit executables.
#if (__GNUC__ && (__x86_64__ || __ppc64__)) || _WIN64
  // Amos' test case.
  {
    const std::size_t SubdivisionLevels = 2;
    typedef geom::SpecialEuclideanCode<3, SubdivisionLevels> SE;
    typedef SE::Quaternion Quaternion;
    typedef SE::Key Key;
    typedef SE::Point Point;
    typedef SE::BBox BBox;

    const BBox Domain = {{{ -16.7445, 22.5165, -24.5775}},
      {{83.0295, 99.6975, 39.039}}
    };
    {
      // Extents = 1.
      const double Spacing = 1;
      SE sec(Domain, Spacing);
      // The length of a quarter circle, subdivided.
      const double Dx = 0.25 * 2 * 3.14 / (1 << SubdivisionLevels);
      const double Dt = 1.1 * std::sqrt(3.) * 0.5 * Spacing;

      Quaternion q1 = Quaternion(0.456504, 0.178287, -0.365171, 0.791497);
      q1 /= abs(q1);
      Quaternion q2;
      Point t1 = {{22.267, 42.5099, 16.7199}};
      Point t2;

      const Key key = sec.encode(q1, t1);
      sec.decode(key, &q2, &t2);
#if 0
      std::cerr << "Dx = " << Dx << '\n'
                << "distance = " << euclideanDistance(q1, q2) << '\n'
                << "Encoded: " << q1 << '\n'
                << "Decoded: " << q2 << '\n';
#endif
      assert(euclideanDistance(q1, q2) < Dx);
      assert(stlib::ext::euclideanDistance(t1, t2) < Dt);
    }
  }
  {
    const std::size_t SubdivisionLevels = 4;
    typedef geom::SpecialEuclideanCode<3, SubdivisionLevels> SE;
    typedef SE::Quaternion Quaternion;
    typedef SE::Key Key;
    typedef SE::Point Point;
    typedef SE::BBox BBox;

    const BBox Domain = {{{ -16.7445, 22.5165, -24.5775}},
      {{83.0295, 99.6975, 39.039}}
    };
    {
      // Extents = 1.
      const double Spacing = 1;
      SE sec(Domain, Spacing);
      // The length of a quarter circle, subdivided.
      const double Dx = 0.25 * 2 * 3.14 / (1 << SubdivisionLevels);
      const double Dt = 1.1 * std::sqrt(3.) * 0.5 * Spacing;

      Quaternion q1 = Quaternion(0.456504, 0.178287, -0.365171, 0.791497);
      q1 /= abs(q1);
      Quaternion q2;
      Point t1 = {{22.267, 42.5099, 16.7199}};
      Point t2;

      const Key key = sec.encode(q1, t1);
      sec.decode(key, &q2, &t2);
#if 0
      std::cerr << "Dx = " << Dx << '\n'
                << "distance = " << euclideanDistance(q1, q2) << '\n'
                << "Encoded: " << q1 << '\n'
                << "Decoded: " << q2 << '\n';
#endif
      assert(euclideanDistance(q1, q2) < Dx);
      assert(stlib::ext::euclideanDistance(t1, t2) < Dt);
    }
  }
#endif

  return 0;
}
