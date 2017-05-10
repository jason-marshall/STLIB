// -*- C++ -*-

#include "stlib/geom/spatialIndexing/SpecialEuclideanCodeUniformGrid.h"
#include "stlib/geom/kernel/Point.h"
#include "stlib/numerical/equality.h"

using namespace stlib;

template<std::size_t _D, typename _Key = std::size_t>
class SpecialEuclideanCodeUniformGridTester;

template<typename _Key>
class SpecialEuclideanCodeUniformGridTester<3, _Key> :
  public geom::SpecialEuclideanCodeUniformGrid<3, _Key>
{
public:
  typedef geom::SpecialEuclideanCodeUniformGrid<3, _Key> Base;
  typedef typename Base::Key Key;
  typedef typename Base::BBox BBox;
  typedef typename Base::Point Point;

  using Base::_lower;
  using Base::_spacing;
  using Base::_bitsPerTranslation;
  using Base::_bitsPerRotation;

  SpecialEuclideanCodeUniformGridTester
  (const BBox& domain, const double spacing,
   const std::size_t bitsPerRotationCoordinate) :
    Base(domain, spacing, bitsPerRotationCoordinate)
  {
  }
};


typedef SpecialEuclideanCodeUniformGridTester<3> SE;
typedef std::array<std::size_t, 3> SizeList;
typedef SE::Quaternion Quaternion;


// Euclidean distance for unit quaternions.
double
euclideanDistance(const Quaternion& a, const Quaternion& b)
{
  return std::min(abs(a - b), abs(a + b));
}


int
main()
{
  using numerical::areEqual;

  typedef SE::Key Key;
  typedef SE::Point Point;
  typedef SE::BBox BBox;

  // Test set.
  const std::array<BBox, 2> domains = {{
      {{{0, 0, 0}}, {{1, 1, 1}}},
      {{{ -1, -2, -3}}, {{2, 3, 5}}}
    }
  };
  const std::array<double, 5> spacings = {{2, 1, 0.5, 0.3, 0.2}};
  const std::array<std::size_t, 3> bitsPerRot = {{4, 5, 6}};
  std::array<Point, 3> x = {{
      {{1, 0, 0}},
      {{ -1, 0, 0}},
      {{1, 2, 3}}
    }
  };
  std::array<Point, 3> y = {{
      {{0, 1, 0}},
      {{0, -1, 0}},
      {{0, 0, 0}}
    }
  };
  std::array<Point, 3> t = {{
      {{0, 0, 0}},
      {{1, 1, 1}},
      {{0.1, 0.2, 0.3}}
    }
  };
  stlib::ext::normalize(&x[2]);
  geom::computeAnOrthogonalVector(x[2], &y[2]);
  stlib::ext::normalize(&y[2]);
  std::array<Quaternion, 5> q;
  q[0] = Quaternion(1, 0, 0, 0);
  q[1] = Quaternion(-1, 0, 0, 0);
  q[2] = Quaternion(0, 1, 0, 0);
  q[3] = Quaternion(0.6, 0.8, 0, 0);
  q[4] = Quaternion(-0.6, 0.8, 0, 0);

  // These tests require 64-bit executables.
#if (__GNUC__ && (__x86_64__ || __ppc64__)) || _WIN64
  Point x2, y2, t2;
  Quaternion q2;
  for (std::size_t a = 0; a != domains.size(); ++a) {
    for (std::size_t b = 0; b != spacings.size(); ++b) {
      for (std::size_t c = 0; c != bitsPerRot.size(); ++c) {
        SE sec(domains[a], spacings[b], bitsPerRot[c]);
        for (std::size_t i = 0; i != sec._lower.size(); ++i) {
          assert(sec._lower[i] <= domains[a].lower[i]);
        }
        assert(sec._spacing == spacings[b]);
        assert(sec._bitsPerRotation == bitsPerRot[c]);

        // 1.1 is the fudge factor. The std::sqrt(3.) is for the diagonal
        // of a cell.
        const double Dx = 1.1 * std::sqrt(3.) * 2. /
                          ((1 << sec._bitsPerRotation) - 1);
        const double Dt = 1.1 * std::sqrt(3.) * 0.5 * sec._spacing;

        for (std::size_t i = 0; i != x.size(); ++i) {
          const Key key = sec.encode(x[i], y[i], t[i]);
          sec.decode(key, &x2, &y2, &t2);
          assert(stlib::ext::euclideanDistance(x[i], x2) < Dx);
          assert(stlib::ext::euclideanDistance(y[i], y2) < Dx);
          assert(stlib::ext::euclideanDistance(t[i], t2) < Dt);
#if 0
          std::cout << "Encoded: " << x[i] << ", " << y[i] << ", "
                    << t[i] << '\n'
                    << "Decoded: " << x2 << ", " << y2 << ", "
                    << t2 << '\n';
#endif
        }
        for (std::size_t i = 0; i != q.size(); ++i) {
          const Key key = sec.encode(q[i], t[0]);
          sec.decode(key, &q2, &t2);
          // I don't really know the appropriate threshold. I'll just use
          // the one for rotated axes.
          assert(euclideanDistance(q[i], q2) < Dx);
          assert(stlib::ext::euclideanDistance(t[0], t2) < Dt);
#if 0
          std::cout << "Encoded: " << q[i] << '\n'
                    << "Decoded: " << q2 << '\n';
#endif
        }
      }
    }
  }
#endif

  const BBox Domain = {{{0, 0, 0}}, {{1, 1, 1}}};
  const std::size_t BitsPerRotation = 5;
  {
    // Extents = 1.
    const double Spacing = 2;
    SE sec(Domain, Spacing, BitsPerRotation);
    assert(areEqual(sec._lower, Point{{-0.5, -0.5, -0.5}}));
    assert(sec._bitsPerTranslation == (SizeList{{0, 0, 0}}));
  }
  // These tests require 64-bit executables.
#if (__GNUC__ && (__x86_64__ || __ppc64__)) || _WIN64
  {
    // Extents = 2.
    const double Spacing = 1;
    SE sec(Domain, Spacing, BitsPerRotation);
    assert(areEqual(sec._lower, Point{{-0.5, -0.5, -0.5}}));
    assert(sec._bitsPerTranslation == (SizeList{{1, 1, 1}}));
  }
  {
    // Extents = 3.
    const double Spacing = 0.5;
    SE sec(Domain, Spacing, BitsPerRotation);
    assert(areEqual(sec._lower, Point{{-0.5, -0.5, -0.5}}));
    assert(sec._bitsPerTranslation == (SizeList{{2, 2, 2}}));
  }
  {
    // Extents = 4.
    const double Spacing = 0.3;
    SE sec(Domain, Spacing, BitsPerRotation);
    assert(areEqual(sec._lower, Point{{-0.1, -0.1, -0.1}}));
    assert(sec._bitsPerTranslation == (SizeList{{2, 2, 2}}));
  }
  {
    // Extents = 5.
    const double Spacing = 0.2;
    SE sec(Domain, Spacing, BitsPerRotation);
    assert(areEqual(sec._lower, Point{{-0.3, -0.3, -0.3}}));
    assert(sec._bitsPerTranslation == (SizeList{{3, 3, 3}}));
  }
#endif

  return 0;
}
