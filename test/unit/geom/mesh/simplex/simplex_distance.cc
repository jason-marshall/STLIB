// -*- C++ -*-

#include "stlib/geom/mesh/simplex/simplex_distance.h"

#include "stlib/geom/mesh/simplex/SimplexCondNum.h"
#include "stlib/numerical/random/uniform.h"

#include <iostream>
#include <vector>

#include <cassert>

using namespace stlib;

// Use a set of random points and the triangle inequality to check that the
// signed distance is a continuous function.
template<std::size_t N>
void
signedDistanceContinuous()
{
  typedef std::array<double, N> Point;
  typedef numerical::ContinuousUniformGeneratorOpen<> ContinuousUniform;
  typedef ContinuousUniform::DiscreteUniformGenerator DiscreteUniformGenerator;

  const double Eps = std::sqrt(std::numeric_limits<double>::epsilon());

  // Make a vector of random points.
  DiscreteUniformGenerator generator;
  ContinuousUniform uniform(&generator);
  std::vector<Point> points(100);
  for (std::size_t i = 0; i != points.size(); ++i) {
    for (std::size_t j = 0; j != N; ++j) {
      points[i][j] = uniform();
    }
  }

  geom::SimplexCondNum<N, double> condNum;
  std::array < Point, N + 1 > simplex;
  geom::SimplexDistance<N, N, double> simplexDistance;
  // Try a number of simplices with randomly selected vertices.
  const std::size_t Attempts = 100;
  std::size_t testCount = 0;
  for (std::size_t i = 0; i != Attempts; ++i) {
    for (std::size_t j = 0; j != simplex.size(); ++j) {
      simplex[j] = points[generator() % points.size()];
    }
    // Skip the simplices with very poor quality.
    condNum.setFunction(simplex);
    if (condNum.computeContent() <= Eps || condNum() < 0.01) {
      continue;
    }
    ++testCount;
    simplexDistance.initialize(simplex);
    // Test the triangle inequality for each pair of points.
    for (std::size_t j = 0; j != points.size(); ++j) {
      for (std::size_t k = j + 1; k != points.size(); ++k) {
        double diff2 = simplexDistance(points[j]) -
                       simplexDistance(points[k]);
        diff2 = diff2 * diff2;
        assert(diff2 <= (1 + Eps) *
               stlib::ext::squaredDistance(points[j], points[k]));
      }
    }
  }
  assert(4 * testCount >= Attempts);
}

int
main()
{
  using namespace geom;

  const double eps = 10 * std::numeric_limits<double>::epsilon();

  //---------------------------------------------------------------------------
  // Inside tests.
  //---------------------------------------------------------------------------
  //
  // 1-D, is_in
  //
  {
    typedef std::array<double, 1> Pt;
    typedef std::array < Pt, 1 + 1 > Simplex;

    Simplex s = {{Pt{{0.}}, Pt{{1.}}}};
    assert(isIn(s, Pt{{0.}}));
    assert(isIn(s, Pt{{0.5}}));
    assert(isIn(s, Pt{{1.}}));
    assert(! isIn(s, Pt{{-1.}}));
    assert(! isIn(s, Pt{{2.}}));
  }

  //
  // 2-D, isIn
  //
  {
    typedef std::array<double, 2> Pt;
    typedef std::array < Pt, 2 + 1 > Simplex;

    Simplex s = {{Pt{{0., 0.}}, Pt{{1., 0.}}, Pt{{0., 1.}}}};
    // Negative distance.
    assert(isIn(s, Pt{{0.25, 0.25}}));
    // Zero distance, at the vertices.
    assert(isIn(s, Pt{{0., 0.}}));
    assert(isIn(s, Pt{{1., 0.}}));
    assert(isIn(s, Pt{{0., 1.}}));
    // Zero distance, on the edges.
    assert(isIn(s, Pt{{0.5, 0.}}));
    assert(isIn(s, Pt{{0.5, 0.5}}));
    assert(isIn(s, Pt{{0., 0.5}}));
    // Positive distance, outside one line.
    assert(! isIn(s, Pt{{1., 1.}}));
    assert(! isIn(s, Pt{{-1., 0.5}}));
    assert(! isIn(s, Pt{{0.5, -1.}}));
    // Positive distance, outside two lines.
    assert(! isIn(s, Pt{{-1., -1.}}));
    assert(! isIn(s, Pt{{2., -1.}}));
    assert(! isIn(s, Pt{{-1., 2.}}));
  }

  //
  // 3-D, isIn
  //
  {
    typedef std::array<double, 3> Pt;
    typedef std::array < Pt, 3 + 1 > Simplex;

    Simplex s = {{Pt{{0., 0., 0.}}, Pt{{1., 0., 0.}},
                 Pt{{0., 1., 0.}}, Pt{{0., 0., 1.}}}};
    // Negative distance.
    assert(isIn(s, Pt{{0.25, 0.25, 0.25}}));
    // Zero distance, at the vertices.
    assert(isIn(s, Pt{{0., 0., 0.}}));
    assert(isIn(s, Pt{{1., 0., 0.}}));
    assert(isIn(s, Pt{{0., 1., 0.}}));
    assert(isIn(s, Pt{{0., 0., 1.}}));
    // Zero distance, on the edges.
    assert(isIn(s, Pt{{0.5, 0., 0.}}));
    assert(isIn(s, Pt{{0., 0.5, 0.}}));
    assert(isIn(s, Pt{{0., 0., 0.5}}));
    assert(isIn(s, Pt{{0.5, 0.5, 0.}}));
    assert(isIn(s, Pt{{0.5, 0., 0.5}}));
    assert(isIn(s, Pt{{0., 0.5, 0.5}}));
    // Zero distance, on the faces.
    assert(isIn(s, Pt{{0.25, 0.25, 0.}}));
    assert(isIn(s, Pt{{0.25, 0., 0.25}}));
    assert(isIn(s, Pt{{0., 0.25, 0.25}}));
    assert(isIn(s, Pt{{0.25, 0.25, 0.5}}));
    // Positive distance from the faces.
    assert(! isIn(s, Pt{{0.25, 0.25, -1.}}));
    assert(! isIn(s, Pt{{0.25, -1., 0.25}}));
    assert(! isIn(s, Pt{{-1., 0.25, 0.25}}));
    assert(! isIn(s, Pt{{1., 1., 1.}}));
    // Positive distance from the edges.
    assert(! isIn(s, Pt{{0.5, -1., -1.}}));
    assert(! isIn(s, Pt{{-1., 0.5, -1.}}));
    assert(! isIn(s, Pt{{-1., -1., 0.5}}));
    assert(! isIn(s, Pt{{1., 1., -1.}}));
    assert(! isIn(s, Pt{{1., -1., 1.}}));
    assert(! isIn(s, Pt{{-1., 1., 1.}}));
    // Positive distance from the vertices.
    assert(! isIn(s, Pt{{-1., -1., -1.}}));
    assert(! isIn(s, Pt{{2., 0., 0.}}));
    assert(! isIn(s, Pt{{0., 2., 0.}}));
    assert(! isIn(s, Pt{{0., 0., 2.}}));
  }


  //---------------------------------------------------------------------------
  // Interior distance.
  //---------------------------------------------------------------------------

  //
  // 2-D, distance_interior
  //
  {
    typedef std::array<double, 2> Pt;
    typedef std::array < Pt, 2 + 1 > Simplex;

    Simplex s = {{Pt{{0., 0.}}, Pt{{1., 0.}}, Pt{{0., 1.}}}};
    // Negative distance.
    assert(computeDistanceInterior(s, Pt{{0.25, 0.25}}) < 0);
    // Zero distance, at the vertices.
    assert(std::abs(computeDistanceInterior(s, Pt{{0., 0.}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{1., 0.}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{0., 1.}})) < eps);
    // Zero distance, on the edges.
    assert(std::abs(computeDistanceInterior(s, Pt{{0.5, 0.}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{0.5, 0.5}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{0., 0.5}})) < eps);
  }

  //
  // 3-D, distance_interior
  //
  {
    typedef std::array<double, 3> Pt;
    typedef std::array < Pt, 3 + 1 > Simplex;

    Simplex s = {{Pt{{0., 0., 0.}}, Pt{{1., 0., 0.}}, Pt{{0., 1., 0.}},
                 Pt{{0., 0., 1.}}}};
    // Negative distance.
    assert(computeDistanceInterior(s, Pt{{0.25, 0.25, 0.25}}) < 0);
    // Zero distance, at the vertices.
    assert(std::abs(computeDistanceInterior(s, Pt{{0., 0., 0.}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{1., 0., 0.}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{0., 1., 0.}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{0., 0., 1.}})) < eps);
    // Zero distance, on the edges.
    assert(std::abs(computeDistanceInterior(s, Pt{{0.5, 0., 0.}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{0., 0.5, 0.}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{0., 0., 0.5}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{0.5, 0.5, 0.}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{0.5, 0., 0.5}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{0., 0.5, 0.5}})) < eps);
    // Zero distance, on the faces.
    assert(std::abs(computeDistanceInterior(s, Pt{{0.25, 0.25, 0.}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{0.25, 0., 0.25}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{0., 0.25, 0.25}})) < eps);
    assert(std::abs(computeDistanceInterior(s, Pt{{0.5, 0.25, 0.25}})) < eps);
  }

  //---------------------------------------------------------------------------
  // Distance.
  //---------------------------------------------------------------------------

  //
  // 1-simplex in 1-D, signed distance
  //
  {
    typedef std::array<double, 1> Pt;
    typedef std::array < Pt, 1 + 1 > Simplex;

    Simplex s = {{Pt{{0.}}, Pt{{1.}}}};

    assert(std::abs(computeDistance(s, Pt{{0.}}) - 0) < eps);
    assert(std::abs(computeDistance(s, Pt{{0.5}}) - -0.5) < eps);
    assert(std::abs(computeDistance(s, Pt{{1.}}) - 0) < eps);
    assert(std::abs(computeDistance(s, Pt{{-1.}}) - 1) < eps);
    assert(std::abs(computeDistance(s, Pt{{2.}}) - 1) < eps);
  }

  //
  // 1-simplex in 2-D, unsigned distance
  //
  {
    typedef std::array<double, 2> Pt;
    typedef std::array < Pt, 1 + 1 > Simplex;

    Simplex s = {{Pt{{0., 0.}}, Pt{{1., 0.}}}};

    assert(std::abs(computeDistance(s, Pt{{0., 0.}}) - 0) < eps);
    assert(std::abs(computeDistance(s, Pt{{0.5, 0.}}) - 0) < eps);
    assert(std::abs(computeDistance(s, Pt{{1., 0.}}) -  0) < eps);
    assert(std::abs(computeDistance(s, Pt{{-1., 0.}}) - 1) < eps);
    assert(std::abs(computeDistance(s, Pt{{2., 0.}}) - 1) < eps);

    assert(std::abs(computeDistance(s, Pt{{0., 1.}}) - 1) < eps);
    assert(std::abs(computeDistance(s, Pt{{0.5, 1.}}) - 1) < eps);
    assert(std::abs(computeDistance(s, Pt{{1., 1.}}) -  1) < eps);
    assert(std::abs(computeDistance(s, Pt{{-1., 1.}}) - std::sqrt(2.)) < eps);
    assert(std::abs(computeDistance(s, Pt{{2., 1.}}) - std::sqrt(2.)) < eps);
  }

  //
  // 1-simplex in 3-D, unsigned distance
  //
  {
    typedef std::array<double, 3> Pt;
    typedef std::array < Pt, 1 + 1 > S;

    S s = {{Pt{{0., 0., 0.}}, Pt{{1., 0., 0.}}}};

    assert(std::abs(computeDistance(s, Pt{{0., 0., 0.}}) - 0) < eps);
    assert(std::abs(computeDistance(s, Pt{{0.5, 0., 0.}}) - 0) < eps);
    assert(std::abs(computeDistance(s, Pt{{1., 0., 0.}}) -  0) < eps);
    assert(std::abs(computeDistance(s, Pt{{-1., 0., 0.}}) - 1) < eps);
    assert(std::abs(computeDistance(s, Pt{{2., 0., 0.}}) - 1) < eps);

    assert(std::abs(computeDistance(s, Pt{{0., 1., 0.}}) - 1) < eps);
    assert(std::abs(computeDistance(s, Pt{{0.5, 1., 0.}}) - 1) < eps);
    assert(std::abs(computeDistance(s, Pt{{1., 1., 0.}}) -  1) < eps);
    assert(std::abs(computeDistance(s, Pt{{-1., 1., 0.}}) - std::sqrt(2.)) < eps);
    assert(std::abs(computeDistance(s, Pt{{2., 1., 0.}}) - std::sqrt(2.)) < eps);
  }

  //
  // 2-simplex in 2-D, signed distance
  //
  {
    typedef std::array<double, 2> Pt;
    typedef std::array < Pt, 2 + 1 > Simplex;

    Simplex s = {{Pt{{0., 0.}}, Pt{{1., 0.}}, Pt{{0., 1.}}}};

    // Face 0.
    assert(std::abs(computeDistance(s, Pt{{0.5, -1.}}) - 1) < eps);
    // Face 1.
    assert(std::abs(computeDistance(s, Pt{{1., 1.}}) - std::sqrt(2.) / 2) < eps);
    // Face 2.
    assert(std::abs(computeDistance(s, Pt{{-1., 0.5}}) - 1) < eps);
    // Vertex 0.
    assert(std::abs(computeDistance(s, Pt{{-1., -1.}}) - std::sqrt(2.)) < eps);
    // Vertex 1.
    assert(std::abs(computeDistance(s, Pt{{2., -1.}}) - std::sqrt(2.)) < eps);
    // Vertex 2.
    assert(std::abs(computeDistance(s, Pt{{-1., 2.}}) - std::sqrt(2.)) < eps);
    // Inside.
    assert(std::abs(computeDistance(s, Pt{{0.25, 0.25}}) + 0.25) < eps);
  }

  //
  // 2-simplex in 3-D, unsigned distance
  //
  {
    typedef std::array<double, 3> Pt;
    typedef std::array < Pt, 2 + 1 > Simplex;

    Simplex s = {{Pt{{0., 0., 0.}}, Pt{{1., 0., 0.}}, Pt{{0., 1., 0.}}}};

    // Face 0.
    assert(std::abs(computeDistance(s, Pt{{0.5, -1., 0.}}) - 1) < eps);
    // Face 1.
    assert(std::abs(computeDistance(s, Pt{{1., 1., 0.}}) - std::sqrt(2.) / 2) < eps);
    // Face 2.
    assert(std::abs(computeDistance(s, Pt{{-1., 0.5, 0.}}) - 1) < eps);
    // Vertex 0.
    assert(std::abs(computeDistance(s, Pt{{-1., -1., 0.}}) - std::sqrt(2.)) < eps);
    // Vertex 1.
    assert(std::abs(computeDistance(s, Pt{{2., -1., 0.}}) - std::sqrt(2.)) < eps);
    // Vertex 2.
    assert(std::abs(computeDistance(s, Pt{{-1., 2., 0.}}) - std::sqrt(2.)) < eps);
    // Inside.
    assert(std::abs(computeDistance(s, Pt{{0.25, 0.25, 0.}}) - 0) < eps);

    // Face 0, above.
    assert(std::abs(computeDistance(s, Pt{{0.5, -1., 1.}}) - std::sqrt(2.)) < eps);
    // Face 1, above.
    assert(std::abs(computeDistance(s, Pt{{1., 1., 1.}}) - std::sqrt(3. / 2.)) < eps);
    // Face 2, above.
    assert(std::abs(computeDistance(s, Pt{{-1., 0.5, 1.}}) - std::sqrt(2.)) < eps);
    // Vertex 0, above.
    assert(std::abs(computeDistance(s, Pt{{-1., -1., 1.}}) - std::sqrt(3.)) < eps);
    // Vertex 1, above.
    assert(std::abs(computeDistance(s, Pt{{2., -1., 1.}}) - std::sqrt(3.)) < eps);
    // Vertex 2, above.
    assert(std::abs(computeDistance(s, Pt{{-1., 2., 1.}}) - std::sqrt(3.)) < eps);
    // Inside, above.
    assert(std::abs(computeDistance(s, Pt{{0.25, 0.25, 1.}}) - 1) < eps);
  }

  //
  // 3-D, signed distance
  //
  {
    typedef std::array<double, 3> Pt;
    typedef std::array < Pt, 3 + 1 > Simplex;

    Simplex s = {{Pt{{0., 0., 0.}}, Pt{{1., 0., 0.}}, Pt{{0., 1., 0.}},
                 Pt{{0., 0., 1.}}}};

    // Negative distance.
    assert(std::abs(computeDistance(s, Pt{{0.1, 0.2, 0.2}}) + 0.1) < eps);
    assert(std::abs(computeDistance(s, Pt{{0.2, 0.1, 0.2}}) + 0.1) < eps);
    assert(std::abs(computeDistance(s, Pt{{0.2, 0.2, 0.1}}) + 0.1) < eps);
    assert(std::abs(computeDistance(s, Pt{{0.25, 0.25, 0.25}}) +
                    std::sqrt(1. / 48.)) < eps);

    // Zero distance, at the vertices.
    assert(std::abs(computeDistance(s, Pt{{0., 0., 0.}})) < eps);
    assert(std::abs(computeDistance(s, Pt{{1., 0., 0.}})) < eps);
    assert(std::abs(computeDistance(s, Pt{{0., 1., 0.}})) < eps);
    assert(std::abs(computeDistance(s, Pt{{0., 0., 1.}})) < eps);
    // Zero distance, on the edges.
    assert(std::abs(computeDistance(s, Pt{{0.5, 0., 0.}})) < eps);
    assert(std::abs(computeDistance(s, Pt{{0., 0.5, 0.}})) < eps);
    assert(std::abs(computeDistance(s, Pt{{0., 0., 0.5}})) < eps);
    assert(std::abs(computeDistance(s, Pt{{0.5, 0.5, 0.}})) < eps);
    assert(std::abs(computeDistance(s, Pt{{0.5, 0., 0.5}})) < eps);
    assert(std::abs(computeDistance(s, Pt{{0., 0.5, 0.5}})) < eps);
    // Zero distance, on the faces.
    assert(std::abs(computeDistance(s, Pt{{0.25, 0.25, 0.}})) < eps);
    assert(std::abs(computeDistance(s, Pt{{0.25, 0., 0.25}})) < eps);
    assert(std::abs(computeDistance(s, Pt{{0., 0.25, 0.25}})) < eps);
    assert(std::abs(computeDistance(s, Pt{{0.5, 0.25, 0.25}})) < eps);

    // Positive distance from the faces.
    assert(std::abs(computeDistance(s, Pt{{0.25, 0.25, -1.}}) - 1) < eps);
    assert(std::abs(computeDistance(s, Pt{{0.25, -1., 0.25}}) - 1) < eps);
    assert(std::abs(computeDistance(s, Pt{{-1., 0.25, 0.25}}) - 1) < eps);
    assert(std::abs(computeDistance(s, Pt{{1., 1., 1.}}) - std::sqrt(4. / 3.))
           < eps);
    // Positive distance from the edges.
    assert(std::abs(computeDistance(s, Pt{{0.5, -1., -1.}}) - std::sqrt(2.))
           < eps);
    assert(std::abs(computeDistance(s, Pt{{-1., 0.5, -1.}}) - std::sqrt(2.))
           < eps);
    assert(std::abs(computeDistance(s, Pt{{-1., -1., 0.5}}) - std::sqrt(2.))
           < eps);
    assert(std::abs(computeDistance(s, Pt{{1., 1., -1.}}) - std::sqrt(3. / 2.))
           < eps);
    assert(std::abs(computeDistance(s, Pt{{1., -1., 1.}}) - std::sqrt(3. / 2.))
           < eps);
    assert(std::abs(computeDistance(s, Pt{{-1., 1., 1.}}) - std::sqrt(3. / 2.))
           < eps);
    // Positive distance from the vertices.
    assert(std::abs(computeDistance(s, Pt{{-1., -1., -1.}}) - std::sqrt(3.))
           < eps);
    assert(std::abs(computeDistance(s, Pt{{2., 0., 0.}}) - 1) < eps);
    assert(std::abs(computeDistance(s, Pt{{0., 2., 0.}}) - 1) < eps);
    assert(std::abs(computeDistance(s, Pt{{0., 0., 2.}}) - 1) < eps);
  }

  //---------------------------------------------------------------------------
  // Unsigned distance.
  //---------------------------------------------------------------------------

  //
  // 1-D, unsigned distance
  //
  {
    typedef std::array<double, 1> Pt;
    typedef std::array < Pt, 1 + 1 > Simplex;

    Simplex s = {{Pt{{0.}}, Pt{{1.}}}};

    assert(std::abs(computeUnsignedDistance(s, Pt{{0.}}) - 0) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.5}}) - 0) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{1.}}) - 0) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{-1.}}) - 1) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{2.}}) - 1) < eps);
  }

  //
  // 2-D, unsigned distance
  //
  {
    typedef std::array<double, 2> Pt;
    typedef std::array < Pt, 2 + 1 > Simplex;

    Simplex s = {{Pt{{0., 0.}}, Pt{{1., 0.}}, Pt{{0., 1.}}}};

    // Face 0.
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.5, -1.}}) - 1) < eps);
    // Face 1.
    assert(std::abs(computeUnsignedDistance(s, Pt{{1., 1.}}) - std::sqrt(2.) / 2)
           < eps);
    // Face 2.
    assert(std::abs(computeUnsignedDistance(s, Pt{{-1., 0.5}}) - 1)
           < eps);
    // Vertex 0.
    assert(std::abs(computeUnsignedDistance(s, Pt{{-1., -1.}}) - std::sqrt(2.))
           < eps);
    // Vertex 1.
    assert(std::abs(computeUnsignedDistance(s, Pt{{2., -1.}}) - std::sqrt(2.))
           < eps);
    // Vertex 2.
    assert(std::abs(computeUnsignedDistance(s, Pt{{-1., 2.}}) - std::sqrt(2.))
           < eps);
    // Inside.
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.25, 0.25}})) < eps);
  }

  //
  // 3-D, unsigned distance
  //
  {
    typedef std::array<double, 3> Pt;
    typedef std::array < Pt, 3 + 1 > Simplex;

    Simplex s = {{Pt{{0., 0., 0.}}, Pt{{1., 0., 0.}}, Pt{{0., 1., 0.}},
                 Pt{{0., 0., 1.}}}};

    // Inside
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.1, 0.2, 0.2}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.2, 0.1, 0.2}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.2, 0.2, 0.1}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.25, 0.25, 0.25}})) < eps);

    // Zero distance, at the vertices.
    assert(std::abs(computeUnsignedDistance(s, Pt{{0., 0., 0.}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{1., 0., 0.}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0., 1., 0.}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0., 0., 1.}})) < eps);
    // Zero distance, on the edges.
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.5, 0., 0.}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0., 0.5, 0.}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0., 0., 0.5}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.5, 0.5, 0.}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.5, 0., 0.5}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0., 0.5, 0.5}})) < eps);
    // Zero distance, on the faces.
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.25, 0.25, 0.}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.25, 0., 0.25}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0., 0.25, 0.25}})) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.5, 0.25, 0.25}})) < eps);

    // Positive distance from the faces.
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.25, 0.25, -1.}}) - 1)
           < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.25, -1., 0.25}}) - 1)
           < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{-1., 0.25, 0.25}}) - 1)
           < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{1., 1., 1.}})
                    - std::sqrt(4. / 3.)) < eps);
    // Positive distance from the edges.
    assert(std::abs(computeUnsignedDistance(s, Pt{{0.5, -1., -1.}})
                    - std::sqrt(2.)) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{-1., 0.5, -1.}})
                    - std::sqrt(2.)) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{-1., -1., 0.5}})
                    - std::sqrt(2.)) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{1., 1., -1.}})
                    - std::sqrt(3. / 2.)) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{1., -1., 1.}})
                    - std::sqrt(3. / 2.)) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{-1., 1., 1.}})
                    - std::sqrt(3. / 2.)) < eps);
    // Positive distance from the vertices.
    assert(std::abs(computeUnsignedDistance(s, Pt{{-1., -1., -1.}})
                    - std::sqrt(3.)) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{2., 0., 0.}}) - 1) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0., 2., 0.}}) - 1) < eps);
    assert(std::abs(computeUnsignedDistance(s, Pt{{0., 0., 2.}}) - 1) < eps);
  }

  //---------------------------------------------------------------------------
  // Signed distance.
  //---------------------------------------------------------------------------
  {
    //
    // 1-D point.
    //
    typedef std::array<double, 1> Pt;
    {
      Pt p = {{2.0}}, n = {{1.0}}, x = {{5.0}};
      assert(std::abs(computeSignedDistance(p, n, x) - 3) < eps);
    }
    {
      Pt p = {{2.0}}, n = {{ -1.0}}, x = {{5.0}};
      assert(std::abs(computeSignedDistance(p, n, x) + 3) < eps);
    }
  }

  {
    //
    // 2-D point.
    //
    typedef std::array<double, 2> Pt;
    {
      Pt p = {{0, 0}}, n = {{1, 0}}, x = {{1, 0}};
      assert(std::abs(computeSignedDistance(p, n, x) - 1) < eps);
    }
    {
      Pt p = {{0, 0}}, n = {{1, 0}}, x = {{ -1, 0}};
      assert(std::abs(computeSignedDistance(p, n, x) + 1) < eps);
    }
    {
      Pt p = {{0, 0}}, n = {{1, 0}}, x = {{1, 1}};
      assert(std::abs(computeSignedDistance(p, n, x) - std::sqrt(2.0)) < eps);
    }
    {
      Pt p = {{0, 0}}, n = {{1, 0}}, x = {{1, 1}};
      assert(std::abs(computeSignedDistance(p, n, x) - std::sqrt(2.0)) < eps);
    }
    {
      Pt p = {{0, 0}}, n = {{1, 0}}, x = {{1, -1}};
      assert(std::abs(computeSignedDistance(p, n, x) - std::sqrt(2.0)) < eps);
    }
    {
      Pt p = {{0, 0}}, n = {{1, 0}}, x = {{ -1, 1}};
      assert(std::abs(computeSignedDistance(p, n, x) + std::sqrt(2.0)) < eps);
    }
    {
      Pt p = {{0, 0}}, n = {{1, 0}}, x = {{ -1, -1}};
      assert(std::abs(computeSignedDistance(p, n, x) + std::sqrt(2.0)) < eps);
    }
  }
  {
    //
    // 3-D point.
    //
    typedef std::array<double, 3> Pt;
    {
      Pt p = {{0., 0., 0.}}, n = {{1., 0., 0.}}, x = {{1., 0., 0.}};
      assert(std::abs(computeSignedDistance(p, n, x) - 1) < eps);
    }
    {
      Pt p = {{0., 0., 0.}}, n = {{1., 0., 0.}}, x = {{ -1, 0, 0}};
      assert(std::abs(computeSignedDistance(p, n, x) + 1) < eps);
    }

    {
      Pt p = {{0., 0., 0.}}, n = {{1., 0., 0.}}, x = {{1, 1, 1}};
      assert(std::abs(computeSignedDistance(p, n, x) - std::sqrt(3.0)) < eps);
    }
    {
      Pt p = {{0., 0., 0.}}, n = {{1., 0., 0.}}, x = {{ -1, 1, 1}};
      assert(std::abs(computeSignedDistance(p, n, x) + std::sqrt(3.0)) < eps);
    }
  }

  {
    //
    // 1-simplex, 2-D point, distance.
    //
    typedef std::array<double, 2> Pt;
    typedef std::array < Pt, 1 + 1 > S;

    S s = {{Pt{{0., 0.}}, Pt{{1., 0.}}}};

    assert(std::abs(computeSignedDistance(s, Pt{{0.1, 1.}}) + 1) < eps);
    assert(std::abs(computeSignedDistance(s, Pt{{0.9, 1.}}) + 1) < eps);

    assert(std::abs(computeSignedDistance(s, Pt{{0.1, -1.}}) - 1) < eps);
    assert(std::abs(computeSignedDistance(s, Pt{{0.9, -1.}}) - 1) < eps);

    assert(computeSignedDistance(s, Pt{{-1., 1.}}) ==
           std::numeric_limits<double>::max());
    assert(computeSignedDistance(s, Pt{{2., 1.}}) ==
           std::numeric_limits<double>::max());
  }
  {
    //
    // 1-simplex, 2-D point, distance, closest point.
    //
    typedef std::array<double, 2> Pt;
    typedef std::array < Pt, 1 + 1 > S;

    S s = {{Pt{{0., 0.}}, Pt{{1., 0.}}}};
    Pt cp;

    assert(std::abs(computeSignedDistance(s, Pt{{0.1, 1.}}, &cp) + 1) < eps);
    assert(geom::computeDistance(cp, Pt{{0.1, 0.}}) < eps);
    assert(std::abs(computeSignedDistance(s, Pt{{0.9, 1.}}, &cp) + 1) < eps);
    assert(geom::computeDistance(cp, Pt{{0.9, 0.}}) < eps);

    assert(std::abs(computeSignedDistance(s, Pt{{0.1, -1.}}, &cp) - 1) < eps);
    assert(geom::computeDistance(cp, Pt{{0.1, 0.}}) < eps);
    assert(std::abs(computeSignedDistance(s, Pt{{0.9, -1.}}, &cp) - 1) < eps);
    assert(geom::computeDistance(cp, Pt{{0.9, 0.}}) < eps);

    assert(computeSignedDistance(s, Pt{{-1., 1.}}, &cp) ==
           std::numeric_limits<double>::max());
    assert(computeSignedDistance(s, Pt{{2., 1.}}, &cp) ==
           std::numeric_limits<double>::max());
  }
  {
    //
    // 1-simplex, 3-D point.
    //
    typedef std::array<double, 3> Pt;
    typedef std::array < Pt, 1 + 1 > S;

    S s = {{Pt{{0., 0., 0.}}, Pt{{1., 0., 0.}}}};
    Pt n = {{0, 1, 0}};

    assert(std::abs(computeSignedDistance(s, n, Pt{{0., 0., 0.}}) - 0) < eps);
    assert(std::abs(computeSignedDistance(s, n, Pt{{0.5, 0., 0.}}) - 0) < eps);
    assert(std::abs(computeSignedDistance(s, n, Pt{{1., 0., 0.}}) -  0) < eps);
    assert(computeSignedDistance(s, n, Pt{{-1., 0., 0.}}) ==
           std::numeric_limits<double>::max());
    assert(computeSignedDistance(s, n, Pt{{2., 0., 0.}}) ==
           std::numeric_limits<double>::max());

    assert(std::abs(computeSignedDistance(s, n, Pt{{0., 1., 0.}}) - 1) < eps);
    assert(std::abs(computeSignedDistance(s, n, Pt{{0.5, 1., 0.}}) - 1) < eps);
    assert(std::abs(computeSignedDistance(s, n, Pt{{1., 1., 0.}}) -  1) < eps);
    assert(computeSignedDistance(s, n, Pt{{-1., 1., 0.}}) ==
           std::numeric_limits<double>::max());
    assert(computeSignedDistance(s, n, Pt{{2., 1., 0.}}) ==
           std::numeric_limits<double>::max());
  }
  {
    //
    // 2-simplex, 3-D point.
    //
    typedef std::array<double, 3> Pt;
    typedef std::array < Pt, 2 + 1 > Simplex;

    Simplex s = {{Pt{{0., 0., 0.}}, Pt{{1., 0., 0.}}, Pt{{0., 1., 0.}}}};

    // Face 0.
    assert(computeSignedDistance(s, Pt{{0.5, -1., 0.}}) ==
           std::numeric_limits<double>::max());
    // Face 1.
    assert(computeSignedDistance(s, Pt{{1., 1., 0.}})  ==
           std::numeric_limits<double>::max());
    // Face 2.
    assert(computeSignedDistance(s, Pt{{-1., 0.5, 0.}}) ==
           std::numeric_limits<double>::max());
    // Vertex 0.
    assert(computeSignedDistance(s, Pt{{-1., -1., 0.}}) ==
           std::numeric_limits<double>::max());
    // Vertex 1.
    assert(computeSignedDistance(s, Pt{{2., -1., 0.}}) ==
           std::numeric_limits<double>::max());
    // Vertex 2.
    assert(computeSignedDistance(s, Pt{{-1., 2., 0.}}) ==
           std::numeric_limits<double>::max());
    // Inside.
    assert(std::abs(computeSignedDistance(s, Pt{{0.25, 0.25, 0.}}) - 0) < eps);

    // Face 0, above.
    assert(computeSignedDistance(s, Pt{{0.5, -1., 1.}}) ==
           std::numeric_limits<double>::max());
    // Face 1, above.
    assert(computeSignedDistance(s, Pt{{1., 1., 1.}})  ==
           std::numeric_limits<double>::max());
    // Face 2, above.
    assert(computeSignedDistance(s, Pt{{-1., 0.5, 1.}}) ==
           std::numeric_limits<double>::max());
    // Vertex 0, above.
    assert(computeSignedDistance(s, Pt{{-1., -1., 1.}}) ==
           std::numeric_limits<double>::max());
    // Vertex 1, above.
    assert(computeSignedDistance(s, Pt{{2., -1., 1.}}) ==
           std::numeric_limits<double>::max());
    // Vertex 2, above.
    assert(computeSignedDistance(s, Pt{{-1., 2., 1.}}) ==
           std::numeric_limits<double>::max());
    // Inside, above.
    assert(std::abs(computeSignedDistance(s, Pt{{0.25, 0.25, 1.}}) - 1) < eps);

    // Face 0, below.
    assert(computeSignedDistance(s, Pt{{0.5, -1., -1.}}) ==
           std::numeric_limits<double>::max());
    // Face 1, below.
    assert(computeSignedDistance(s, Pt{{1., 1., -1.}})  ==
           std::numeric_limits<double>::max());
    // Face 2, below.
    assert(computeSignedDistance(s, Pt{{-1., 0.5, -1.}}) ==
           std::numeric_limits<double>::max());
    // Vertex 0, below.
    assert(computeSignedDistance(s, Pt{{-1., -1., -1.}}) ==
           std::numeric_limits<double>::max());
    // Vertex 1, below.
    assert(computeSignedDistance(s, Pt{{2., -1., -1.}}) ==
           std::numeric_limits<double>::max());
    // Vertex 2, below.
    assert(computeSignedDistance(s, Pt{{-1., 2., -1.}}) ==
           std::numeric_limits<double>::max());
    // Inside, below.
    assert(std::abs(computeSignedDistance(s, Pt{{0.25, 0.25, -1.}}) + 1) < eps);
  }


  //---------------------------------------------------------------------------
  // Project to a lower dimension.
  //---------------------------------------------------------------------------
  {
    //
    // 2-D -> 1-D
    //
    typedef std::array<double, 1> P1;
    typedef std::array<double, 2> P2;
    typedef std::array < P1, 1 + 1 > S1;
    typedef std::array < P2, 1 + 1 > S2;

    {
      S2 s2 = {{{{0., 0.}}, {{1., 0.}}}};
      P2 x2 = {{2, 0}};
      S1 s1;
      P1 x1, y1;
      project(s2, x2, &s1, &x1);
      assert(std::abs(s1[0][0] - 0) < eps);
      assert(std::abs(s1[1][0] - 1) < eps);
      assert(std::abs(x1[0] - 2) < eps);
      project(s2, x2, &s1, &x1, &y1);
      assert(std::abs(y1[0] - 0) < eps);
    }
    {
      S2 s2 = {{{{1., 0.}}, {{2., 0.}}}};
      P2 x2 = {{3, 0}};
      S1 s1;
      P1 x1, y1;
      project(s2, x2, &s1, &x1);
      assert(std::abs(s1[0][0] - 0) < eps);
      assert(std::abs(s1[1][0] - 1) < eps);
      assert(std::abs(x1[0] - 2) < eps);
      project(s2, x2, &s1, &x1, &y1);
      assert(std::abs(y1[0] - 0) < eps);
    }
    {
      S2 s2 = {{{{0., 0.}}, {{1., 0.}}}};
      P2 x2 = {{2, 3}};
      S1 s1;
      P1 x1, y1;
      project(s2, x2, &s1, &x1);
      assert(std::abs(s1[0][0] - 0) < eps);
      assert(std::abs(s1[1][0] - 1) < eps);
      assert(std::abs(x1[0] - 2) < eps);
      project(s2, x2, &s1, &x1, &y1);
      assert(std::abs(y1[0] - 3) < eps);
    }
    {
      S2 s2 = {{{{0., 0.}}, {{0., 1.}}}};
      P2 x2 = {{0, 2}};
      S1 s1;
      P1 x1, y1;
      project(s2, x2, &s1, &x1);
      assert(std::abs(s1[0][0] - 0) < eps);
      assert(std::abs(s1[1][0] - 1) < eps);
      assert(std::abs(x1[0] - 2) < eps);
      project(s2, x2, &s1, &x1, &y1);
      assert(std::abs(y1[0] - 0) < eps);
    }
    {
      S2 s2 = {{{{1., 1.}}, {{2., 2.}}}};
      P2 x2 = {{0, 4}};
      S1 s1;
      P1 x1, y1;
      project(s2, x2, &s1, &x1);
      assert(std::abs(s1[0][0] - 0) < eps);
      assert(std::abs(s1[1][0] - std::sqrt(2.)) < eps);
      assert(std::abs(x1[0] - std::sqrt(2.)) < eps);
      project(s2, x2, &s1, &x1, &y1);
      assert(std::abs(y1[0] - 2 * std::sqrt(2.)) < eps);
    }
  }

  {
    //
    // 3-D -> 2-D
    //
    typedef std::array<double, 1> P1;
    typedef std::array<double, 2> P2;
    typedef std::array<double, 3> P3;
    typedef std::array < P2, 2 + 1 > S2;
    typedef std::array < P3, 2 + 1 > S3;

    {
      // A triangle and point that lie in the xy plane.
      S3 s3 = {{{{0., 0., 0.}}, {{1., 0., 0.}}, {{0., 1., 0.}}}};
      P3 x3 = {{1, 1, 0}};
      S2 s2;
      P2 x2;
      P1 z1;
      project(s3, x3, &s2, &x2);
      assert(geom::computeDistance(s2[0], P2{{0., 0.}}) < eps);
      assert(geom::computeDistance(s2[1], P2{{1., 0.}}) < eps);
      assert(geom::computeDistance(s2[2], P2{{0., 1.}}) < eps);
      assert(geom::computeDistance(x2, P2{{1., 1.}}) < eps);
      project(s3, x3, &s2, &x2, &z1);
      assert(std::abs(z1[0] - 0) < eps);
    }
    {
      // The above test, translated by (1, 2, 3).
      S3 s3 = {{{{0., 0., 0.}}, {{1., 0., 0.}}, {{0., 1., 0.}}}};
      P3 x3 = {{1, 1, 0}};
      s3[0] += P3{{1., 2., 3.}};
      s3[1] += P3{{1., 2., 3.}};
      s3[2] += P3{{1., 2., 3.}};
      x3 += P3{{1., 2., 3.}};
      S2 s2;
      P2 x2;
      P1 z1;
      project(s3, x3, &s2, &x2);
      assert(geom::computeDistance(s2[0], P2{{0., 0.}}) < eps);
      assert(geom::computeDistance(s2[1], P2{{1., 0.}}) < eps);
      assert(geom::computeDistance(s2[2], P2{{0., 1.}}) < eps);
      assert(geom::computeDistance(x2, P2{{1., 1.}}) < eps);
      project(s3, x3, &s2, &x2, &z1);
      assert(std::abs(z1[0] - 0) < eps);
    }
    {
      // A triangle in the xy plane and point above the xy plane.
      S3 s3 = {{{{0., 0., 0.}}, {{1., 0., 0.}}, {{0., 1., 0.}}}};
      P3 x3 = {{1, 1, 10}};
      S2 s2;
      P2 x2;
      P1 z1;
      project(s3, x3, &s2, &x2);
      assert(geom::computeDistance(s2[0], P2{{0., 0.}}) < eps);
      assert(geom::computeDistance(s2[1], P2{{1., 0.}}) < eps);
      assert(geom::computeDistance(s2[2], P2{{0., 1.}}) < eps);
      assert(geom::computeDistance(x2, P2{{1., 1.}}) < eps);
      project(s3, x3, &s2, &x2, &z1);
      assert(std::abs(z1[0] - 10) < eps);
    }
    {
      // The above test, translated by (-2, -3, -5).
      S3 s3 = {{{{0., 0., 0.}}, {{1., 0., 0.}}, {{0., 1., 0.}}}};
      P3 x3 = {{1, 1, 10}};
      s3[0] -= P3{{2., 3., 5.}};
      s3[1] -= P3{{2., 3., 5.}};
      s3[2] -= P3{{2., 3., 5.}};
      x3 -= P3{{2., 3., 5.}};
      S2 s2;
      P2 x2;
      P1 z1;
      project(s3, x3, &s2, &x2);
      assert(geom::computeDistance(s2[0], P2{{0., 0.}}) < eps);
      assert(geom::computeDistance(s2[1], P2{{1., 0.}}) < eps);
      assert(geom::computeDistance(s2[2], P2{{0., 1.}}) < eps);
      assert(geom::computeDistance(x2, P2{{1., 1.}}) < eps);
      project(s3, x3, &s2, &x2, &z1);
      assert(std::abs(z1[0] - 10) < eps);
    }
    {
      // A triangle and point that lies in the xz plane.
      S3 s3 = {{{{0., 0., 0.}}, {{1., 0., 0.}}, {{0., 0., 1.}}}};
      P3 x3 = {{1, 0, 1}};
      S2 s2;
      P2 x2;
      P1 z1;
      project(s3, x3, &s2, &x2);
      assert(geom::computeDistance(s2[0], P2{{0., 0.}}) < eps);
      assert(geom::computeDistance(s2[1], P2{{1., 0.}}) < eps);
      assert(geom::computeDistance(s2[2], P2{{0., 1.}}) < eps);
      assert(geom::computeDistance(x2, P2{{1., 1.}}) < eps);
      project(s3, x3, &s2, &x2, &z1);
      assert(std::abs(z1[0] - 0) < eps);
    }
    {
      // A triangle and point that will be rotated by pi.
      S3 s3 = {{{{0., 0., 0.}}, {{1., 0., 0.}}, {{0., -1., 0.}}}};
      P3 x3 = {{1, -1, 0}};
      S2 s2;
      P2 x2;
      P1 z1;
      project(s3, x3, &s2, &x2);
      assert(geom::computeDistance(s2[0], P2{{0., 0.}}) < eps);
      assert(geom::computeDistance(s2[1], P2{{1., 0.}}) < eps);
      assert(geom::computeDistance(s2[2], P2{{0., 1.}}) < eps);
      assert(geom::computeDistance(x2, P2{{1., 1.}}) < eps);
      project(s3, x3, &s2, &x2, &z1);
      assert(std::abs(z1[0] - 0) < eps);
    }
    {
      // A triangle and point that will be translated by (2,3,5) and
      // rotated by pi.
      S3 s3 = {{{{0., 0., 0.}}, {{1., 0., 0.}}, {{0., -1., 0.}}}};
      P3 x3 = {{1, -1, 0}};
      s3[0] -= P3{{2., 3., 5.}};
      s3[1] -= P3{{2., 3., 5.}};
      s3[2] -= P3{{2., 3., 5.}};
      x3 -= P3{{2., 3., 5.}};
      S2 s2;
      P2 x2;
      P1 z1;
      project(s3, x3, &s2, &x2);
      assert(geom::computeDistance(s2[0], P2{{0., 0.}}) < eps);
      assert(geom::computeDistance(s2[1], P2{{1., 0.}}) < eps);
      assert(geom::computeDistance(s2[2], P2{{0., 1.}}) < eps);
      assert(geom::computeDistance(x2, P2{{1., 1.}}) < eps);
      project(s3, x3, &s2, &x2, &z1);
      assert(std::abs(z1[0] - 0) < eps);
    }
  }

  signedDistanceContinuous<2>();
  signedDistanceContinuous<3>();

  return 0;
}
