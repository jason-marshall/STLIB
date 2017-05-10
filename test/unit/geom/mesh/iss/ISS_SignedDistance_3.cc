// -*- C++ -*-

#include "stlib/geom/mesh/iss/ISS_SignedDistance.h"
#include "stlib/geom/mesh/iss/IndSimpSet.h"
#include "stlib/geom/mesh/iss/build.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

int
main()
{
  typedef geom::IndSimpSet<3, 2> ISS;
  typedef ISS::Vertex Vertex;
  typedef geom::ISS_SignedDistance<ISS> ISS_SD;

  const double eps = 10 * std::numeric_limits<double>::epsilon();

  {
    //
    // Data for a triangle.
    //
    const std::size_t numVertices = 3;
    double vertices[] = {0, 0, 0,   // 0
                         1, 0, 0,    // 1
                         0, 1, 0
                        };   // 2
    const std::size_t numSimplices = 1;
    std::size_t simplices[] = {0, 1, 2};

    ISS iss;
    build(&iss, numVertices, vertices, numSimplices, simplices);
    ISS_SD x(iss);
    Vertex cp, grad;
    std::size_t index;

    // On the face.
    assert(std::abs(x(Vertex{{0.25, 0.25, 0.}}) - 0) < eps);
    assert(std::abs(x(Vertex{{0.25, 0.25, 0.}}, &cp) - 0) < eps);
    assert(geom::computeDistance(cp, Vertex{{0.25, 0.25, 0.}}) < eps);
    assert(std::abs(x(Vertex{{0.25, 0.25, 0.}}, &cp, &grad, &index) - 0) < eps);
    assert(geom::computeDistance(cp, Vertex{{0.25, 0.25, 0.}}) < eps);
    assert(geom::computeDistance(grad, Vertex{{0, 0, 1}}) < eps);
    assert(index == 0);

    // Above the face.
    assert(std::abs(x(Vertex{{0.25, 0.25, 1}}) - 1) < eps);
    assert(std::abs(x(Vertex{{0.25, 0.25, 1}}, &cp) - 1) < eps);
    assert(geom::computeDistance(cp, Vertex{{0.25, 0.25, 0.}}) < eps);
    assert(std::abs(x(Vertex{{0.25, 0.25, 1}}, &cp, &grad, &index) - 1) < eps);
    assert(geom::computeDistance(cp, Vertex{{0.25, 0.25, 0.}}) < eps);
    assert(geom::computeDistance(grad, Vertex{{0, 0, 1}}) < eps);
    assert(index == 0);

    // Below the face.
    assert(std::abs(x(Vertex{{0.25, 0.25, -1}}) + 1) < eps);
    assert(std::abs(x(Vertex{{0.25, 0.25, -1}}, &cp) + 1) < eps);
    assert(geom::computeDistance(cp, Vertex{{0.25, 0.25, 0.}}) < eps);
    assert(std::abs(x(Vertex{{0.25, 0.25, -1}}, &cp, &grad, &index) + 1) < eps);
    assert(geom::computeDistance(cp, Vertex{{0.25, 0.25, 0.}}) < eps);
    assert(geom::computeDistance(grad, Vertex{{0, 0, 1}}) < eps);
    assert(index == 0);

    // Vertex distance
    assert(std::abs(x(Vertex{{0, -3, 4}}) - 5) < eps);
    assert(std::abs(x(Vertex{{0, -3, 4}}, &cp) - 5) < eps);
    assert(geom::computeDistance(cp, Vertex{{0, 0, 0}}) < eps);
    assert(std::abs(x(Vertex{{0, -3, 4}}, &cp, &grad, &index) - 5) < eps);
    assert(geom::computeDistance(cp, Vertex{{0, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Vertex{{0, -3. / 5., 4. / 5.}}) < eps);
    assert(index == 0);

    geom::ISS_SD_Distance<ISS> df(x);
    assert(std::abs(df(Vertex{{0.25, 0.25, 0.}}) - 0) < eps);
  }


  {
    //
    // Data for an octahedron
    //
    const std::size_t numVertices = 6;
    double vertices[] = {1, 0, 0,    // 0
                         -1, 0, 0,    // 1
                         0, 1, 0,     // 2
                         0, -1, 0,    // 3
                         0, 0, 1,     // 4
                         0, 0, -1
                        };   // 5
    const std::size_t numSimplices = 8;
    std::size_t simplices[] = {0, 2, 4,
                               2, 0, 5,
                               2, 1, 4,
                               1, 2, 5,
                               1, 3, 4,
                               3, 1, 5,
                               3, 0, 4,
                               0, 3, 5
                              };

    ISS iss;
    build(&iss, numVertices, vertices, numSimplices, simplices);
    ISS_SD x(iss);
    Vertex cp, grad;
    std::size_t index;

    // Face, outside.
    assert(std::abs(x(Vertex{{1, 1, 1}}) - 2. / std::sqrt(3.)) < eps);
    assert(std::abs(x(Vertex{{1, 1, 1}}, &cp) - 2. / std::sqrt(3.)) < eps);
    assert(geom::computeDistance(cp, Vertex{{1. / 3., 1. / 3., 1. / 3.}}) < eps);
    assert(std::abs(x(Vertex{{1, 1, 1}}, &cp, &grad, &index) - 2. / std::sqrt(3.))
           < eps);
    assert(geom::computeDistance(cp, Vertex{{1. / 3., 1. / 3., 1. / 3.}}) < eps);
    assert(geom::computeDistance(grad, Vertex{{1, 1, 1}} / std::sqrt(3.)) < eps);
    assert(index == 0);

    // Face, inside.
    assert(std::abs(x(Vertex{{0, 0, 0}}) + 1. / std::sqrt(3.)) < eps);
    assert(std::abs(x(Vertex{{0, 0, 0}}, &cp) + 1. / std::sqrt(3.)) < eps);
    x(Vertex{{0.1, 0.1, 0.1}}, &cp);
    assert(geom::computeDistance(cp, Vertex{{1. / 3., 1. / 3., 1. / 3.}}) < eps);

    // Edge, outside.
    assert(std::abs(x(Vertex{{1, 1, 0}}) - std::sqrt(2.) / 2.) < eps);
    assert(std::abs(x(Vertex{{1, 1, 0}}, &cp) - std::sqrt(2.) / 2.) < eps);
    assert(geom::computeDistance(cp, Vertex{{0.5, 0.5, 0}}) < eps);
    assert(std::abs(x(Vertex{{1, 1, 0}}, &cp, &grad, &index) - std::sqrt(2.) / 2.)
           < eps);
    assert(geom::computeDistance(cp, Vertex{{0.5, 0.5, 0}}) < eps);
    assert(geom::computeDistance(grad, Vertex{{1, 1, 0}} / std::sqrt(2.)) < eps);
    assert(index == 0 || index == 1);

    // Vertex, outside.
    assert(std::abs(x(Vertex{{2, 0, 0}}) - 1) < eps);
    assert(std::abs(x(Vertex{{2, 0, 0}}, &cp) - 1) < eps);
    assert(geom::computeDistance(cp, Vertex{{1, 0, 0}}) < eps);
    assert(std::abs(x(Vertex{{2, 0, 0}}, &cp, &grad, &index) - 1) < eps);
    assert(geom::computeDistance(cp, Vertex{{1, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Vertex{{1, 0, 0}}) < eps);
    assert(index == 0 || index == 1 || index == 6 || index == 7);
  }

  return 0;
}
