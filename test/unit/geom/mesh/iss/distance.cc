// -*- C++ -*-

#include "stlib/geom/mesh/iss/distance.h"
#include "stlib/geom/mesh/iss/build.h"

#include <iostream>

#include <cassert>

using namespace stlib;

#define PT ext::make_array<double>

int
main()
{
  const double eps = 10 * std::numeric_limits<double>::epsilon();

  {
    typedef geom::IndSimpSetIncAdj<2, 1> ISS;
    typedef ISS::Vertex Vertex;

    //
    // Data for a square.
    //
    const std::size_t numVertices = 4;
    double vertices[] = {0, 0,   // 0
                         1, 0,    // 1
                         1, 1,    // 2
                         0, 1
                        };   // 3
    const std::size_t numSimplices = 4;
    std::size_t simplices[] = {0, 1,
                               1, 2,
                               2, 3,
                               3, 0
                              };

    ISS mesh;
    build(&mesh, numVertices, vertices, numSimplices, simplices);
    Vertex cp;

    // (0, 0)
    assert(std::abs(geom::computeSignedDistance(mesh, Vertex{{0, 0}}) - 0) < eps);
    assert(std::abs(geom::computeSignedDistance(mesh, Vertex{{0, 0}}, &cp) - 0) < eps);
    assert(geom::computeDistance(cp, Vertex{{0, 0}}) < eps);

    // (0, -1)
    assert(std::abs(geom::computeSignedDistance(mesh, Vertex{{0, -1}}) - 1) < eps);
    assert(std::abs(geom::computeSignedDistance(mesh, Vertex{{0, -1}}, &cp) - 1) <
           eps);
    assert(geom::computeDistance(cp, Vertex{{0, 0}}) < eps);

    // (-1, 0)
    assert(std::abs(geom::computeSignedDistance(mesh, Vertex{{-1, 0}}) - 1) < eps);
    assert(std::abs(geom::computeSignedDistance(mesh, Vertex{{-1, 0}}, &cp) - 1) <
           eps);
    assert(geom::computeDistance(cp, Vertex{{0, 0}}) < eps);

    // (0.5, -1)
    assert(std::abs(geom::computeSignedDistance(mesh, Vertex{{0.5, -1}}) - 1) < eps);
    assert(std::abs(geom::computeSignedDistance(mesh, Vertex{{0.5, -1}}, &cp) - 1) <
           eps);
    assert(geom::computeDistance(cp, Vertex{{0.5, 0}}) < eps);

    // (0.5, 0.25)
    assert(std::abs(geom::computeSignedDistance(mesh, Vertex{{0.5, 0.25}}) + 0.25) <
           eps);
    assert(std::abs(geom::computeSignedDistance(mesh, Vertex{{0.5, 0.25}}, &cp)
                    + 0.25) < eps);
    assert(geom::computeDistance(cp, Vertex{{0.5, 0}}) < eps);

    // (-3, -4)
    assert(std::abs(geom::computeSignedDistance(mesh, Vertex{{-3, -4}}) - 5) < eps);
    assert(std::abs(geom::computeSignedDistance(mesh, Vertex{{-3, -4}}, &cp) - 5) <
           eps);
    assert(geom::computeDistance(cp, Vertex{{0, 0}}) < eps);

    std::vector<Vertex> points;
    points.push_back(Vertex{{0.5, 0.25}});
    points.push_back(Vertex{{-3, -4}});
    std::vector<double> distances;
    geom::computeSignedDistance(mesh, points.begin(), points.end(),
                                std::back_inserter(distances));
    assert(distances.size() == points.size());
    assert(std::abs(distances[0] + 0.25) < eps);
    assert(std::abs(distances[1] - 5) < eps);

    distances.clear();
    std::vector<Vertex> closestPoints;
    geom::computeSignedDistance(mesh, points.begin(), points.end(),
                                std::back_inserter(distances),
                                std::back_inserter(closestPoints));
    assert(distances.size() == points.size());
    assert(closestPoints.size() == points.size());
    assert(std::abs(distances[0] + 0.25) < eps);
    assert(std::abs(distances[1] - 5) < eps);
    assert(geom::computeDistance(closestPoints[0], Vertex{{0.5, 0}}) < eps);
    assert(geom::computeDistance(closestPoints[1], Vertex{{0, 0}}) < eps);
  }

  {
    typedef geom::IndSimpSetIncAdj<3, 2> ISS;
    typedef ISS::Vertex Vertex;

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

    ISS mesh;
    build(&mesh, numVertices, vertices, numSimplices, simplices);
    Vertex p, cp;

    // Face, outside.
    p = Vertex{{1, 1, 1}};
    assert(std::abs(geom::computeSignedDistance(mesh, p) - 2. / std::sqrt(3.))
           < eps);
    assert(std::abs(geom::computeSignedDistance(mesh, p, &cp) -
                    2. / std::sqrt(3.)) < eps);
    assert(geom::computeDistance(cp, Vertex{{1. / 3., 1. / 3., 1. / 3.}}) < eps);

    // Face, inside.
    p = Vertex{{0, 0, 0}};
    assert(std::abs(geom::computeSignedDistance(mesh, p) + 1. / std::sqrt(3.))
           < eps);
    assert(std::abs(geom::computeSignedDistance(mesh, p, &cp) +
                    1. / std::sqrt(3.)) < eps);
    assert(std::abs(geom::computeDistance(p, cp) - 1. / std::sqrt(3.)) < eps);

    // Edge, outside.
    p = Vertex{{1, 1, 0}};
    assert(std::abs(geom::computeSignedDistance(mesh, p) - std::sqrt(2.) / 2.)
           < eps);
    assert(std::abs(geom::computeSignedDistance(mesh, p, &cp) -
                    std::sqrt(2.) / 2.) < eps);
    assert(geom::computeDistance(cp, Vertex{{0.5, 0.5, 0}}) < eps);

    // Vertex, outside.
    p = Vertex{{2, 0, 0}};
    assert(std::abs(geom::computeSignedDistance(mesh, p) - 1) < eps);
    assert(std::abs(geom::computeSignedDistance(mesh, p, &cp) - 1) < eps);
    assert(geom::computeDistance(cp, Vertex{{1, 0, 0}}) < eps);

    std::vector<Vertex> points;
    points.push_back(Vertex{{1, 1, 1}});
    points.push_back(Vertex{{2, 0, 0}});
    std::vector<double> distances;
    geom::computeSignedDistance(mesh, points.begin(), points.end(),
                                std::back_inserter(distances));
    assert(distances.size() == points.size());
    assert(std::abs(distances[0] - 2. / std::sqrt(3.)) < eps);
    assert(std::abs(distances[1] - 1) < eps);

    distances.clear();
    std::vector<Vertex> closestPoints;
    geom::computeSignedDistance(mesh, points.begin(), points.end(),
                                std::back_inserter(distances),
                                std::back_inserter(closestPoints));
    assert(distances.size() == points.size());
    assert(closestPoints.size() == points.size());
    assert(std::abs(distances[0] - 2. / std::sqrt(3.)) < eps);
    assert(std::abs(distances[1] - 1) < eps);
    assert(geom::computeDistance(closestPoints[0], Vertex{{1. / 3., 1. / 3., 1. / 3.}})
           < eps);
    assert(geom::computeDistance(closestPoints[1], Vertex{{1, 0, 0}}) < eps);
  }

  return 0;
}
