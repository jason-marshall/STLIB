// -*- C++ -*-

#include "stlib/geom/mesh/iss/contact.h"
#include "stlib/geom/mesh/iss/build.h"

#include <iostream>
#include <vector>

#include <cassert>

using namespace stlib;

int
main()
{
  const double Epsilon = 10 * std::numeric_limits<double>::epsilon();

  //
  // 2-D.
  //
  {
    typedef geom::IndSimpSet<2, 1> ISS;
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

    ISS surface;
    build(&surface, numVertices, vertices, numSimplices, simplices);
    std::vector<Vertex> points;
    points.push_back(Vertex{{0., 0.}});
    points.push_back(Vertex{{0.5, 0.1}});
    points.push_back(Vertex{{0.5, -0.1}});

    geom::removeContact(surface, points.begin(), points.end());
    assert(geom::computeDistance(points[0], Vertex{{0.,
                                 0.}}) < Epsilon);
    assert(geom::computeDistance(points[1], Vertex{{0.5,
                                 0}}) < Epsilon);
    assert(geom::computeDistance(points[2], Vertex{{0.5,
                                 -0.1}}) < Epsilon);
  }
  {
    typedef geom::IndSimpSet<3, 2> ISS;
    typedef ISS::Vertex Vertex;

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

    ISS surface;
    build(&surface, numVertices, vertices, numSimplices, simplices);
    std::vector<Vertex> points;
    points.push_back(Vertex{{0.25, 0.25, 0}});
    points.push_back(Vertex{{0.25, 0.25, 0.1}});
    points.push_back(Vertex{{0.25, 0.25, -0.1}});

    geom::removeContact(surface, points.begin(), points.end());
    assert(geom::computeDistance(points[0], Vertex{{0.25, 0.25,
                                 0}}) < Epsilon);
    assert(geom::computeDistance(points[1], Vertex{{0.25, 0.25,
                                 0.1}}) < Epsilon);
    assert(geom::computeDistance(points[2], Vertex{{0.25, 0.25,
                                 0}}) < Epsilon);
  }

  return 0;
}
