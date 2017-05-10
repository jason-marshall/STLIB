// -*- C++ -*-

#include "stlib/geom/mesh/iss/PointsOnManifold.h"

#include <cassert>

using namespace stlib;

int
main()
{
  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());

  //--------------------------------------------------------------------------
  // 2-D space, 1-D mesh, 1st degree splines.
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSet<2, 1> ISS;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::PointsOnManifold<2, 1, 1> POM;
    typedef POM::Vertex Vertex;

    std::vector<Vertex> vertices(4);
    vertices[0] = Vertex{{0., 0.}};
    vertices[1] = Vertex{{1., 0.}};
    vertices[2] = Vertex{{1., 1.}};
    vertices[3] = Vertex{{0., 1.}};

    std::vector<IndexedSimplex> indexedSimplices(4);
    indexedSimplices[0] = IndexedSimplex{{0, 1}};
    indexedSimplices[1] = IndexedSimplex{{1, 2}};
    indexedSimplices[2] = IndexedSimplex{{2, 3}};
    indexedSimplices[3] = IndexedSimplex{{3, 0}};

    std::array<std::size_t, 2> corners = {{0, 1}};

    ISS iss(vertices, indexedSimplices);

    POM x(iss, corners.begin(), corners.end());

    //
    // Accessors.
    //

    // Surface/Corner feature.
    for (std::size_t i = 0; i != vertices.size(); ++i) {
      if (std::count(corners.begin(), corners.end(), i)) {
        assert(! x.isVertexASurfaceFeature(i));
        assert(x.isVertexACornerFeature(i));
      }
      else {
        assert(x.isVertexASurfaceFeature(i));
        assert(! x.isVertexACornerFeature(i));
      }
    }

    // Has point.
    assert(!x.hasPoint(0));
    assert(!x.hasPoint(std::size_t(-1)));
    assert(!x.hasPoint(100));

    //
    // Insert points.
    //

    // The points.
    //  --4--
    //  |   |
    //  5   3
    //  |   |
    //  0-1-2

    // Insert at vertex.
    {
      x.insertAtVertex(0, 0);
    }
    {
      x.insertAtVertex(2, 1);
    }
    // Insert near point.
    {
      const Vertex location = x.insertNearPoint(1, Vertex{{0.5, -1.}},
                              0);
      assert(geom::computeDistance(location, Vertex{{0.5,
                                   0.}}) <= eps);
    }
    {
      const Vertex location = x.insertNearPoint(3, Vertex{{2., 0.5}},
                              2);
      assert(geom::computeDistance(location, Vertex{{1.,
                                   0.5}}) <= eps);
    }
    {
      const Vertex location = x.insertNearPoint(4, Vertex{{0.5, 0.75}},
                              3);
      assert(geom::computeDistance(location, Vertex{{0.5,
                                   1.}}) <= eps);
    }
    {
      const Vertex location = x.insertNearPoint(5, Vertex{{-1., 0.5}},
                              4);
      assert(geom::computeDistance(location, Vertex{{0.,
                                   0.5}}) <= eps);
    }

    //
    // Has point.
    //
    assert(x.hasPoint(0));
    assert(x.hasPoint(1));
    assert(x.hasPoint(2));
    assert(x.hasPoint(3));
    assert(x.hasPoint(4));
    assert(x.hasPoint(5));
    assert(!x.hasPoint(100));

    //
    // Is on Surface.
    //
    assert(! x.isOnSurface(0));
    assert(x.isOnSurface(1));
    assert(! x.isOnSurface(2));
    assert(x.isOnSurface(3));
    assert(x.isOnSurface(4));
    assert(x.isOnSurface(5));

    //
    // Is on corner.
    //
    assert(x.isOnCorner(0));
    assert(! x.isOnCorner(1));
    assert(x.isOnCorner(2));
    assert(! x.isOnCorner(3));
    assert(! x.isOnCorner(4));
    assert(! x.isOnCorner(5));

    //
    // Simplex index.
    //
    assert(x.getSimplexIndex(1) == 0);
    assert(x.getSimplexIndex(3) == 1);
    assert(x.getSimplexIndex(4) == 2);
    assert(x.getSimplexIndex(5) == 3);

    //
    // Vertex index.
    //
    assert(x.getVertexIndex(0) == 0);
    assert(x.getVertexIndex(2) == 1);

    //
    // Closest points.
    //

    // Corners.
    {
      // Closest point for a corner.
      Vertex location = x.computeClosestPoint(0, Vertex{{2., 3.}});
      assert(location == vertices[0]);
    }
    {
      // Closest point for a corner.
      Vertex location = x.computeClosestPoint(2, Vertex{{5., 7.}});
      assert(location == vertices[1]);
    }

    // Simplex with two adjacent simplices, two corners.
    {
      // Closest point on a surface.
      // Same simplex, interior.
      Vertex location = x.computeClosestPoint(1, Vertex{{0.5, 0.25}});
      assert(geom::computeDistance(location, Vertex{{0.5,
                                   0.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(1, Vertex{{-1., 2.}});
      assert(geom::computeDistance(location, Vertex{{0., 0.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(1, Vertex{{2., 2.}});
      assert(geom::computeDistance(location, Vertex{{1., 0.}}) <= eps);
    }

    // Simplex with two adjacent simplices, one corner.
    {
      // Closest point on a surface.
      // Same simplex, interior.
      Vertex location = x.computeClosestPoint(3, Vertex{{2., 0.75}});
      assert(geom::computeDistance(location, Vertex{{1.,
                                   0.75}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(3, Vertex{{2., -1.}});
      assert(geom::computeDistance(location, Vertex{{1., 0.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(3, Vertex{{2., 2.}});
      assert(geom::computeDistance(location, Vertex{{1., 1.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, interior.
      Vertex location = x.computeClosestPoint(3, Vertex{{0.5, 2.}});
      assert(geom::computeDistance(location, Vertex{{0.5,
                                   1.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, end point.
      Vertex location = x.computeClosestPoint(3, Vertex{{-1., 0.}});
      assert(geom::computeDistance(location, Vertex{{0., 1.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Incident corner.
      Vertex location = x.computeClosestPoint(3, Vertex{{0.5, -1.}});
      assert(geom::computeDistance(location, Vertex{{1., 0.}}) <= eps);
    }

    // Simplex with two adjacent simplices, no corners.
    {
      // Closest point on a surface.
      // Same simplex, interior.
      Vertex location = x.computeClosestPoint(4, Vertex{{0.75, 2.}});
      assert(geom::computeDistance(location, Vertex{{0.75,
                                   1.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(4, Vertex{{2., 3.}});
      assert(geom::computeDistance(location, Vertex{{1., 1.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(4, Vertex{{-1., 3.}});
      assert(geom::computeDistance(location, Vertex{{0., 1.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, interior.
      Vertex location = x.computeClosestPoint(4, Vertex{{2., 0.5}});
      assert(geom::computeDistance(location, Vertex{{1.,
                                   0.5}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, interior.
      Vertex location = x.computeClosestPoint(4, Vertex{{-1., 0.1}});
      assert(geom::computeDistance(location, Vertex{{0.,
                                   0.1}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, end point.
      Vertex location = x.computeClosestPoint(4, Vertex{{2., -1.}});
      assert(geom::computeDistance(location, Vertex{{1., 0.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, end point.
      Vertex location = x.computeClosestPoint(4, Vertex{{2., 1.}});
      assert(geom::computeDistance(location, Vertex{{1., 1.}}) <= eps);
    }

    //
    // Update points.
    //
    {
      Vertex location = x.computeClosestPoint(3, Vertex{{0.75, 1.}});
      assert(geom::computeDistance(location, Vertex{{0.75,
                                   1.}}) <= eps);
      x.updatePoint();
      assert(x.isOnSurface(3));
      assert(x.getSimplexIndex(3) == 2);
    }

    //
    // Erase points.
    //

    // Erase the points on edges.
    x.erase(1);
    x.erase(3);
    x.erase(4);
    x.erase(5);

    // Has point.
    assert(x.hasPoint(0));
    assert(! x.hasPoint(1));
    assert(x.hasPoint(2));
    assert(! x.hasPoint(3));
    assert(! x.hasPoint(4));
    assert(! x.hasPoint(5));
    assert(! x.hasPoint(100));

    // Erase the points on vertices.
    x.erase(0);
    x.erase(2);

    // Has point.
    assert(! x.hasPoint(0));
    assert(! x.hasPoint(1));
    assert(! x.hasPoint(2));
    assert(! x.hasPoint(3));
    assert(! x.hasPoint(4));
    assert(! x.hasPoint(5));
    assert(! x.hasPoint(100));
  }

  //--------------------------------------------------------------------------
  // 2-D space, 1-D mesh, 1st degree splines.
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSet<2, 1> ISS;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::PointsOnManifold<2, 1, 1> POM;
    typedef POM::Vertex Vertex;

    std::vector<Vertex> vertices(4);
    vertices[0] = Vertex{{0., 0.}};
    vertices[1] = Vertex{{1., 0.}};
    vertices[2] = Vertex{{2, 0}};
    vertices[3] = Vertex{{3, 0}};

    std::vector<IndexedSimplex> indexedSimplices(3);
    indexedSimplices[0] = IndexedSimplex{{0, 1}};
    indexedSimplices[1] = IndexedSimplex{{1, 2}};
    indexedSimplices[2] = IndexedSimplex{{2, 3}};

    ISS iss(vertices, indexedSimplices);

    POM x(iss);

    //
    // Accessors.
    //

    // Surface/Corner feature.
    assert(! x.isVertexASurfaceFeature(0));
    assert(x.isVertexASurfaceFeature(1));
    assert(x.isVertexASurfaceFeature(2));
    assert(! x.isVertexASurfaceFeature(3));

    assert(x.isVertexACornerFeature(0));
    assert(! x.isVertexACornerFeature(1));
    assert(! x.isVertexACornerFeature(2));
    assert(x.isVertexACornerFeature(3));

    // Has point.
    assert(!x.hasPoint(0));
    assert(!x.hasPoint(100));

    //
    // Insert points.
    //

    // The vertices and the points.
    // 0---1---2---3
    // 0 4 1 5 2 6 3

    // Insert at the vertices.
    x.insertAtVertices();

    // Insert in the simplices.
    {
      const Vertex location = x.insertNearPoint(4, Vertex{{0.5, -1.}},
                              0);
      assert(geom::computeDistance(location, Vertex{{0.5,
                                   0.}}) <= eps);
    }
    {
      const Vertex location = x.insertNearPoint(5, Vertex{{1.5, -1.}},
                              1);
      assert(geom::computeDistance(location, Vertex{{1.5,
                                   0.}}) <= eps);
    }
    {
      const Vertex location = x.insertNearPoint(6, Vertex{{2.5, -1.}},
                              2);
      assert(geom::computeDistance(location, Vertex{{2.5,
                                   0.}}) <= eps);
    }

    //
    // Has point.
    //
    assert(x.hasPoint(0));
    assert(x.hasPoint(1));
    assert(x.hasPoint(2));
    assert(x.hasPoint(3));
    assert(x.hasPoint(4));
    assert(x.hasPoint(5));
    assert(x.hasPoint(6));
    assert(!x.hasPoint(100));

    //
    // Is on Surface.
    //
    assert(! x.isOnSurface(0));
    assert(x.isOnSurface(1));
    assert(x.isOnSurface(2));
    assert(! x.isOnSurface(3));
    assert(x.isOnSurface(4));
    assert(x.isOnSurface(5));
    assert(x.isOnSurface(6));

    //
    // Is on corner.
    //
    assert(x.isOnCorner(0));
    assert(! x.isOnCorner(1));
    assert(! x.isOnCorner(2));
    assert(x.isOnCorner(3));
    assert(! x.isOnCorner(4));
    assert(! x.isOnCorner(5));
    assert(! x.isOnCorner(6));

    //
    // Vertex index.
    //
    assert(x.getVertexIndex(0) == 0);
    assert(x.getVertexIndex(3) == 3);

    //
    // Simplex index.
    //
    assert(x.getSimplexIndex(4) == 0);
    assert(x.getSimplexIndex(5) == 1);
    assert(x.getSimplexIndex(6) == 2);
    assert(x.getSimplexIndex(1) == 0 || x.getSimplexIndex(1) == 1);
    assert(x.getSimplexIndex(2) == 1 || x.getSimplexIndex(2) == 2);

    //
    // Closest points.
    //

    // Corners.
    {
      // Closest point for a corner.
      Vertex location = x.computeClosestPoint(0, Vertex{{2., 3.}});
      assert(location == vertices[0]);
    }
    {
      // Closest point for a corner.
      Vertex location = x.computeClosestPoint(3, Vertex{{5., 7.}});
      assert(location == vertices[3]);
    }

    // Simplex with one adjacent simplex, one corner.
    {
      // Closest point on a surface.
      // Same simplex, interior.
      Vertex location = x.computeClosestPoint(4, Vertex{{0.75, 23.}});
      assert(geom::computeDistance(location, Vertex{{0.75,
                                   0.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(4, Vertex{{-1., 23.}});
      assert(geom::computeDistance(location, Vertex{{0., 0.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, interior.
      Vertex location = x.computeClosestPoint(4, Vertex{{1.5, 23.}});
      assert(geom::computeDistance(location, Vertex{{1.5,
                                   0.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, end point.
      Vertex location = x.computeClosestPoint(4, Vertex{{5., 23.}});
      assert(geom::computeDistance(location, Vertex{{2., 0.}}) <= eps);
    }

    // Surface vertex.
    {
      // Interior.
      Vertex location = x.computeClosestPoint(1, Vertex{{0.5, 23.}});
      assert(geom::computeDistance(location, Vertex{{0.5,
                                   0.}}) <= eps);
    }
    {
      // Interior.
      Vertex location = x.computeClosestPoint(1, Vertex{{1.75, 23.}});
      assert(geom::computeDistance(location, Vertex{{1.75,
                                   0.}}) <= eps);
    }
    {
      // End point.
      Vertex location = x.computeClosestPoint(1, Vertex{{-1., 23.}});
      assert(geom::computeDistance(location, Vertex{{0., 0.}}) <= eps);
    }
    {
      // End point.
      Vertex location = x.computeClosestPoint(1, Vertex{{3., 23.}});
      assert(geom::computeDistance(location, Vertex{{2., 0.}}) <= eps);
    }

    //
    // Update points.
    //
    {
      Vertex location = x.computeClosestPoint(4, Vertex{{1.5, 1.}});
      assert(geom::computeDistance(location, Vertex{{1.5,
                                   0.}}) <= eps);
      x.updatePoint();
      assert(x.isOnSurface(4));
      assert(x.getSimplexIndex(4) == 1);
    }
    {
      Vertex location = x.computeClosestPoint(1, Vertex{{1.5, 1.}});
      assert(geom::computeDistance(location, Vertex{{1.5,
                                   0.}}) <= eps);
      x.updatePoint();
      assert(x.isOnSurface(1));
      assert(x.getSimplexIndex(1) == 1);
    }

    //
    // Erase points.
    //

    // Erase the points on edges.
    x.erase(4);
    x.erase(5);
    x.erase(6);

    // Has point.
    assert(x.hasPoint(0));
    assert(x.hasPoint(1));
    assert(x.hasPoint(2));
    assert(x.hasPoint(3));
    assert(! x.hasPoint(4));
    assert(! x.hasPoint(5));
    assert(! x.hasPoint(6));
    assert(! x.hasPoint(100));

    // Erase the points on vertices.
    x.erase(0);
    x.erase(1);
    x.erase(2);
    x.erase(3);

    // Has point.
    assert(! x.hasPoint(0));
    assert(! x.hasPoint(1));
    assert(! x.hasPoint(2));
    assert(! x.hasPoint(3));
    assert(! x.hasPoint(4));
    assert(! x.hasPoint(5));
    assert(! x.hasPoint(6));
    assert(! x.hasPoint(100));
  }

  //--------------------------------------------------------------------------
  // 2-D space, 1-D mesh, 1st degree splines.
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSet<2, 1> ISS;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::PointsOnManifold<2, 1, 1> POM;
    typedef POM::Vertex Vertex;

    std::vector<Vertex> vertices(4);
    vertices[0] = Vertex{{0., 0.}};
    vertices[1] = Vertex{{1., 0.}};
    vertices[2] = Vertex{{2., 0.}};
    vertices[3] = Vertex{{3., 0.}};

    std::vector<IndexedSimplex> indexedSimplices(3);
    indexedSimplices[0] = IndexedSimplex{{0, 1}};
    indexedSimplices[1] = IndexedSimplex{{1, 2}};
    indexedSimplices[2] = IndexedSimplex{{2, 3}};

    ISS iss(vertices, indexedSimplices);

    POM x(iss);

    //
    // Accessors.
    //

    // Surface/Corner feature.
    assert(! x.isVertexASurfaceFeature(0));
    assert(x.isVertexASurfaceFeature(1));
    assert(x.isVertexASurfaceFeature(2));
    assert(! x.isVertexASurfaceFeature(3));

    assert(x.isVertexACornerFeature(0));
    assert(! x.isVertexACornerFeature(1));
    assert(! x.isVertexACornerFeature(2));
    assert(x.isVertexACornerFeature(3));

    // Has point.
    assert(!x.hasPoint(0));
    assert(!x.hasPoint(100));

    //
    // Insert points.
    //

    std::vector<Vertex> points(vertices);
    {
      const double cornerEpsilon = x.getMaxCornerDistance() / 2.0;
      points[0][0] += cornerEpsilon;
      points[1][1] += cornerEpsilon;
      points[2][1] -= cornerEpsilon;
      points[3][0] -= cornerEpsilon;
    }

    // The vertices and the points.
    // 0---1---2---3
    // 0   1   2   3

    // Insert at the vertices.
    x.insert(points.begin(), points.end(), points.begin());

    // Check the closest points where they were inserted.
    for (std::size_t i = 0; i != points.size(); ++i) {
      assert(geom::computeDistance(points[i], vertices[i]) <=
             10.0 * std::numeric_limits<double>::epsilon());
    }

    //
    // Has point.
    //
    assert(x.hasPoint(0));
    assert(x.hasPoint(1));
    assert(x.hasPoint(2));
    assert(x.hasPoint(3));
    assert(!x.hasPoint(100));

    //
    // Is on Surface.
    //
    assert(! x.isOnSurface(0));
    assert(x.isOnSurface(1));
    assert(x.isOnSurface(2));
    assert(! x.isOnSurface(3));

    //
    // Is on corner.
    //
    assert(x.isOnCorner(0));
    assert(! x.isOnCorner(1));
    assert(! x.isOnCorner(2));
    assert(x.isOnCorner(3));

    //
    // Vertex index.
    //
    assert(x.getVertexIndex(0) == 0);
    assert(x.getVertexIndex(3) == 3);

    //
    // Simplex index.
    //
    assert(x.getSimplexIndex(1) == 0 || x.getSimplexIndex(1) == 1);
    assert(x.getSimplexIndex(2) == 1 || x.getSimplexIndex(2) == 2);
  }















  //--------------------------------------------------------------------------
  // 3-D space, 1-D mesh, 1st degree splines.
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSet<3, 1> ISS;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::PointsOnManifold<3, 1, 1> POM;
    typedef POM::Vertex Vertex;

    std::vector<Vertex> vertices(4);
    vertices[0] = Vertex{{0., 0., 0.}};
    vertices[1] = Vertex{{1., 0., 0.}};
    vertices[2] = Vertex{{1., 1., 0.}};
    vertices[3] = Vertex{{0., 1., 0.}};

    std::vector<IndexedSimplex> indexedSimplices(4);
    indexedSimplices[0] = IndexedSimplex{{0, 1}};
    indexedSimplices[1] = IndexedSimplex{{1, 2}};
    indexedSimplices[2] = IndexedSimplex{{2, 3}};
    indexedSimplices[3] = IndexedSimplex{{3, 0}};

    std::array<std::size_t, 2> corners = {{0, 1}};

    ISS iss(vertices, indexedSimplices);

    POM x(iss, corners.begin(), corners.end());

    //
    // Accessors.
    //

    // Surface/Corner feature.
    for (std::size_t i = 0; i != vertices.size(); ++i) {
      if (std::count(corners.begin(), corners.end(), i)) {
        assert(! x.isVertexASurfaceFeature(i));
        assert(x.isVertexACornerFeature(i));
      }
      else {
        assert(x.isVertexASurfaceFeature(i));
        assert(! x.isVertexACornerFeature(i));
      }
    }

    // Has point.
    assert(!x.hasPoint(0));
    assert(!x.hasPoint(100));

    //
    // Insert points.
    //

    // The points.
    //  --4--
    //  |   |
    //  5   3
    //  |   |
    //  0-1-2

    // Insert at vertex.
    {
      x.insertAtVertex(0, 0);
    }
    {
      x.insertAtVertex(2, 1);
    }
    // Insert near point.
    {
      const Vertex location = x.insertNearPoint(1, Vertex{{0.5, -1.,
                              0.}}, 0);
      assert(geom::computeDistance(location, Vertex{{0.5, 0.,
                                   0.}}) <= eps);
    }
    {
      const Vertex location = x.insertNearPoint(3, Vertex{{2, 0.5, 0}},
                                                2);
      assert(geom::computeDistance(location, Vertex{{1, 0.5,
                                   0}}) <= eps);
    }
    {
      const Vertex location = x.insertNearPoint(4, Vertex{{0.5, 0.75,
                              0}}, 3);
      assert(geom::computeDistance(location, Vertex{{0.5, 1,
                                   0}}) <= eps);
    }
    {
      const Vertex location = x.insertNearPoint(5, Vertex{{-1, 0.5,
                              0}}, 4);
      assert(geom::computeDistance(location, Vertex{{0, 0.5,
                                   0}}) <= eps);
    }

    //
    // Has point.
    //
    assert(x.hasPoint(0));
    assert(x.hasPoint(1));
    assert(x.hasPoint(2));
    assert(x.hasPoint(3));
    assert(x.hasPoint(4));
    assert(x.hasPoint(5));
    assert(!x.hasPoint(100));

    //
    // Is on Surface.
    //
    assert(! x.isOnSurface(0));
    assert(x.isOnSurface(1));
    assert(! x.isOnSurface(2));
    assert(x.isOnSurface(3));
    assert(x.isOnSurface(4));
    assert(x.isOnSurface(5));

    //
    // Is on corner.
    //
    assert(x.isOnCorner(0));
    assert(! x.isOnCorner(1));
    assert(x.isOnCorner(2));
    assert(! x.isOnCorner(3));
    assert(! x.isOnCorner(4));
    assert(! x.isOnCorner(5));

    //
    // Simplex index.
    //
    assert(x.getSimplexIndex(1) == 0);
    assert(x.getSimplexIndex(3) == 1);
    assert(x.getSimplexIndex(4) == 2);
    assert(x.getSimplexIndex(5) == 3);

    //
    // Vertex index.
    //
    assert(x.getVertexIndex(0) == 0);
    assert(x.getVertexIndex(2) == 1);

    //
    // Closest points.
    //

    // Corners.
    {
      // Closest point for a corner.
      Vertex location = x.computeClosestPoint(0, Vertex{{2, 3, 0}});
      assert(location == vertices[0]);
    }
    {
      // Closest point for a corner.
      Vertex location = x.computeClosestPoint(2, Vertex{{5, 7, 0}});
      assert(location == vertices[1]);
    }

    // Simplex with two adjacent simplices, two corners.
    {
      // Closest point on a surface.
      // Same simplex, interior.
      Vertex location = x.computeClosestPoint(1, Vertex{{0.5, 0.25,
                                              0}});
      assert(geom::computeDistance(location, Vertex{{0.5, 0,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(1, Vertex{{-1, 2, 0}});
      assert(geom::computeDistance(location, Vertex{{0., 0.,
                                   0.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(1, Vertex{{2, 2, 0}});
      assert(geom::computeDistance(location, Vertex{{1, 0,
                                   0}}) <= eps);
    }

    // Simplex with two adjacent simplices, one corner.
    {
      // Closest point on a surface.
      // Same simplex, interior.
      Vertex location = x.computeClosestPoint(3, Vertex{{2, 0.75, 0}});
      assert(geom::computeDistance(location, Vertex{{1, 0.75,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(3, Vertex{{2, -1, 0}});
      assert(geom::computeDistance(location, Vertex{{1, 0,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(3, Vertex{{2, 2, 0}});
      assert(geom::computeDistance(location, Vertex{{1, 1,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, interior.
      Vertex location = x.computeClosestPoint(3, Vertex{{0.5, 2, 0}});
      assert(geom::computeDistance(location, Vertex{{0.5, 1,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, end point.
      Vertex location = x.computeClosestPoint(3, Vertex{{-1, 0, 0}});
      assert(geom::computeDistance(location, Vertex{{0, 1,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Incident corner.
      Vertex location = x.computeClosestPoint(3, Vertex{{0.5, -1, 0}});
      assert(geom::computeDistance(location, Vertex{{1, 0,
                                   0}}) <= eps);
    }

    // Simplex with two adjacent simplices, no corners.
    {
      // Closest point on a surface.
      // Same simplex, interior.
      Vertex location = x.computeClosestPoint(4, Vertex{{0.75, 2, 0}});
      assert(geom::computeDistance(location, Vertex{{0.75, 1,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(4, Vertex{{2, 3, 0}});
      assert(geom::computeDistance(location, Vertex{{1, 1,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(4, Vertex{{-1, 3, 0}});
      assert(geom::computeDistance(location, Vertex{{0, 1,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, interior.
      Vertex location = x.computeClosestPoint(4, Vertex{{2, 0.5, 0}});
      assert(geom::computeDistance(location, Vertex{{1, 0.5,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, interior.
      Vertex location = x.computeClosestPoint(4, Vertex{{-1, 0.1, 0}});
      assert(geom::computeDistance(location, Vertex{{0, 0.1,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, end point.
      Vertex location = x.computeClosestPoint(4, Vertex{{2, -1, 0}});
      assert(geom::computeDistance(location, Vertex{{1, 0,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, end point.
      Vertex location = x.computeClosestPoint(4, Vertex{{2, 1, 0}});
      assert(geom::computeDistance(location, Vertex{{1, 1,
                                   0}}) <= eps);
    }

    //
    // Update points.
    //
    {
      Vertex location = x.computeClosestPoint(3, Vertex{{0.75, 1, 0}});
      assert(geom::computeDistance(location, Vertex{{0.75, 1,
                                   0}}) <= eps);
      x.updatePoint();
      assert(x.isOnSurface(3));
      assert(x.getSimplexIndex(3) == 2);
    }

    //
    // Erase points.
    //

    // Erase the points on edges.
    x.erase(1);
    x.erase(3);
    x.erase(4);
    x.erase(5);

    // Has point.
    assert(x.hasPoint(0));
    assert(! x.hasPoint(1));
    assert(x.hasPoint(2));
    assert(! x.hasPoint(3));
    assert(! x.hasPoint(4));
    assert(! x.hasPoint(5));
    assert(! x.hasPoint(100));

    // Erase the points on vertices.
    x.erase(0);
    x.erase(2);

    // Has point.
    assert(! x.hasPoint(0));
    assert(! x.hasPoint(1));
    assert(! x.hasPoint(2));
    assert(! x.hasPoint(3));
    assert(! x.hasPoint(4));
    assert(! x.hasPoint(5));
    assert(! x.hasPoint(100));
  }

  //--------------------------------------------------------------------------
  // 2-D space, 1-D mesh, 1st degree splines.
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSet<3, 1> ISS;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::PointsOnManifold<3, 1, 1> POM;
    typedef POM::Vertex Vertex;

    std::vector<Vertex> vertices(4);
    vertices[0] = Vertex{{0., 0., 0.}};
    vertices[1] = Vertex{{1, 0, 0}};
    vertices[2] = Vertex{{2, 0, 0}};
    vertices[3] = Vertex{{3, 0, 0}};

    std::vector<IndexedSimplex> indexedSimplices(3);
    indexedSimplices[0] = IndexedSimplex{{0, 1}};
    indexedSimplices[1] = IndexedSimplex{{1, 2}};
    indexedSimplices[2] = IndexedSimplex{{2, 3}};

    ISS iss(vertices, indexedSimplices);

    POM x(iss);

    //
    // Accessors.
    //

    // Surface/Corner feature.
    assert(! x.isVertexASurfaceFeature(0));
    assert(x.isVertexASurfaceFeature(1));
    assert(x.isVertexASurfaceFeature(2));
    assert(! x.isVertexASurfaceFeature(3));

    assert(x.isVertexACornerFeature(0));
    assert(! x.isVertexACornerFeature(1));
    assert(! x.isVertexACornerFeature(2));
    assert(x.isVertexACornerFeature(3));

    // Has point.
    assert(!x.hasPoint(0));
    assert(!x.hasPoint(100));

    //
    // Insert points.
    //

    // The vertices and the points.
    // 0---1---2---3
    // 0 4 1 5 2 6 3

    // Insert at the vertices.
    x.insertAtVertices();

    // Insert in the simplices.
    {
      const Vertex location = x.insertNearPoint(4, Vertex{{0.5, -1,
                              0}}, 0);
      assert(geom::computeDistance(location, Vertex{{0.5, 0,
                                   0}}) <= eps);
    }
    {
      const Vertex location = x.insertNearPoint(5, Vertex{{1.5, -1,
                              0}}, 1);
      assert(geom::computeDistance(location, Vertex{{1.5, 0,
                                   0}}) <= eps);
    }
    {
      const Vertex location = x.insertNearPoint(6, Vertex{{2.5, -1,
                              0}}, 2);
      assert(geom::computeDistance(location, Vertex{{2.5, 0,
                                   0}}) <= eps);
    }

    //
    // Has point.
    //
    assert(x.hasPoint(0));
    assert(x.hasPoint(1));
    assert(x.hasPoint(2));
    assert(x.hasPoint(3));
    assert(x.hasPoint(4));
    assert(x.hasPoint(5));
    assert(x.hasPoint(6));
    assert(!x.hasPoint(100));

    //
    // Is on Surface.
    //
    assert(! x.isOnSurface(0));
    assert(x.isOnSurface(1));
    assert(x.isOnSurface(2));
    assert(! x.isOnSurface(3));
    assert(x.isOnSurface(4));
    assert(x.isOnSurface(5));
    assert(x.isOnSurface(6));

    //
    // Is on corner.
    //
    assert(x.isOnCorner(0));
    assert(! x.isOnCorner(1));
    assert(! x.isOnCorner(2));
    assert(x.isOnCorner(3));
    assert(! x.isOnCorner(4));
    assert(! x.isOnCorner(5));
    assert(! x.isOnCorner(6));

    //
    // Vertex index.
    //
    assert(x.getVertexIndex(0) == 0);
    assert(x.getVertexIndex(3) == 3);

    //
    // Simplex index.
    //
    assert(x.getSimplexIndex(4) == 0);
    assert(x.getSimplexIndex(5) == 1);
    assert(x.getSimplexIndex(6) == 2);
    assert(x.getSimplexIndex(1) == 0 || x.getSimplexIndex(1) == 1);
    assert(x.getSimplexIndex(2) == 1 || x.getSimplexIndex(2) == 2);

    //
    // Closest points.
    //

    // Corners.
    {
      // Closest point for a corner.
      Vertex location = x.computeClosestPoint(0, Vertex{{2, 3, 0}});
      assert(location == vertices[0]);
    }
    {
      // Closest point for a corner.
      Vertex location = x.computeClosestPoint(3, Vertex{{5, 7, 0}});
      assert(location == vertices[3]);
    }

    // Simplex with one adjacent simplex, one corner.
    {
      // Closest point on a surface.
      // Same simplex, interior.
      Vertex location = x.computeClosestPoint(4, Vertex{{0.75, 23,
                                              0}});
      assert(geom::computeDistance(location, Vertex{{0.75, 0,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Same simplex, end point.
      Vertex location = x.computeClosestPoint(4, Vertex{{-1, 23, 0}});
      assert(geom::computeDistance(location, Vertex{{0., 0.,
                                   0.}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, interior.
      Vertex location = x.computeClosestPoint(4, Vertex{{1.5, 23, 0}});
      assert(geom::computeDistance(location, Vertex{{1.5, 0,
                                   0}}) <= eps);
    }
    {
      // Closest point on a surface.
      // Adjacent simplex, end point.
      Vertex location = x.computeClosestPoint(4, Vertex{{5, 23, 0}});
      assert(geom::computeDistance(location, Vertex{{2, 0,
                                   0}}) <= eps);
    }

    // Surface vertex.
    {
      // Interior.
      Vertex location = x.computeClosestPoint(1, Vertex{{0.5, 23, 0}});
      assert(geom::computeDistance(location, Vertex{{0.5, 0,
                                   0}}) <= eps);
    }
    {
      // Interior.
      Vertex location = x.computeClosestPoint(1, Vertex{{1.75, 23,
                                              0}});
      assert(geom::computeDistance(location, Vertex{{1.75, 0,
                                   0}}) <= eps);
    }
    {
      // End point.
      Vertex location = x.computeClosestPoint(1, Vertex{{-1, 23, 0}});
      assert(geom::computeDistance(location, Vertex{{0., 0.,
                                   0.}}) <= eps);
    }
    {
      // End point.
      Vertex location = x.computeClosestPoint(1, Vertex{{3, 23, 0}});
      assert(geom::computeDistance(location, Vertex{{2, 0,
                                   0}}) <= eps);
    }

    //
    // Update points.
    //
    {
      Vertex location = x.computeClosestPoint(4, Vertex{{1.5, 1, 0}});
      assert(geom::computeDistance(location, Vertex{{1.5, 0,
                                   0}}) <= eps);
      x.updatePoint();
      assert(x.isOnSurface(4));
      assert(x.getSimplexIndex(4) == 1);
    }
    {
      Vertex location = x.computeClosestPoint(1, Vertex{{1.5, 1, 0}});
      assert(geom::computeDistance(location, Vertex{{1.5, 0,
                                   0}}) <= eps);
      x.updatePoint();
      assert(x.isOnSurface(1));
      assert(x.getSimplexIndex(1) == 1);
    }

    //
    // Erase points.
    //

    // Erase the points on edges.
    x.erase(4);
    x.erase(5);
    x.erase(6);

    // Has point.
    assert(x.hasPoint(0));
    assert(x.hasPoint(1));
    assert(x.hasPoint(2));
    assert(x.hasPoint(3));
    assert(! x.hasPoint(4));
    assert(! x.hasPoint(5));
    assert(! x.hasPoint(6));
    assert(! x.hasPoint(100));

    // Erase the points on vertices.
    x.erase(0);
    x.erase(1);
    x.erase(2);
    x.erase(3);

    // Has point.
    assert(! x.hasPoint(0));
    assert(! x.hasPoint(1));
    assert(! x.hasPoint(2));
    assert(! x.hasPoint(3));
    assert(! x.hasPoint(4));
    assert(! x.hasPoint(5));
    assert(! x.hasPoint(6));
    assert(! x.hasPoint(100));
  }

  //--------------------------------------------------------------------------
  // 2-D space, 1-D mesh, 1st degree splines.
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSet<3, 1> ISS;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::PointsOnManifold<3, 1, 1> POM;
    typedef POM::Vertex Vertex;

    std::vector<Vertex> vertices(4);
    vertices[0] = Vertex{{0., 0., 0.}};
    vertices[1] = Vertex{{1, 0, 0}};
    vertices[2] = Vertex{{2, 0, 0}};
    vertices[3] = Vertex{{3, 0, 0}};

    std::vector<IndexedSimplex> indexedSimplices(3);
    indexedSimplices[0] = IndexedSimplex{{0, 1}};
    indexedSimplices[1] = IndexedSimplex{{1, 2}};
    indexedSimplices[2] = IndexedSimplex{{2, 3}};

    ISS iss(vertices, indexedSimplices);

    POM x(iss);

    //
    // Accessors.
    //

    // Surface/Corner feature.
    assert(! x.isVertexASurfaceFeature(0));
    assert(x.isVertexASurfaceFeature(1));
    assert(x.isVertexASurfaceFeature(2));
    assert(! x.isVertexASurfaceFeature(3));

    assert(x.isVertexACornerFeature(0));
    assert(! x.isVertexACornerFeature(1));
    assert(! x.isVertexACornerFeature(2));
    assert(x.isVertexACornerFeature(3));

    // Has point.
    assert(!x.hasPoint(0));
    assert(!x.hasPoint(100));

    //
    // Insert points.
    //

    std::vector<Vertex> points(vertices);
    {
      const double cornerEpsilon = x.getMaxCornerDistance() / 2.0;
      points[0][0] += cornerEpsilon;
      points[1][1] += cornerEpsilon;
      points[2][2] -= cornerEpsilon;
      points[3][0] -= cornerEpsilon;
    }

    // The vertices and the points.
    // 0---1---2---3
    // 0   1   2   3

    // Insert at the vertices.
    x.insert(points.begin(), points.end(), points.begin());

    // Check the closest points where they were inserted.
    for (std::size_t i = 0; i != points.size(); ++i) {
      assert(geom::computeDistance(points[i], vertices[i]) <=
             10.0 * std::numeric_limits<double>::epsilon());
    }

    //
    // Has point.
    //
    assert(x.hasPoint(0));
    assert(x.hasPoint(1));
    assert(x.hasPoint(2));
    assert(x.hasPoint(3));
    assert(!x.hasPoint(100));

    //
    // Is on Surface.
    //
    assert(! x.isOnSurface(0));
    assert(x.isOnSurface(1));
    assert(x.isOnSurface(2));
    assert(! x.isOnSurface(3));

    //
    // Is on corner.
    //
    assert(x.isOnCorner(0));
    assert(! x.isOnCorner(1));
    assert(! x.isOnCorner(2));
    assert(x.isOnCorner(3));

    //
    // Vertex index.
    //
    assert(x.getVertexIndex(0) == 0);
    assert(x.getVertexIndex(3) == 3);

    //
    // Simplex index.
    //
    assert(x.getSimplexIndex(1) == 0 || x.getSimplexIndex(1) == 1);
    assert(x.getSimplexIndex(2) == 1 || x.getSimplexIndex(2) == 2);
  }















  //--------------------------------------------------------------------------
  // 3-D space, 2-D mesh, 1st degree splines.
  // No edges or corners.
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSetIncAdj<3, 2> ISS;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::PointsOnManifold<3, 2, 1> POM;
    typedef POM::Vertex Vertex;

    std::vector<Vertex> vertices(4);
    vertices[0] = Vertex{{0., 0., 0.}};
    vertices[1] = Vertex{{1, 0, 0}};
    vertices[2] = Vertex{{0, 1, 0}};
    vertices[3] = Vertex{{0, 0, 1}};

    std::vector<IndexedSimplex> indexedSimplices(4);
    indexedSimplices[0] = IndexedSimplex{{1, 2, 3}};
    indexedSimplices[1] = IndexedSimplex{{0, 3, 2}};
    indexedSimplices[2] = IndexedSimplex{{0, 1, 3}};
    indexedSimplices[3] = IndexedSimplex{{0, 2, 1}};

    ISS iss(vertices, indexedSimplices);

    std::size_t* cornersBegin = 0;
    std::size_t* cornersEnd = 0;
    POM x(iss, iss.getFacesBeginning(), iss.getFacesBeginning(),
          cornersBegin, cornersEnd);

    //
    // Accessors.
    //

    // Surface/Corner feature.
    for (std::size_t i = 0; i != vertices.size(); ++i) {
      assert(x.isVertexASurfaceFeature(i));
      assert(! x.isVertexAnEdgeFeature(i));
      assert(! x.isVertexACornerFeature(i));
    }

    // Has point.
    assert(!x.hasPoint(0));
    assert(!x.hasPoint(100));

    //
    // Insert points at the vertices.
    //
    x.insertAtVertices();

    //
    // Has point.
    //
    assert(x.hasPoint(0));
    assert(x.hasPoint(1));
    assert(x.hasPoint(2));
    assert(x.hasPoint(3));
    assert(!x.hasPoint(100));

    //
    // Is on Surface.
    //
    assert(x.isOnSurface(0));
    assert(x.isOnSurface(1));
    assert(x.isOnSurface(2));
    assert(x.isOnSurface(3));

    //
    // Is on corner.
    //
    assert(! x.isOnCorner(0));
    assert(! x.isOnCorner(1));
    assert(! x.isOnCorner(2));
    assert(! x.isOnCorner(3));

    //
    // Simplex index.
    //
    assert(x.getSurfaceSimplexIndex(0) != 0);
    assert(x.getSurfaceSimplexIndex(1) != 1);
    assert(x.getSurfaceSimplexIndex(2) != 2);
    assert(x.getSurfaceSimplexIndex(3) != 3);

    //
    // Closest points.
    //

    // Surface.
    {
      // Closest point for a surface.
      Vertex location = x.computeClosestPoint(0, Vertex{{0., 0., 0.}});
      assert(geom::computeDistance(location, Vertex{{0., 0.,
                                   0.}}) <= eps);
      assert(location == vertices[0]);
    }

    //
    // Update points.
    //
    {
      Vertex location = x.computeClosestPoint(0, Vertex{{0.1, 0.1,
                                              -1}});
      assert(geom::computeDistance(location, Vertex{{0.1, 0.1,
                                   0}}) <= eps);
      x.updatePoint();
      assert(x.isOnSurface(0));
      assert(x.getSurfaceSimplexIndex(0) == 3);
    }

    //
    // Erase points.
    //

    x.erase(0);
    x.erase(1);
    x.erase(2);
    x.erase(3);

    // Has point.
    assert(! x.hasPoint(0));
    assert(! x.hasPoint(1));
    assert(! x.hasPoint(2));
    assert(! x.hasPoint(3));
    assert(! x.hasPoint(100));
  }







  //--------------------------------------------------------------------------
  // 3-D space, 2-D mesh, 1st degree splines.
  // No edges or corners.  Angle constructor.
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSetIncAdj<3, 2> ISS;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::PointsOnManifold<3, 2, 1> POM;
    typedef POM::Vertex Vertex;

    std::vector<Vertex> vertices(4);
    vertices[0] = Vertex{{0., 0., 0.}};
    vertices[1] = Vertex{{1, 0, 0}};
    vertices[2] = Vertex{{0, 1, 0}};
    vertices[3] = Vertex{{0, 0, 1}};

    std::vector<IndexedSimplex> indexedSimplices(4);
    indexedSimplices[0] = IndexedSimplex{{1, 2, 3}};
    indexedSimplices[1] = IndexedSimplex{{0, 3, 2}};
    indexedSimplices[2] = IndexedSimplex{{0, 1, 3}};
    indexedSimplices[3] = IndexedSimplex{{0, 2, 1}};

    ISS iss(vertices, indexedSimplices);

    POM x(iss);

    //
    // Accessors.
    //

    // Surface/Corner feature.
    for (std::size_t i = 0; i != vertices.size(); ++i) {
      assert(x.isVertexASurfaceFeature(i));
      assert(! x.isVertexAnEdgeFeature(i));
      assert(! x.isVertexACornerFeature(i));
    }
  }







  //--------------------------------------------------------------------------
  // 3-D space, 2-D mesh, 1st degree splines.
  // Edges and corners.  Use the dihedral angle in the constructor.
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSetIncAdj<3, 2> ISS;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::PointsOnManifold<3, 2, 1> POM;
    typedef POM::Vertex Vertex;

    std::vector<Vertex> vertices(4);
    vertices[0] = Vertex{{0., 0., 0.}};
    vertices[1] = Vertex{{1, 0, 0}};
    vertices[2] = Vertex{{0, 1, 0}};
    vertices[3] = Vertex{{0, 0, 1}};

    std::vector<IndexedSimplex> indexedSimplices(4);
    indexedSimplices[0] = IndexedSimplex{{1, 2, 3}};
    indexedSimplices[1] = IndexedSimplex{{0, 3, 2}};
    indexedSimplices[2] = IndexedSimplex{{0, 1, 3}};
    indexedSimplices[3] = IndexedSimplex{{0, 2, 1}};

    ISS iss(vertices, indexedSimplices);

    POM x(iss, 0.1);

    //
    // Accessors.
    //

    // Surface/Corner feature.
    for (std::size_t i = 0; i != vertices.size(); ++i) {
      assert(! x.isVertexASurfaceFeature(i));
      assert(! x.isVertexAnEdgeFeature(i));
      assert(x.isVertexACornerFeature(i));
    }

    // Has point.
    assert(!x.hasPoint(0));
    assert(!x.hasPoint(100));

    //
    // Insert points at the vertices.
    //
    x.insertAtVertices();

    //
    // Has point.
    //
    assert(x.hasPoint(0));
    assert(x.hasPoint(1));
    assert(x.hasPoint(2));
    assert(x.hasPoint(3));
    assert(!x.hasPoint(100));

    //
    // Is on Surface.
    //
    assert(! x.isOnSurface(0));
    assert(! x.isOnSurface(1));
    assert(! x.isOnSurface(2));
    assert(! x.isOnSurface(3));

    //
    // Is on edge.
    //
    assert(! x.isOnEdge(0));
    assert(! x.isOnEdge(1));
    assert(! x.isOnEdge(2));
    assert(! x.isOnEdge(3));

    //
    // Is on corner.
    //
    assert(x.isOnCorner(0));
    assert(x.isOnCorner(1));
    assert(x.isOnCorner(2));
    assert(x.isOnCorner(3));

    //
    // Closest points.
    //

    // Surface.
    {
      // Closest point for a surface.
      Vertex location = x.computeClosestPoint(0, Vertex{{0., 0., 0.}});
      assert(geom::computeDistance(location, Vertex{{0., 0.,
                                   0.}}) <= eps);
      assert(location == vertices[0]);
    }

    //
    // Update points.
    //
    {
      Vertex location = x.computeClosestPoint(0, Vertex{{0.1, 0.1,
                                              -1}});
      assert(geom::computeDistance(location, Vertex{{0., 0.,
                                   0.}}) <= eps);
      x.updatePoint();
      assert(x.isOnCorner(0));
      assert(x.getVertexIndex(0) == 0);
    }

    //
    // Erase points.
    //

    x.erase(0);
    x.erase(1);
    x.erase(2);
    x.erase(3);

    // Has point.
    assert(! x.hasPoint(0));
    assert(! x.hasPoint(1));
    assert(! x.hasPoint(2));
    assert(! x.hasPoint(3));
    assert(! x.hasPoint(100));

    //
    // Insert points and edges.
    //

    x.insertAtVerticesAndEdges();

    // Has point.
    assert(x.hasPoint(0));
    assert(x.hasPoint(1));
    assert(x.hasPoint(2));
    assert(x.hasPoint(3));
    assert(! x.hasPoint(4));

    // Has edge.
    assert(x.hasEdge(0, 1));
    assert(x.hasEdge(0, 2));
    assert(x.hasEdge(0, 3));
    assert(x.hasEdge(1, 2));
    assert(x.hasEdge(1, 3));
    assert(x.hasEdge(2, 3));

    // Erase an edge.
    x.erase(0, 1);
    assert(! x.hasEdge(0, 1));

    // Clear the edges.
    x.clearEdges();

    // Has edge.
    assert(! x.hasEdge(0, 1));
    assert(! x.hasEdge(0, 2));
    assert(! x.hasEdge(0, 3));
    assert(! x.hasEdge(1, 2));
    assert(! x.hasEdge(1, 3));
    assert(! x.hasEdge(2, 3));

    // Clear the points.
    x.clearPoints();

    // Has point.
    assert(! x.hasPoint(0));
    assert(! x.hasPoint(1));
    assert(! x.hasPoint(2));
    assert(! x.hasPoint(3));

    //
    // Split an edge.
    //

    // Insert points and edges.
    x.insertAtVerticesAndEdges();

    // Split an edge.
    x.insert(4, Vertex{{0.5, 0, 0}});
    x.splitEdge(0, 1, 4);

    // Has edge.
    assert(x.hasEdge(0, 4));
    assert(x.hasEdge(4, 1));
  }





  //--------------------------------------------------------------------------
  // 3-D space, 2-D mesh, 1st degree splines.
  // Corners.  Use the solid angle in the constructor.
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSetIncAdj<3, 2> ISS;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::PointsOnManifold<3, 2, 1> POM;
    typedef POM::Vertex Vertex;

    std::vector<Vertex> vertices(4);
    vertices[0] = Vertex{{0., 0., 0.}};
    vertices[1] = Vertex{{1, 0, 0}};
    vertices[2] = Vertex{{0, 1, 0}};
    vertices[3] = Vertex{{0, 0, 1}};

    std::vector<IndexedSimplex> indexedSimplices(4);
    indexedSimplices[0] = IndexedSimplex{{1, 2, 3}};
    indexedSimplices[1] = IndexedSimplex{{0, 3, 2}};
    indexedSimplices[2] = IndexedSimplex{{0, 1, 3}};
    indexedSimplices[3] = IndexedSimplex{{0, 2, 1}};

    ISS iss(vertices, indexedSimplices);

    POM x(iss, -1, 0.1);

    //
    // Accessors.
    //

    // Surface/Corner feature.
    for (std::size_t i = 0; i != vertices.size(); ++i) {
      assert(! x.isVertexASurfaceFeature(i));
      assert(! x.isVertexAnEdgeFeature(i));
      assert(x.isVertexACornerFeature(i));
    }
  }





  //--------------------------------------------------------------------------
  // 3-D space, 2-D mesh, 1st degree splines.
  // Only surface.  Use the boundary angle in the constructor.
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSetIncAdj<3, 2> ISS;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::PointsOnManifold<3, 2, 1> POM;
    typedef POM::Vertex Vertex;

    std::vector<Vertex> vertices(4);
    vertices[0] = Vertex{{0., 0., 0.}};
    vertices[1] = Vertex{{1, 0, 0}};
    vertices[2] = Vertex{{0, 1, 0}};
    vertices[3] = Vertex{{0, 0, 1}};

    std::vector<IndexedSimplex> indexedSimplices(4);
    indexedSimplices[0] = IndexedSimplex{{1, 2, 3}};
    indexedSimplices[1] = IndexedSimplex{{0, 3, 2}};
    indexedSimplices[2] = IndexedSimplex{{0, 1, 3}};
    indexedSimplices[3] = IndexedSimplex{{0, 2, 1}};

    ISS iss(vertices, indexedSimplices);

    POM x(iss, -1, -1, 0.1);

    //
    // Accessors.
    //

    // Surface/Corner feature.
    for (std::size_t i = 0; i != vertices.size(); ++i) {
      assert(x.isVertexASurfaceFeature(i));
      assert(! x.isVertexAnEdgeFeature(i));
      assert(! x.isVertexACornerFeature(i));
    }
  }




  //--------------------------------------------------------------------------
  // 3-D space, 2-D mesh, 1st degree splines.
  // Mesh with boundary.
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSetIncAdj<3, 2> ISS;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::PointsOnManifold<3, 2, 1> POM;
    typedef POM::Vertex Vertex;

    std::vector<Vertex> vertices(4);
    vertices[0] = Vertex{{0., 0., 0.}};
    vertices[1] = Vertex{{1, 0, 0}};
    vertices[2] = Vertex{{0, 1, 0}};
    vertices[3] = Vertex{{0, 0, 1}};

    std::vector<IndexedSimplex> indexedSimplices(3);
    indexedSimplices[0] = IndexedSimplex{{0, 3, 2}};
    indexedSimplices[1] = IndexedSimplex{{0, 1, 3}};
    indexedSimplices[2] = IndexedSimplex{{0, 2, 1}};

    ISS iss(vertices, indexedSimplices);

    POM x(iss);

    //
    // Accessors.
    //

    // Surface/Edge/Corner features.
    assert(x.isVertexASurfaceFeature(0));
    assert(x.isVertexAnEdgeFeature(1));
    assert(x.isVertexAnEdgeFeature(2));
    assert(x.isVertexAnEdgeFeature(3));
  }






  //--------------------------------------------------------------------------
  // 3-D space, 2-D mesh, 1st degree splines.
  // Mesh with boundary.  Use boundary angle.
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSetIncAdj<3, 2> ISS;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::PointsOnManifold<3, 2, 1> POM;
    typedef POM::Vertex Vertex;

    std::vector<Vertex> vertices(4);
    vertices[0] = Vertex{{0., 0., 0.}};
    vertices[1] = Vertex{{1, 0, 0}};
    vertices[2] = Vertex{{0, 1, 0}};
    vertices[3] = Vertex{{0, 0, 1}};

    std::vector<IndexedSimplex> indexedSimplices(3);
    indexedSimplices[0] = IndexedSimplex{{0, 3, 2}};
    indexedSimplices[1] = IndexedSimplex{{0, 1, 3}};
    indexedSimplices[2] = IndexedSimplex{{0, 2, 1}};

    ISS iss(vertices, indexedSimplices);

    POM x(iss, -1, -1, 0.1);

    //
    // Accessors.
    //

    // Surface/Edge/Corner features.
    assert(x.isVertexASurfaceFeature(0));
    assert(x.isVertexACornerFeature(1));
    assert(x.isVertexACornerFeature(2));
    assert(x.isVertexACornerFeature(3));
  }

  return 0;
}
