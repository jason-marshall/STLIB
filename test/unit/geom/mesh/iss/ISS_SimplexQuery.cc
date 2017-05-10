// -*- C++ -*-

#include "stlib/geom/mesh/iss/ISS_SimplexQuery.h"
#include "stlib/geom/mesh/iss/IndSimpSet.h"
#include "stlib/geom/mesh/iss/build.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  //
  // 3-D space.  3-D simplex.
  //
  {
    typedef geom::IndSimpSet<3, 3> ISS;
    typedef ISS::Vertex Vertex;
    typedef geom::ISS_SimplexQuery<ISS> SQ;
    typedef SQ::BBox BBox;

    //
    // Data for an octahedron
    //

    const std::size_t numVertices = 7;
    double vertices[] = { 0, 0, 0,   // 0
                          1, 0, 0,    // 1
                          -1, 0, 0,   // 2
                          0, 1, 0,    // 3
                          0, -1, 0,   // 4
                          0, 0, 1,    // 5
                          0, 0, -1
                        }; // 6
    const std::size_t numTets = 8;
    std::size_t tets[] = { 0, 1, 3, 5,  // 0
                           0, 3, 2, 5,   // 1
                           0, 2, 4, 5,   // 2
                           0, 4, 1, 5,   // 3
                           0, 3, 1, 6,   // 4
                           0, 2, 3, 6,   // 5
                           0, 4, 2, 6,   // 6
                           0, 1, 4, 6
                         }; // 7

    // Construct from vertices and tetrahedra.
    ISS mesh;
    build(&mesh, numVertices, vertices, numTets, tets);
    SQ sq(mesh);

    //
    // Point queries.
    //

    {
      Vertex x = {{10, 10, 10}};
      std::vector<std::size_t> indices;
      sq.computePointQuery(std::back_inserter(indices), x);
      assert(indices.size() == 0);
    }
    {
      Vertex x = {{1, 1, 1}};
      std::vector<std::size_t> indices;
      sq.computePointQuery(std::back_inserter(indices), x);
      assert(indices.size() == 0);
    }
    {
      Vertex x = {{0.1, 0.1, 0.1}};
      std::vector<std::size_t> indices;
      sq.computePointQuery(std::back_inserter(indices), x);
      assert(indices.size() == 1);
    }
    {
      Vertex x = {{0.5, 0.1, 0}};
      std::vector<std::size_t> indices;
      sq.computePointQuery(std::back_inserter(indices), x);
      assert(indices.size() == 2);
    }
    {
      Vertex x = {{0.5, 0, 0}};
      std::vector<std::size_t> indices;
      sq.computePointQuery(std::back_inserter(indices), x);
      assert(indices.size() == 4);
    }
    {
      Vertex x = {{0, 0, 0}};
      std::vector<std::size_t> indices;
      sq.computePointQuery(std::back_inserter(indices), x);
      assert(indices.size() == 8);
    }

    //
    // Window, bounding box queries.
    //

    {
      BBox window = {{{9, 9, 9}}, {{10, 10, 10}}};
      std::vector<std::size_t> indices;
      sq.computeWindowQuery(std::back_inserter(indices), window);
      assert(indices.size() == 0);
    }
    {
      BBox window = {{{0.5, 0.5, 0.5}}, {{1, 1, 1}}};
      std::vector<std::size_t> indices;
      sq.computeWindowQuery(std::back_inserter(indices), window);
      assert(indices.size() == 1);
    }
    {
      BBox window = {{{0.1, 0.1, 0.1}}, {{1, 1, 1}}};
      std::vector<std::size_t> indices;
      sq.computeWindowQuery(std::back_inserter(indices), window);
      assert(indices.size() == 1);
    }
    {
      BBox window = {{{ -1, 0.1, 0.1}}, {{1, 1, 1}}};
      std::vector<std::size_t> indices;
      sq.computeWindowQuery(std::back_inserter(indices), window);
      assert(indices.size() == 2);
    }
    {
      BBox window = {{{ -1, -1, 0.1}}, {{1, 1, 1}}};
      std::vector<std::size_t> indices;
      sq.computeWindowQuery(std::back_inserter(indices), window);
      assert(indices.size() == 4);
    }
    {
      BBox window = {{{ -1, -1, -1}}, {{1, 1, 1}}};
      std::vector<std::size_t> indices;
      sq.computeWindowQuery(std::back_inserter(indices), window);
      assert(indices.size() == 8);
    }
    {
      BBox window = {{{0, 0, 0}}, {{0, 0, 0}}};
      std::vector<std::size_t> indices;
      sq.computeWindowQuery(std::back_inserter(indices), window);
      assert(indices.size() == 8);
    }

    //
    // Minimum distance queries.
    //

    assert(sq.computeMinimumDistanceIndex(Vertex{{10., 10.,
              10.}}) == 0);
    assert(sq.computeMinimumDistanceIndex(Vertex{{1., 1.,
                                          1.}}) == 0);
    assert(sq.computeMinimumDistanceIndex(Vertex{{0.1, 0.1,
                                          0.1}}) == 0);

    assert(sq.computeMinimumDistanceIndex(Vertex{{-10., 10.,
                                          10.}}) == 1);
    assert(sq.computeMinimumDistanceIndex(Vertex{{-1., 1.,
                                          1.}}) == 1);
    assert(sq.computeMinimumDistanceIndex(Vertex{{-0.1, 0.1,
                                          0.1}}) == 1);

    assert(sq.computeMinimumDistanceIndex(Vertex{{-10., -10.,
                                          10.}}) == 2);
    assert(sq.computeMinimumDistanceIndex(Vertex{{-1., -1.,
                                          1.}}) == 2);
    assert(sq.computeMinimumDistanceIndex(Vertex{{-0.1, -0.1,
                                          0.1}}) == 2);

    assert(sq.computeMinimumDistanceIndex(Vertex{{10., -10.,
                                          10.}}) == 3);
    assert(sq.computeMinimumDistanceIndex(Vertex{{1., -1.,
                                          1.}}) == 3);
    assert(sq.computeMinimumDistanceIndex(Vertex{{0.1, -0.1,
                                          0.1}}) == 3);

    assert(sq.computeMinimumDistanceIndex(Vertex{{10., 10.,
                                          -10.}}) == 4);
    assert(sq.computeMinimumDistanceIndex(Vertex{{1., 1.,
                                          -1.}}) == 4);
    assert(sq.computeMinimumDistanceIndex(Vertex{{0.1, 0.1,
                                          -0.1}}) == 4);

    assert(sq.computeMinimumDistanceIndex(Vertex{{-10., 10.,
                                          -10.}}) == 5);
    assert(sq.computeMinimumDistanceIndex(Vertex{{-1., 1.,
                                          -1.}}) == 5);
    assert(sq.computeMinimumDistanceIndex(Vertex{{-0.1, 0.1,
                                          -0.1}}) == 5);

    assert(sq.computeMinimumDistanceIndex(Vertex{{-10., -10.,
                                          -10.}}) == 6);
    assert(sq.computeMinimumDistanceIndex(Vertex{{-1., -1.,
                                          -1.}}) == 6);
    assert(sq.computeMinimumDistanceIndex(Vertex{{-0.1, -0.1,
                                          -0.1}}) == 6);

    assert(sq.computeMinimumDistanceIndex(Vertex{{10., -10.,
                                          -10.}}) == 7);
    assert(sq.computeMinimumDistanceIndex(Vertex{{1., -1.,
                                          -1.}}) == 7);
    assert(sq.computeMinimumDistanceIndex(Vertex{{0.1, -0.1,
                                          -0.1}}) == 7);

    std::size_t index;

    index = sq.computeMinimumDistanceIndex(Vertex{{0., 0.1, 0.1}});
    assert(index == 0 || index == 1);

    index = sq.computeMinimumDistanceIndex(Vertex{{0., 0., 0.1}});
    assert(index < 4);

    index = sq.computeMinimumDistanceIndex(Vertex{{0., 0., -0.1}});
    assert(4 <= index && index < 8);

    index = sq.computeMinimumDistanceIndex(Vertex{{0., 0., 0.}});
    assert(index < 8);
  }

  return 0;
}
