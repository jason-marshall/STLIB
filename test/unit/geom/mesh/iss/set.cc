// -*- C++ -*-

// CONTINUE: Add tests for set_of_incident_*.

#include "stlib/geom/mesh/iss/set.h"
#include "stlib/geom/mesh/iss/build.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

// Distance function for a circle.
class Circle :
  public std::unary_function<const std::array<double, 2>&, double>
{
private:
  // Types.
  typedef std::unary_function<const std::array<double, 2>&, double> Base;

  // Data.
  std::array<double, 2> _center;
  double _radius;

public:
  // Types.
  typedef Base::argument_type argument_type;
  typedef Base::result_type result_type;

  // Constructor.
  Circle(const std::array<double, 2> center = {{0., 0.}},
         const double radius = 1.0) :
    _center(center),
    _radius(radius) {}

  // Functor.
  result_type
  operator()(argument_type x) const
  {
    static std::array<double, 2> y;
    y = x;
    y -= _center;
    return stlib::ext::magnitude(y) - _radius;
  }
};



int
main()
{
  //
  // determineVerticesInside
  //
  {
    typedef geom::IndSimpSet<2, 2> Mesh;
    typedef std::array<double, 2> Point;

    const std::size_t numVertices = 5;
    double vertices[] = {0, 0,
                         1, 0,
                         0, 1,
                         -1, 0,
                         0, -1
                        };
    const std::size_t numSimplices = 4;
    std::size_t indexed_simplices[] = {0, 1, 2,
                                       0, 2, 3,
                                       0, 3, 4,
                                       0, 4, 1
                                      };

    Mesh mesh;
    build(&mesh, numVertices, vertices, numSimplices, indexed_simplices);
    assert(mesh.vertices.size() == 5);
    assert(mesh.indexedSimplices.size() == 4);

    {
      std::vector<std::size_t> indices;
      Circle f(Point{{0., 0.}}, 0.1);
      determineVerticesInside(mesh, f, std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      std::vector<std::size_t> indices;
      Circle f(Point{{0., 0.}}, 1.1);
      determineVerticesInside(mesh, f, std::back_inserter(indices));
      assert(indices.size() == 5);
    }
    {
      std::vector<std::size_t> indices;
      Circle f(Point{{0.5, 0.}}, 1.4);
      determineVerticesInside(mesh, f, std::back_inserter(indices));
      assert(indices.size() == 4);
    }
  }


  //
  // set_of_simplices_inside
  //
  {
    typedef geom::IndSimpSet<2, 2> Mesh;
    typedef std::array<double, 2> Point;

    const std::size_t numVertices = 5;
    double vertices[] = { 0, 0,
                          1, 0,
                          0, 1,
                          -1, 0,
                          0, -1
                        };
    const std::size_t numSimplices = 4;
    std::size_t indexedSimplices[] = { 0, 1, 2,
                                       0, 2, 3,
                                       0, 3, 4,
                                       0, 4, 1
                                     };

    Mesh mesh;
    build(&mesh, numVertices, vertices, numSimplices, indexedSimplices);
    assert(mesh.vertices.size() == 5);
    assert(mesh.indexedSimplices.size() == 4);

    {
      std::vector<std::size_t> indices;
      Circle f(Point{{0., 0.}}, 0.1);
      determineSimplicesInside(mesh, f, std::back_inserter(indices));
      assert(indices.size() == 0);
    }
    {
      std::vector<std::size_t> indices;
      Circle f(Point{{0., 0.}}, 1.1);
      determineSimplicesInside(mesh, f, std::back_inserter(indices));
      assert(indices.size() == 4);
    }
    {
      std::vector<std::size_t> indices;
      Circle f(Point{{0.5, 0.}}, 0.5);
      determineSimplicesInside(mesh, f, std::back_inserter(indices));
      assert(indices.size() == 2);
    }
  }

  //
  // labelComponents
  //
  {
    typedef geom::IndSimpSetIncAdj<1, 1> Mesh;
    typedef Mesh::Vertex Vertex;
    typedef Mesh::IndexedSimplex IndexedSimplex;

    {
      Mesh mesh;
      std::vector<std::size_t> labels;
      assert(geom::labelComponents(mesh, &labels) == 0);
      assert(labels.size() == 0);
    }
    {
      Vertex vertices[] = {{{0}},
        {{1}},
        {{2}},
        {{3}},
        {{4}}
      };
      IndexedSimplex simplices[] = {{{0, 1}},
        {{1, 2}},
        {{3, 4}}
      };
      Mesh mesh;
      build(&mesh, sizeof(vertices) / sizeof(Vertex), vertices,
            sizeof(simplices) / sizeof(IndexedSimplex), simplices);
      std::vector<std::size_t> labels;
      assert(geom::labelComponents(mesh, &labels) == 2);
      assert(labels.size() == 3);
      assert(labels[0] == 0);
      assert(labels[1] == 0);
      assert(labels[2] == 1);
    }
  }

  {
    typedef geom::IndSimpSetIncAdj<2, 2> Mesh;
    typedef Mesh::Vertex Vertex;
    typedef Mesh::IndexedSimplex IndexedSimplex;

    {
      Mesh mesh;
      std::vector<std::size_t> labels;
      assert(geom::labelComponents(mesh, &labels) == 0);
      assert(labels.size() == 0);
    }
    {
      // 3 2 5
      // 0 1 4
      Vertex vertices[] = {{{0, 0}},
        {{1, 0}},
        {{1, 1}},
        {{0, 1}},
        {{2, 0}},
        {{2, 1}}
      };
      IndexedSimplex simplices[] = {{{0, 1, 2}},
        {{2, 3, 0}},
        {{4, 5, 2}}
      };
      Mesh mesh;
      build(&mesh, sizeof(vertices) / sizeof(Vertex), vertices,
            sizeof(simplices) / sizeof(IndexedSimplex), simplices);
      std::vector<std::size_t> labels;
      assert(geom::labelComponents(mesh, &labels) == 2);
      assert(labels.size() == 3);
      assert(labels[0] == 0);
      assert(labels[1] == 0);
      assert(labels[2] == 1);
    }
  }

  return 0;
}
