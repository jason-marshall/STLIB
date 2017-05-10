// -*- C++ -*-

#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/buildFromSimplices.h"

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
  // build_from_simplices
  //
  {
    // 1-D space, 0-D simplex.
    typedef std::array<double, 1> Pt;
    typedef geom::IndSimpSet<1, 0> Mesh;

    std::vector<Pt> points(10);
    const double eps = 10 * std::numeric_limits<double>::epsilon();
    points[0][0] = 0;
    points[1][0] = 0 + eps;
    points[2][0] = 0 - eps;
    points[3][0] = 2;
    points[4][0] = 2 - eps;
    points[5][0] = 2 - eps;
    points[6][0] = 10;
    points[7][0] = 10 + eps;
    points[8][0] = 10 + 2 * eps;
    points[9][0] = 10 - eps;

    Mesh mesh;
    geom::buildFromSimplices(points.begin(), points.end(), &mesh);

    assert(mesh.vertices.size() == 3);
    assert(mesh.indexedSimplices.size() == 10);
  }
  {
    // 2-D space, 2-D simplex.
    typedef std::array<double, 2> Pt;
    typedef geom::IndSimpSet<2, 2> Mesh;

    std::vector<Pt> points(12);
    points[0] = Pt{{0, 0}};
    points[1] = Pt{{1, 0}};
    points[2] = Pt{{0, 1}};

    points[3] = Pt{{0, 0}};
    points[4] = Pt{{0, 1}};
    points[5] = Pt{{-1, 0}};

    points[6] = Pt{{0, 0}};
    points[7] = Pt{{-1, 0}};
    points[8] = Pt{{0, -1}};

    points[9]  = Pt{{0, 0}};
    points[10] = Pt{{0, -1}};
    points[11] = Pt{{1, 0}};

    Mesh mesh;
    geom::buildFromSimplices(points.begin(), points.end(), &mesh);

    assert(mesh.vertices.size() == 5);
    assert(mesh.indexedSimplices.size() == 4);
  }

  //
  // Convert from quad mesh.  3-D space.
  //
  {
    typedef geom::QuadMesh<3> QuadMesh;
    typedef geom::IndSimpSet<3, 2> TriangleMesh;
    //
    // Data for a cube
    //
    double vertices[] = {0, 0, 0, // 0
                         1, 0, 0, // 1
                         1, 1, 0, // 2
                         0, 1, 0, // 3
                         0, 0, 1, // 4
                         1, 0, 1, // 5
                         1, 1, 1, // 6
                         0, 1, 1
                        };// 7
    const std::size_t numberOfVertices = sizeof(vertices) / (3 * sizeof(double));
    std::size_t faces[] = {0, 3, 2, 1,
                           4, 5, 6, 7,
                           1, 2, 6, 5,
                           0, 4, 7, 3,
                           0, 1, 5, 4,
                           2, 3, 7, 6
                          };
    const std::size_t numberOfFaces = sizeof(faces) / (4 * sizeof(std::size_t));

    // Construct from vertices and faces.
    QuadMesh quadMesh(numberOfVertices, vertices, numberOfFaces, faces);

    // Build the triangle mesh.
    TriangleMesh mesh;
    geom::buildFromQuadMesh(quadMesh, &mesh);
    assert(mesh.vertices.size() == numberOfVertices);
    assert(mesh.indexedSimplices.size() == 2 * numberOfFaces);
  }

  //
  // build_from_subset_vertices
  //
  {
    typedef geom::IndSimpSet<2, 2> Mesh;

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

    Mesh x;
    std::vector<std::size_t> vertexIndices;

    geom::buildFromSubsetVertices(mesh, vertexIndices.begin(),
                                  vertexIndices.end(), &x);
    assert(x.vertices.size() == 0);
    assert(x.indexedSimplices.size() == 0);

    vertexIndices.push_back(0);
    geom::buildFromSubsetVertices(mesh, vertexIndices.begin(),
                                  vertexIndices.end(), &x);
    assert(x.vertices.size() == 0);
    assert(x.indexedSimplices.size() == 0);

    vertexIndices.push_back(1);
    geom::buildFromSubsetVertices(mesh, vertexIndices.begin(),
                                  vertexIndices.end(), &x);
    assert(x.vertices.size() == 0);
    assert(x.indexedSimplices.size() == 0);

    vertexIndices.push_back(2);
    geom::buildFromSubsetVertices(mesh, vertexIndices.begin(),
                                  vertexIndices.end(), &x);
    assert(x.vertices.size() == 3);
    assert(x.indexedSimplices.size() == 1);

    vertexIndices.push_back(3);
    geom::buildFromSubsetVertices(mesh, vertexIndices.begin(),
                                  vertexIndices.end(), &x);
    assert(x.vertices.size() == 4);
    assert(x.indexedSimplices.size() == 2);

    vertexIndices.push_back(4);
    geom::buildFromSubsetVertices(mesh, vertexIndices.begin(),
                                  vertexIndices.end(), &x);
    assert(x.vertices.size() == 5);
    assert(x.indexedSimplices.size() == 4);
  }


  //
  // buildFromSubsetSimplices
  //
  {
    typedef geom::IndSimpSet<2, 2> Mesh;

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

    Mesh x;
    std::vector<std::size_t> simplexIndices;

    geom::buildFromSubsetSimplices(mesh, simplexIndices.begin(),
                                   simplexIndices.end(), &x);
    assert(x.vertices.size() == 0);
    assert(x.indexedSimplices.size() == 0);

    simplexIndices.push_back(0);
    geom::buildFromSubsetSimplices(mesh, simplexIndices.begin(),
                                   simplexIndices.end(), &x);
    assert(x.vertices.size() == 3);
    assert(x.indexedSimplices.size() == 1);

    simplexIndices.push_back(1);
    geom::buildFromSubsetSimplices(mesh, simplexIndices.begin(),
                                   simplexIndices.end(), &x);
    assert(x.vertices.size() == 4);
    assert(x.indexedSimplices.size() == 2);

    simplexIndices.push_back(2);
    geom::buildFromSubsetSimplices(mesh, simplexIndices.begin(),
                                   simplexIndices.end(), &x);
    assert(x.vertices.size() == 5);
    assert(x.indexedSimplices.size() == 3);

    simplexIndices.push_back(3);
    geom::buildFromSubsetSimplices(mesh, simplexIndices.begin(),
                                   simplexIndices.end(), &x);
    assert(x.vertices.size() == 5);
    assert(x.indexedSimplices.size() == 4);
  }


  //
  // buildFromVerticesInside
  //
  {
    typedef geom::IndSimpSet<2, 2> Mesh;
    typedef std::array<double, 2> Pt;

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

    Mesh x;
    {
      Circle f(Pt{{0., 0.}}, 0.1);
      geom::buildFromVerticesInside(mesh, f, &x);
      assert(x.vertices.size() == 0);
      assert(x.indexedSimplices.size() == 0);
    }
    {
      Circle f(Pt{{0., 0.}}, 1.1);
      geom::buildFromVerticesInside(mesh, f, &x);
      assert(x.vertices.size() == 5);
      assert(x.indexedSimplices.size() == 4);
    }
    {
      Circle f(Pt{{0.5, 0.}}, 1.4);
      geom::buildFromVerticesInside(mesh, f, &x);
      assert(x.vertices.size() == 4);
      assert(x.indexedSimplices.size() == 2);
    }
  }


  //
  // buildFromSimplicesInside
  //
  {
    typedef geom::IndSimpSet<2, 2> Mesh;
    typedef std::array<double, 2> Pt;

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

    Mesh x;
    {
      Circle f(Pt{{0., 0.}}, 0.1);
      geom::buildFromSimplicesInside(mesh, f, &x);
      assert(x.vertices.size() == 0);
      assert(x.indexedSimplices.size() == 0);
    }
    {
      Circle f(Pt{{0., 0.}}, 1.0);
      geom::buildFromSimplicesInside(mesh, f, &x);
      assert(x.vertices.size() == 5);
      assert(x.indexedSimplices.size() == 4);
    }
    {
      Circle f(Pt{{0.5, 0.}}, 0.5);
      geom::buildFromSimplicesInside(mesh, f, &x);
      assert(x.vertices.size() == 4);
      assert(x.indexedSimplices.size() == 2);
    }
  }

  //
  // merge.
  //
  {
    typedef geom::IndSimpSet<1, 1> Mesh;

    const std::size_t numVertices0 = 2;
    double vertices0[] = {0, 1};
    const std::size_t numSimplices0 = 1;
    std::size_t indexedSimplices0[] = {0, 1};

    const std::size_t numVertices1 = 3;
    double vertices1[] = {2, 3, 4};
    const std::size_t numSimplices1 = 2;
    std::size_t indexedSimplices1[] = {0, 1, 1, 2};

    {
      std::array<Mesh, 2> meshes;
      build(&meshes[0], numVertices0, vertices0,
            numSimplices0, indexedSimplices0);
      build(&meshes[1], numVertices1, vertices1,
            numSimplices1, indexedSimplices1);

      Mesh mesh;
      geom::merge(meshes.begin(), meshes.end(), &mesh);

      assert(mesh.vertices.size() == 5);
      assert(mesh.indexedSimplices.size() == 3);

      assert(mesh.vertices[0][0] == 0);
      assert(mesh.vertices[1][0] == 1);
      assert(mesh.vertices[2][0] == 2);
      assert(mesh.vertices[3][0] == 3);
      assert(mesh.vertices[4][0] == 4);

      assert(mesh.indexedSimplices[0][0] == 0);
      assert(mesh.indexedSimplices[0][1] == 1);
      assert(mesh.indexedSimplices[1][0] == 2);
      assert(mesh.indexedSimplices[1][1] == 3);
      assert(mesh.indexedSimplices[2][0] == 3);
      assert(mesh.indexedSimplices[2][1] == 4);
    }
    {
      Mesh a, b;
      build(&a, numVertices0, vertices0, numSimplices0, indexedSimplices0);
      build(&b, numVertices1, vertices1, numSimplices1, indexedSimplices1);

      Mesh mesh;
      geom::merge2(a, b, &mesh);

      assert(mesh.vertices.size() == 5);
      assert(mesh.indexedSimplices.size() == 3);

      assert(mesh.vertices[0][0] == 0);
      assert(mesh.vertices[1][0] == 1);
      assert(mesh.vertices[2][0] == 2);
      assert(mesh.vertices[3][0] == 3);
      assert(mesh.vertices[4][0] == 4);

      assert(mesh.indexedSimplices[0][0] == 0);
      assert(mesh.indexedSimplices[0][1] == 1);
      assert(mesh.indexedSimplices[1][0] == 2);
      assert(mesh.indexedSimplices[1][1] == 3);
      assert(mesh.indexedSimplices[2][0] == 3);
      assert(mesh.indexedSimplices[2][1] == 4);
    }
  }

  return 0;
}
