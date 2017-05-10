// -*- C++ -*-

#include "stlib/geom/mesh/iss/ISS_VertexField.h"
#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/file_io.h"

#include "stlib/geom/kernel/BBox.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  const double eps = 100.0 * std::numeric_limits<double>::epsilon();

  //
  // 3-D space.  3-D simplex.  1-D field.
  //
  {
    typedef geom::IndSimpSet<3, 3> ISS;
    typedef ISS::Vertex Vertex;
    typedef geom::ISS_VertexField<ISS> ISS_VF;

    //
    // Data for an octahedron
    //
    const std::size_t numVertices = 7;
    double vertices[] = { 0, 0, 0,
                          1, 0, 0,
                          -1, 0, 0,
                          0, 1, 0,
                          0, -1, 0,
                          0, 0, 1,
                          0, 0, -1
                        };
    // z value.
    double fields[] = { 0, 0, 0, 0, 0, 1, -1 };
    const std::size_t numTets = 8;
    std::size_t tets[] = { 0, 1, 3, 5,
                           0, 3, 2, 5,
                           0, 2, 4, 5,
                           0, 4, 1, 5,
                           0, 3, 1, 6,
                           0, 2, 3, 6,
                           0, 4, 2, 6,
                           0, 1, 4, 6
                         };
    {
      // Construct from vertices and tetrahedra.
      ISS mesh;
      build(&mesh, numVertices, vertices, numTets, tets);
      ISS_VF x(mesh, fields);

      // Print the mesh.
      std::cout << "Tetrahedral mesh for an octahedron:\n";
      geom::writeAscii(std::cout, mesh);
      std::cout << "\n";

      // copy constructor
      ISS_VF copyConstructed(x);

      // Interpolation.
      assert(std::abs(x.interpolate(0, Vertex{{0.0, 0.0, 0.0}}) - 0.0) < eps);
      assert(std::abs(x.interpolate(0, Vertex{{0.0, 0.0, 1.0}}) - 1.0) < eps);
      assert(std::abs(x.interpolate(1, Vertex{{0.0, 0.0, 2.0}}) - 2.0) < eps);
    }
  }

  //
  // 3-D space.  3-D simplex.  3-D field.
  //
  {
    typedef geom::IndSimpSet<3, 3> ISS;
    typedef ISS::Vertex Vertex;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef geom::ISS_VertexField<ISS, Vertex> ISS_VF;
    typedef ISS_VF::Field Field;

    //
    // Data for an octahedron
    //
    Vertex vertices[] = {{{0, 0, 0}},
      {{1, 0, 0}},
      {{ -1, 0, 0}},
      {{0, 1, 0}},
      {{0, -1, 0}},
      {{0, 0, 1}},
      {{0, 0, -1}}
    };
    const std::size_t numVertices = sizeof(vertices) / sizeof(Vertex);
    assert(numVertices == 7);
    IndexedSimplex tets[] = {{{0, 1, 3, 5}},
                             {{0, 3, 2, 5}},
                             {{0, 2, 4, 5}},
                             {{0, 4, 1, 5}},
                             {{0, 3, 1, 6}},
                             {{0, 2, 3, 6}},
                             {{0, 4, 2, 6}},
                             {{0, 1, 4, 6}}
                            };
    const std::size_t numTets = sizeof(tets) / sizeof(IndexedSimplex);
    assert(numTets == 8);
    {
      // Construct from vertices and tetrahedra.  The fields are the
      // vertex coordinates.
      ISS mesh;
      build(&mesh, numVertices, vertices, numTets, tets);
      ISS_VF x(mesh, vertices);

      // Interpolation.
      {
        Vertex p = {{0.0, 0.0, 0.0}};
        Field v = x.interpolate(0, p);
        assert(stlib::ext::euclideanDistance(p, v) < eps);
      }
      {
        Vertex p = {{0.0, 0.0, 0.0}};
        Field v = x.interpolate(numTets - 1, p);
        assert(stlib::ext::euclideanDistance(p, v) < eps);
      }
      {
        Vertex p = {{1.0, 2.0, 3.0}};
        Field v = x.interpolate(0, p);
        assert(stlib::ext::euclideanDistance(p, v) < eps);
      }
    }
  }

  //
  // 3-D space.  2-D simplex.  1-D field.
  //
  {
    typedef geom::IndSimpSet<3, 2> ISS;
    typedef ISS::Vertex Vertex;
    typedef geom::ISS_VertexField<ISS> ISS_VF;

    //
    // Data for an octahedron
    //
    const std::size_t numVertices = 6;
    double vertices[] = {
      1, 0, 0,    // 0
      -1, 0, 0,    // 1
      0, 1, 0,     // 2
      0, -1, 0,    // 3
      0, 0, 1,     // 4
      0, 0, -1
    };  // 5
    // z value.
    double fields[] = { 0, 0, 0, 0, 1, -1 };
    const std::size_t numTriangles = 8;
    std::size_t triangles[] = {
      0, 2, 4,
      2, 0, 5,
      2, 1, 4,
      1, 2, 5,
      1, 3, 4,
      3, 1, 5,
      3, 0, 4,
      0, 3, 5
    };
    {
      // Construct from vertices and triangles.
      ISS mesh;
      build(&mesh, numVertices, vertices, numTriangles, triangles);
      ISS_VF x(mesh, fields);

      // Print the mesh.
      std::cout << "Space dimension = " << mesh.N << "\n"
      << "Simplex dimension = " << mesh.M << "\n"
      << "Triangle mesh for an octahedron:\n";
      geom::writeAscii(std::cout, mesh);
      std::cout << "\n";

      // copy constructor
      ISS_VF copyConstructed(x);

      // Interpolation.
      assert(std::abs(x.interpolate(0, Vertex{{1.0, 0.0, 0.0}})
      - 0.0) < eps);
      assert(std::abs(x.interpolate(0, Vertex{{2.0, 1.0, 1.0}})
      - 0.0) < eps);
      assert(std::abs(x.interpolate(0, Vertex{{0.0, 0.0, 1.0}})
      - 1.0) < eps);
      assert(std::abs(x.interpolate(0, Vertex{{2.0, 2.0, 3.0}})
      - 1.0) < eps);
    }
  }

  //
  // 3-D space.  2-D simplex.  3-D field.
  //
  {
    typedef geom::IndSimpSet<3, 2, double> ISS;
    typedef ISS::Vertex Vertex;
    typedef ISS::IndexedSimplex IndexedSimplex;

    typedef geom::ISS_VertexField<ISS, std::array<double, 3> > ISS_VF;
    typedef ISS_VF::Field Field;

    //
    // Data for an octahedron
    //
    Vertex vertices[] = {{{1, 0, 0}},    // 0
      {{ -1, 0, 0}},   // 1
      {{0, 1, 0}},     // 2
      {{0, -1, 0}},    // 3
      {{0, 0, 1}},     // 4
      {{0, 0, -1}}
    };   // 5
    const std::size_t numVertices = sizeof(vertices) / sizeof(Vertex);
    assert(numVertices == 6);

    IndexedSimplex triangles[] = {{{0, 2, 4}},
                                  {{2, 0, 5}},
                                  {{2, 1, 4}},
                                  {{1, 2, 5}},
                                  {{1, 3, 4}},
                                  {{3, 1, 5}},
                                  {{3, 0, 4}},
                                  {{0, 3, 5}}
                                 };
    const std::size_t numTriangles = sizeof(triangles) / sizeof(IndexedSimplex);
    assert(numTriangles == 8);
    {
      // Construct from vertices and triangles.
      ISS mesh;
      build(&mesh, numVertices, vertices, numTriangles, triangles);
      ISS_VF x(mesh, vertices);

      // Interpolation.
      {
        // x + y + z = 1
        Vertex v = {{0.0, 0.0, 0.0}};
        Field cp = {{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}};
        Field f = x.interpolate(0, v);
        assert(stlib::ext::euclideanDistance(f, cp) < eps);
      }
      {
        // x + y + z = 1
        Vertex v = {{1.0, 0.0, 0.0}};
        Field cp = {{1.0, 0.0, 0.0}};
        Field f = x.interpolate(0, v);
        assert(stlib::ext::euclideanDistance(f, cp) < eps);
      }
      {
        // x + y + z = 1
        Vertex v = {{2.0, 1.0, 1.0}};
        Field cp = {{1.0, 0.0, 0.0}};
        Field f = x.interpolate(0, v);
        assert(stlib::ext::euclideanDistance(f, cp) < eps);
      }
    }
  }

  return 0;
}
