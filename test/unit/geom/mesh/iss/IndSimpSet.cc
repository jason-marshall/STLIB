// -*- C++ -*-

#include "stlib/geom/mesh/iss/IndSimpSet.h"

#include "stlib/geom/mesh/iss/file_io.h"
#include "stlib/geom/mesh/iss/equality.h"
#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/buildFromSimplices.h"

#include <iostream>
#include <sstream>

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
    typedef ISS::Simplex Simplex;
    typedef ISS::SimplexConstIterator SimplexConstIterator;
    typedef geom::BBox<double, 3> BBox;
    {
      // default constructor
      ISS x;
      assert(x.N == 3);
      assert(x.M == 3);
      assert(x.vertices.size() == 0);
      assert(x.indexedSimplices.size() == 0);
      //std::cout << "ISS() = " << '\n' << tm << '\n';
    }
    //
    // Data for an octahedron
    //
    const std::size_t num_vertices = 7;
    double vertices[] = { 0, 0, 0,
                          1, 0, 0,
                          -1, 0, 0,
                          0, 1, 0,
                          0, -1, 0,
                          0, 0, 1,
                          0, 0, -1
                        };
    const std::size_t num_tets = 8;
    std::size_t tets[] = { 0, 1, 3, 5,
                           0, 3, 2, 5,
                           0, 2, 4, 5,
                           0, 4, 1, 5,
                           0, 3, 1, 6,
                           0, 2, 3, 6,
                           0, 4, 2, 6,
                           0, 1, 4, 6
                         };

    // Construct from vertices and tetrahedra.
    ISS mesh;
    build(&mesh, num_vertices, vertices, num_tets, tets);

    {
      Simplex s;
      SimplexConstIterator i = mesh.getSimplicesBegin();
      for (std::size_t n = 0; n != num_tets; ++n, ++i) {
        s = *i;
        for (std::size_t m = 0; m != 4; ++m) {
          assert(s[m] == mesh.getSimplexVertex(n, m));
        }
      }
    }

    // Print the mesh.
    std::cout << "Tetrahedral mesh for an octahedron:\n";
    geom::writeAscii(std::cout, mesh);
    std::cout << "\n";

    // copy constructor
    ISS copyConstructed(mesh);
    assert(mesh == copyConstructed);

    // assignment operator
    ISS assigned;
    assigned = mesh;
    assert(mesh == assigned);

    // Make a bbox
    BBox bb = geom::specificBBox<BBox>
      (mesh.vertices.begin(), mesh.vertices.end());
    assert((bb == BBox{{{-1., -1., -1.}}, {{1., 1., 1.}}}));

    // Clear the mesh.
    mesh.clear();
    assert(mesh.vertices.empty() &&
           mesh.indexedSimplices.empty());

    // Construct from a simplex set.
    {
      std::vector<Vertex> vert(num_tets * 4);
      for (std::size_t i = 0; i != vert.size(); ++i) {
        for (std::size_t n = 0; n != 3; ++n) {
          vert[i][n] = vertices[tets[i] + n];
        }
      }
      geom::IndSimpSet<3, 3> bm;
      geom::buildFromSimplices(vert.begin(), vert.end(), &bm);
      assert(bm.vertices.size() == num_vertices);
      assert(bm.indexedSimplices.size() == num_tets);
    }
  }



  //
  // 3-D space.  3-D simplex.  Allocate memory.
  //
  {
    typedef geom::IndSimpSet<3, 3> ISS;
    {
      // Default constructor.
      ISS x;
      assert(x.N == 3);
      assert(x.M == 3);
      assert(x.vertices.size() == 0);
      assert(x.indexedSimplices.size() == 0);
    }
    //
    // Data for an octahedron
    //
    const std::size_t num_vertices = 7;
    double vertices[] = { 0, 0, 0,
                          1, 0, 0,
                          -1, 0, 0,
                          0, 1, 0,
                          0, -1, 0,
                          0, 0, 1,
                          0, 0, -1
                        };
    const std::size_t num_tets = 8;
    std::size_t tets[] = { 0, 1, 3, 5,
                           0, 3, 2, 5,
                           0, 2, 4, 5,
                           0, 4, 1, 5,
                           0, 3, 1, 6,
                           0, 2, 3, 6,
                           0, 4, 2, 6,
                           0, 1, 4, 6
                         };

    // Construct from vertices and tetrahedra.
    ISS mesh;
    build(&mesh, num_vertices, vertices, num_tets, tets);

    // Copy constructor.
    ISS copyConstructed(mesh);
    assert(mesh == copyConstructed);

    // Assignment operator.
    ISS assigned;
    assigned = mesh;
    assert(mesh == assigned);

    // Write to a file in ascii format.
    {
      std::stringstream file;
      geom::writeAscii(file, mesh);
      // Read from a file in ascii format.
      ISS readMesh;
      geom::readAscii(file, &readMesh);
      assert(mesh == readMesh);
    }

    // Write to a file in binary format.
    {
      std::stringstream file;
      geom::writeBinary(file, mesh);
      // Read from a file in binary format.
      ISS readMesh;
      geom::readBinary(file, &readMesh);
      assert(mesh == readMesh);
    }
  }



  //
  // 3-D space.  2-D simplex.
  //
  {
    typedef geom::IndSimpSet<3, 2> ISS;
    typedef geom::BBox<double, 3> BBox;
    {
      // default constructor
      ISS x;
      assert(x.N == 3);
      assert(x.M == 2);
      assert(x.vertices.size() == 0);
      assert(x.indexedSimplices.size() == 0);
    }
    //
    // Data for an octahedron
    //
    const std::size_t numVertices = 6;
    double vertices[] = { 1, 0, 0,    // 0
                          -1, 0, 0,    // 1
                          0, 1, 0,     // 2
                          0, -1, 0,    // 3
                          0, 0, 1,     // 4
                          0, 0, -1
                        };  // 5
    const std::size_t numTriangles = 8;
    std::size_t triangles[] = { 0, 2, 4,
                                2, 0, 5,
                                2, 1, 4,
                                1, 2, 5,
                                1, 3, 4,
                                3, 1, 5,
                                3, 0, 4,
                                0, 3, 5
                              };

    // Construct from vertices and triangles.
    ISS mesh;
    build(&mesh, numVertices, vertices, numTriangles, triangles);

    // Print the mesh.
    std::cout << "Triangle mesh for an octahedron:\n";
    geom::writeAscii(std::cout, mesh);
    std::cout << "\n";

    // copy constructor
    ISS copyConstructed(mesh);
    assert(mesh == copyConstructed);

    // assignment operator
    ISS assigned;
    assigned = mesh;
    assert(mesh == assigned);

    // Make a bbox
    BBox bb = geom::specificBBox<BBox>
      (mesh.vertices.begin(), mesh.vertices.end());
    assert((bb == BBox{{{-1., -1., -1.}}, {{1., 1., 1.}}}));
  }

  return 0;
}
