// -*- C++ -*-

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"

#include "stlib/geom/mesh/iss/file_io.h"
#include "stlib/geom/mesh/iss/equality.h"
#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/buildFromSimplices.h"

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
    typedef geom::IndSimpSetIncAdj<3, 3> ISS;
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
      assert(x.computeFacesSize() == 0);
    }
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
    const std::size_t numFaces = 20;

    // Construct from vertices and tetrahedra.
    ISS mesh;
    build(&mesh, numVertices, vertices, numTets, tets);

    assert(mesh.N == 3);
    assert(mesh.M == 3);
    assert(mesh.vertices.size() == numVertices);
    assert(mesh.indexedSimplices.size() == numTets);
    assert(mesh.computeFacesSize() == numFaces);

    {
      Simplex s;
      SimplexConstIterator i = mesh.getSimplicesBegin();
      for (std::size_t n = 0; n != numTets; ++n, ++i) {
        s = *i;
        for (std::size_t m = 0; m != 4; ++m) {
          assert(s[m] == mesh.getSimplexVertex(n, m));
        }
      }
    }

    // copy constructor
    ISS copy_constructed(mesh);
    assert(mesh == copy_constructed);

    // assignment operator
    ISS assigned;
    assigned = mesh;
    assert(mesh == assigned);

    // Make a bbox
    BBox bb = geom::specificBBox<BBox>
      (mesh.vertices.begin(), mesh.vertices.end());
    assert((bb == BBox{{{-1., -1., -1.}}, {{1., 1., 1.}}}));

    // Construct from a simplex set.
    {
      std::vector<Vertex> vert(numTets * 4);
      for (std::size_t i = 0; i != vert.size(); ++i) {
        for (std::size_t n = 0; n != 3; ++n) {
          vert[i][n] = vertices[ tets[i] + n ];
        }
      }
      geom::IndSimpSetIncAdj<3, 3> bm;
      geom::buildFromSimplices(vert.begin(), vert.end(), &bm);
      assert(bm.vertices.size() == numVertices);
      assert(bm.indexedSimplices.size() == numTets);
      assert(bm.computeFacesSize() == numFaces);
    }
  }



  //
  // 3-D space.  3-D simplex.  Allocate memory.
  //
  {
    typedef geom::IndSimpSetIncAdj<3, 3> ISS;
    {
      // Default constructor.
      ISS x;
      assert(x.N == 3);
      assert(x.M == 3);
      assert(x.vertices.size() == 0);
      assert(x.indexedSimplices.size() == 0);
      assert(x.computeFacesSize() == 0);
    }
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
    const std::size_t numFaces = 20;

    // Construct from vertices and tetrahedra.
    ISS mesh;
    build(&mesh, numVertices, vertices, numTets, tets);

    assert(mesh.N == 3);
    assert(mesh.M == 3);
    assert(mesh.vertices.size() == numVertices);
    assert(mesh.indexedSimplices.size() == numTets);
    assert(mesh.computeFacesSize() == numFaces);

    // Copy constructor.
    ISS copy_constructed(mesh);
    assert(mesh == copy_constructed);

    // Assignment operator.
    ISS assigned;
    assigned = mesh;
    assert(mesh == assigned);
  }



  //
  // 3-D space.  2-D simplex.
  //
  {
    typedef geom::IndSimpSetIncAdj<3, 2> ISS;
    typedef geom::BBox<double, 3> BBox;
    {
      // default constructor
      ISS x;
      assert(x.N == 3);
      assert(x.M == 2);
      assert(x.vertices.size() == 0);
      assert(x.indexedSimplices.size() == 0);
      assert(x.computeFacesSize() == 0);
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
    const std::size_t num_triangles = 8;
    std::size_t triangles[] = { 0, 2, 4,
                                2, 0, 5,
                                2, 1, 4,
                                1, 2, 5,
                                1, 3, 4,
                                3, 1, 5,
                                3, 0, 4,
                                0, 3, 5
                              };
    const std::size_t numFaces = 12;

    // Construct from vertices and triangles.
    ISS mesh;
    build(&mesh, numVertices, vertices, num_triangles, triangles);

    assert(mesh.N == 3);
    assert(mesh.M == 2);
    assert(mesh.vertices.size() == numVertices);
    assert(mesh.indexedSimplices.size() == num_triangles);
    assert(mesh.computeFacesSize() == numFaces);

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

    // Clear the mesh.
    mesh.clear();
    assert(mesh.vertices.empty() &&
           mesh.indexedSimplices.empty() &&
           mesh.incident.empty() &&
           mesh.adjacent.empty());

  }

  //
  // Incident to edge test.
  //
  {
    // Data for an octahedron
    const std::size_t numVertices = 7;
    const double vertices[] = {0, 0, 0,
                               1, 0, 0,
                               -1, 0, 0,
                               0, 1, 0,
                               0, -1, 0,
                               0, 0, 1,
                               0, 0, -1
                              };
    const std::size_t numTets = 8;
    const std::size_t tets[] = {0, 1, 3, 5,
                                0, 3, 2, 5,
                                0, 2, 4, 5,
                                0, 4, 1, 5,
                                0, 3, 1, 6,
                                0, 2, 3, 6,
                                0, 4, 2, 6,
                                0, 1, 4, 6
                               };

    // Construct from vertices and tetrahedra.
    const std::size_t N = 3;
    const std::size_t M = 3;
    geom::IndSimpSetIncAdj<N, M> mesh;
    build(&mesh, numVertices, vertices, numTets, tets);

    std::vector<std::size_t> edgeIncident;
    for (std::size_t i = 0; i != mesh.indexedSimplices.size(); ++i) {
      // The edges that use vertex 0 have 4 incident simplices.
      for (std::size_t j = 1; j != M + 1; ++j) {
        // The intersection of the simplices that are incident to the
        // two vertices are incident to the edge.
        edgeIncident.clear();
        // The two vertices that define the edge.
        const std::size_t a = mesh.indexedSimplices[i][0];
        const std::size_t b = mesh.indexedSimplices[i][j];
        // Take the intersection of the two sorted sequences.
        std::set_intersection(mesh.incident.begin(a),
                              mesh.incident.end(a),
                              mesh.incident.begin(b),
                              mesh.incident.end(b),
                              std::back_inserter(edgeIncident));
        assert(edgeIncident.size() == 4);
      }
      // The other edges 2 incident simplices.
      for (std::size_t j = 1; j != M + 1; ++j) {
        for (std::size_t k = j + 1; k != M + 1; ++k) {
          // The intersection of the simplices that are incident to the
          // two vertices are incident to the edge.
          edgeIncident.clear();
          // The two vertices that define the edge.
          const std::size_t a = mesh.indexedSimplices[i][j];
          const std::size_t b = mesh.indexedSimplices[i][k];
          // Take the intersection of the two sorted sequences.
          std::set_intersection(mesh.incident.begin(a),
                                mesh.incident.end(a),
                                mesh.incident.begin(b),
                                mesh.incident.end(b),
                                std::back_inserter(edgeIncident));
          assert(edgeIncident.size() == 2);
        }
      }
    }
  }

  return 0;
}
