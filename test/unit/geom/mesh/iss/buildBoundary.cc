// -*- C++ -*-

#include "stlib/geom/mesh/iss/build.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  //--------------------------------------------------------------------------
  //--------------------------------------------------------------------------
  // 2-2 meshes.
  //--------------------------------------------------------------------------
  //--------------------------------------------------------------------------
  {
    typedef geom::IndSimpSetIncAdj<2, 2> Mesh;
    typedef geom::IndSimpSetIncAdj<2, 1> Boundary;
    typedef Mesh::Vertex Vertex;
    typedef Mesh::IndexedSimplex IndexedSimplex;
    typedef Mesh::IndexedSimplexFace IndexedSimplexFace;

    //------------------------------------------------------------------------
    // Empty mesh.
    //------------------------------------------------------------------------
    {
      Mesh mesh;
      Boundary boundary;

      // buildBoundary
      {
        geom::buildBoundary(mesh, &boundary);
        assert(boundary.empty());
      }
      // buildBoundary
      {
        std::vector<std::size_t> vertexIndices;
        geom::buildBoundary(mesh, &boundary,
                            std::back_inserter(vertexIndices));
        assert(boundary.empty());
        assert(vertexIndices.empty());
      }
      // buildBoundaryWithoutPacking
      {
        geom::buildBoundaryWithoutPacking(mesh, &boundary);
        assert(boundary.empty());
      }
      // buildBoundaryWithoutPacking
      {
        std::vector<std::size_t> incidentSimplices;
        geom::buildBoundaryWithoutPacking(mesh, &boundary,
                                          std::back_inserter(incidentSimplices));
        assert(boundary.empty());
        assert(incidentSimplices.empty());
      }
      // buildBoundaryOfComponents
      {
        std::vector<std::size_t> delimiters;
        geom::buildBoundaryOfComponents(mesh, &boundary,
                                        std::back_inserter(delimiters));
        assert(boundary.empty());
        assert(delimiters.empty());
      }
      // buildBoundaryOfComponents
      {
        std::vector<std::size_t> delimiters;
        std::vector<std::size_t> incidentSimplices;
        geom::buildBoundaryOfComponents(mesh, &boundary,
                                        std::back_inserter(delimiters),
                                        std::back_inserter(incidentSimplices));
        assert(boundary.empty());
        assert(delimiters.empty());
        assert(incidentSimplices.empty());
      }
      // buildBoundaryOfComponentsWithoutPacking
      {
        std::vector<std::size_t> delimiters;
        geom::buildBoundaryOfComponentsWithoutPacking
        (mesh, &boundary, std::back_inserter(delimiters));
        assert(boundary.empty());
        assert(delimiters.empty());
      }
      // buildBoundaryOfComponentsWithoutPacking
      {
        std::vector<std::size_t> delimiters;
        std::vector<std::size_t> incidentSimplices;
        geom::buildBoundaryOfComponentsWithoutPacking
        (mesh, &boundary, std::back_inserter(delimiters),
         std::back_inserter(incidentSimplices));
        assert(boundary.empty());
        assert(delimiters.empty());
        assert(incidentSimplices.empty());
      }
    }

    //------------------------------------------------------------------------
    // Mesh with one triangle.
    //------------------------------------------------------------------------
    {
      const std::size_t NumberOfVertices = 3;
      std::vector<Vertex> vertices(NumberOfVertices);
      vertices[0] = Vertex{{0., 0.}};
      vertices[1] = Vertex{{1., 0.}};
      vertices[2] = Vertex{{0., 1.}};
      const std::size_t NumberOfSimplices = 1;
      std::vector<IndexedSimplex> indexedSimplices(NumberOfSimplices);
      indexedSimplices[0] = IndexedSimplex{{0, 1, 2}};

      Mesh mesh(vertices, indexedSimplices);
      assert(mesh.vertices.size() == NumberOfVertices);
      assert(mesh.indexedSimplices.size() == NumberOfSimplices);
      Boundary boundary;

      // buildBoundary
      {
        geom::buildBoundary(mesh, &boundary);

        assert(boundary.vertices.size() == 3);
        assert(boundary.indexedSimplices.size() == 3);
        for (std::size_t n = 0; n != 3; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));
      }
      // buildBoundary
      {
        std::vector<std::size_t> vertexIndices;
        geom::buildBoundary(mesh, &boundary,
                            std::back_inserter(vertexIndices));

        assert(boundary.vertices.size() == 3);
        assert(boundary.indexedSimplices.size() == 3);
        for (std::size_t n = 0; n != 3; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));

        assert(vertexIndices.size() == 3);
        for (std::size_t n = 0; n != 3; ++n) {
          assert(vertexIndices[n] == n);
        }
      }
      // buildBoundaryWithoutPacking
      {
        geom::buildBoundaryWithoutPacking(mesh, &boundary);

        assert(boundary.vertices.size() == 3);
        assert(boundary.indexedSimplices.size() == 3);
        for (std::size_t n = 0; n != 3; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));
      }
      // buildBoundaryWithoutPacking
      {
        std::vector<std::size_t> incidentSimplices;
        geom::buildBoundaryWithoutPacking(mesh, &boundary,
                                          std::back_inserter(incidentSimplices));

        assert(boundary.vertices.size() == 3);
        assert(boundary.indexedSimplices.size() == 3);
        for (std::size_t n = 0; n != 3; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));

        assert(incidentSimplices.size() == 3);
        for (std::size_t n = 0; n != 3; ++n) {
          assert(incidentSimplices[n] == 0);
        }
      }
      // buildBoundaryOfComponents
      {
        std::vector<std::size_t> delimiters;
        geom::buildBoundaryOfComponents(mesh, &boundary,
                                        std::back_inserter(delimiters));

        assert(boundary.vertices.size() == 3);
        assert(boundary.indexedSimplices.size() == 3);
        for (std::size_t n = 0; n != 3; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));

        assert(delimiters.size() == 2);
        assert(delimiters[0] == 0);
        assert(delimiters[1] == 3);
      }
      // buildBoundaryOfComponents
      {
        std::vector<std::size_t> delimiters;
        std::vector<std::size_t> incidentSimplices;
        geom::buildBoundaryOfComponents(mesh, &boundary,
                                        std::back_inserter(delimiters),
                                        std::back_inserter(incidentSimplices));

        assert(boundary.vertices.size() == 3);
        assert(boundary.indexedSimplices.size() == 3);
        for (std::size_t n = 0; n != 3; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));

        assert(delimiters.size() == 2);
        assert(delimiters[0] == 0);
        assert(delimiters[1] == 3);

        assert(incidentSimplices.size() == 3);
        for (std::size_t n = 0; n != 3; ++n) {
          assert(incidentSimplices[n] == 0);
        }
      }
      // buildBoundaryOfComponentsWithoutPacking
      {
        std::vector<std::size_t> delimiters;
        geom::buildBoundaryOfComponentsWithoutPacking
        (mesh, &boundary, std::back_inserter(delimiters));

        assert(boundary.vertices.size() == 3);
        assert(boundary.indexedSimplices.size() == 3);
        for (std::size_t n = 0; n != 3; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));

        assert(delimiters.size() == 2);
        assert(delimiters[0] == 0);
        assert(delimiters[1] == 3);
      }
      // buildBoundaryOfComponentsWithoutPacking
      {
        std::vector<std::size_t> delimiters;
        std::vector<std::size_t> incidentSimplices;
        geom::buildBoundaryOfComponentsWithoutPacking
        (mesh, &boundary, std::back_inserter(delimiters),
         std::back_inserter(incidentSimplices));

        assert(boundary.vertices.size() == 3);
        assert(boundary.indexedSimplices.size() == 3);
        for (std::size_t n = 0; n != 3; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));

        assert(delimiters.size() == 2);
        assert(delimiters[0] == 0);
        assert(delimiters[1] == 3);

        assert(incidentSimplices.size() == 3);
        for (std::size_t n = 0; n != 3; ++n) {
          assert(incidentSimplices[n] == 0);
        }
      }
    }


    //------------------------------------------------------------------------
    // Mesh with two triangles and two components.
    //------------------------------------------------------------------------
    {
      const std::size_t NumberOfVertices = 6;
      std::vector<Vertex> vertices(NumberOfVertices);
      vertices[0] = Vertex{{0., 0.}};
      vertices[1] = Vertex{{1., 0.}};
      vertices[2] = Vertex{{0., 1.}};
      vertices[3] = Vertex{{2, 0}};
      vertices[4] = Vertex{{3, 0}};
      vertices[5] = Vertex{{2, 1}};
      const std::size_t NumberOfSimplices = 2;
      std::vector<IndexedSimplex> indexedSimplices(NumberOfSimplices);
      indexedSimplices[0] = IndexedSimplex{{0, 1, 2}};
      indexedSimplices[1] = IndexedSimplex{{3, 4, 5}};

      Mesh mesh(vertices, indexedSimplices);
      assert(mesh.vertices.size() == NumberOfVertices);
      assert(mesh.indexedSimplices.size() == NumberOfSimplices);
      Boundary boundary;

      // buildBoundary
      {
        geom::buildBoundary(mesh, &boundary);

        assert(boundary.vertices.size() == 6);
        assert(boundary.indexedSimplices.size() == 6);
        for (std::size_t n = 0; n != 6; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));
        assert(boundary.indexedSimplices[3] == (IndexedSimplexFace{{4, 5}}));
        assert(boundary.indexedSimplices[4] == (IndexedSimplexFace{{5, 3}}));
        assert(boundary.indexedSimplices[5] == (IndexedSimplexFace{{3, 4}}));
      }
      // buildBoundary
      {
        std::vector<std::size_t> vertexIndices;
        geom::buildBoundary(mesh, &boundary,
                            std::back_inserter(vertexIndices));

        assert(boundary.vertices.size() == 6);
        assert(boundary.indexedSimplices.size() == 6);
        for (std::size_t n = 0; n != 6; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));
        assert(boundary.indexedSimplices[3] == (IndexedSimplexFace{{4, 5}}));
        assert(boundary.indexedSimplices[4] == (IndexedSimplexFace{{5, 3}}));
        assert(boundary.indexedSimplices[5] == (IndexedSimplexFace{{3, 4}}));

        assert(vertexIndices.size() == 6);
        for (std::size_t n = 0; n != 6; ++n) {
          assert(vertexIndices[n] == n);
        }
      }
      // buildBoundaryWithoutPacking
      {
        geom::buildBoundaryWithoutPacking(mesh, &boundary);

        assert(boundary.vertices.size() == 6);
        assert(boundary.indexedSimplices.size() == 6);
        for (std::size_t n = 0; n != 6; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));
        assert(boundary.indexedSimplices[3] == (IndexedSimplexFace{{4, 5}}));
        assert(boundary.indexedSimplices[4] == (IndexedSimplexFace{{5, 3}}));
        assert(boundary.indexedSimplices[5] == (IndexedSimplexFace{{3, 4}}));
      }
      // buildBoundaryWithoutPacking
      {
        std::vector<std::size_t> incidentSimplices;
        geom::buildBoundaryWithoutPacking(mesh, &boundary,
                                          std::back_inserter(incidentSimplices));

        assert(boundary.vertices.size() == 6);
        assert(boundary.indexedSimplices.size() == 6);
        for (std::size_t n = 0; n != 6; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));
        assert(boundary.indexedSimplices[3] == (IndexedSimplexFace{{4, 5}}));
        assert(boundary.indexedSimplices[4] == (IndexedSimplexFace{{5, 3}}));
        assert(boundary.indexedSimplices[5] == (IndexedSimplexFace{{3, 4}}));

        assert(incidentSimplices.size() == 6);
        assert(incidentSimplices[0] == 0);
        assert(incidentSimplices[1] == 0);
        assert(incidentSimplices[2] == 0);
        assert(incidentSimplices[3] == 1);
        assert(incidentSimplices[4] == 1);
        assert(incidentSimplices[5] == 1);
      }
      // buildBoundaryOfComponents
      {
        std::vector<std::size_t> delimiters;
        geom::buildBoundaryOfComponents(mesh, &boundary,
                                        std::back_inserter(delimiters));

        assert(boundary.vertices.size() == 6);
        assert(boundary.indexedSimplices.size() == 6);
        for (std::size_t n = 0; n != 6; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));
        assert(boundary.indexedSimplices[3] == (IndexedSimplexFace{{4, 5}}));
        assert(boundary.indexedSimplices[4] == (IndexedSimplexFace{{5, 3}}));
        assert(boundary.indexedSimplices[5] == (IndexedSimplexFace{{3, 4}}));

        assert(delimiters.size() == 3);
        assert(delimiters[0] == 0);
        assert(delimiters[1] == 3);
        assert(delimiters[2] == 6);
      }
      // buildBoundaryOfComponents
      {
        std::vector<std::size_t> delimiters;
        std::vector<std::size_t> incidentSimplices;
        geom::buildBoundaryOfComponents(mesh, &boundary,
                                        std::back_inserter(delimiters),
                                        std::back_inserter(incidentSimplices));

        assert(boundary.vertices.size() == 6);
        assert(boundary.indexedSimplices.size() == 6);
        for (std::size_t n = 0; n != 6; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));
        assert(boundary.indexedSimplices[3] == (IndexedSimplexFace{{4, 5}}));
        assert(boundary.indexedSimplices[4] == (IndexedSimplexFace{{5, 3}}));
        assert(boundary.indexedSimplices[5] == (IndexedSimplexFace{{3, 4}}));

        assert(delimiters.size() == 3);
        assert(delimiters[0] == 0);
        assert(delimiters[1] == 3);
        assert(delimiters[2] == 6);

        assert(incidentSimplices.size() == 6);
        assert(incidentSimplices[0] == 0);
        assert(incidentSimplices[1] == 0);
        assert(incidentSimplices[2] == 0);
        assert(incidentSimplices[3] == 1);
        assert(incidentSimplices[4] == 1);
        assert(incidentSimplices[5] == 1);
      }
      // buildBoundaryOfComponentsWithoutPacking
      {
        std::vector<std::size_t> delimiters;
        geom::buildBoundaryOfComponentsWithoutPacking
        (mesh, &boundary, std::back_inserter(delimiters));

        assert(boundary.vertices.size() == 6);
        assert(boundary.indexedSimplices.size() == 6);
        for (std::size_t n = 0; n != 6; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));
        assert(boundary.indexedSimplices[3] == (IndexedSimplexFace{{4, 5}}));
        assert(boundary.indexedSimplices[4] == (IndexedSimplexFace{{5, 3}}));
        assert(boundary.indexedSimplices[5] == (IndexedSimplexFace{{3, 4}}));

        assert(delimiters.size() == 3);
        assert(delimiters[0] == 0);
        assert(delimiters[1] == 3);
        assert(delimiters[2] == 6);
      }
      // buildBoundaryOfComponentsWithoutPacking
      {
        std::vector<std::size_t> delimiters;
        std::vector<std::size_t> incidentSimplices;
        geom::buildBoundaryOfComponentsWithoutPacking
        (mesh, &boundary, std::back_inserter(delimiters),
         std::back_inserter(incidentSimplices));

        assert(boundary.vertices.size() == 6);
        assert(boundary.indexedSimplices.size() == 6);
        for (std::size_t n = 0; n != 6; ++n) {
          assert(boundary.vertices[n] == vertices[n]);
        }
        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 0}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{0, 1}}));
        assert(boundary.indexedSimplices[3] == (IndexedSimplexFace{{4, 5}}));
        assert(boundary.indexedSimplices[4] == (IndexedSimplexFace{{5, 3}}));
        assert(boundary.indexedSimplices[5] == (IndexedSimplexFace{{3, 4}}));

        assert(delimiters.size() == 3);
        assert(delimiters[0] == 0);
        assert(delimiters[1] == 3);
        assert(delimiters[2] == 6);

        assert(incidentSimplices.size() == 6);
        assert(incidentSimplices[0] == 0);
        assert(incidentSimplices[1] == 0);
        assert(incidentSimplices[2] == 0);
        assert(incidentSimplices[3] == 1);
        assert(incidentSimplices[4] == 1);
        assert(incidentSimplices[5] == 1);
      }
    }


    //------------------------------------------------------------------------
    // Mesh with three triangles and one component.
    //------------------------------------------------------------------------
    {
      const std::size_t NumberOfVertices = 4;
      std::vector<Vertex> vertices(NumberOfVertices);
      vertices[0] = Vertex{{0., 0.}};
      vertices[1] = Vertex{{1., 0.}};
      vertices[2] = Vertex{{-1, 1}};
      vertices[3] = Vertex{{-1, -1}};
      const std::size_t NumberOfSimplices = 3;
      std::vector<IndexedSimplex> indexedSimplices(NumberOfSimplices);
      indexedSimplices[0] = IndexedSimplex{{0, 1, 2}};
      indexedSimplices[1] = IndexedSimplex{{0, 2, 3}};
      indexedSimplices[2] = IndexedSimplex{{0, 3, 1}};

      Mesh mesh(vertices, indexedSimplices);
      assert(mesh.vertices.size() == NumberOfVertices);
      assert(mesh.indexedSimplices.size() == NumberOfSimplices);
      Boundary boundary;

      // buildBoundary
      {
        geom::buildBoundary(mesh, &boundary);

        assert(boundary.vertices.size() == 3);
        assert(boundary.indexedSimplices.size() == 3);
        assert(boundary.vertices[0] == vertices[1]);
        assert(boundary.vertices[1] == vertices[2]);
        assert(boundary.vertices[2] == vertices[3]);

        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{0, 1}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{2, 0}}));
      }
      // buildBoundary
      {
        std::vector<std::size_t> vertexIndices;
        geom::buildBoundary(mesh, &boundary,
                            std::back_inserter(vertexIndices));

        assert(boundary.vertices.size() == 3);
        assert(boundary.indexedSimplices.size() == 3);
        assert(boundary.vertices[0] == vertices[1]);
        assert(boundary.vertices[1] == vertices[2]);
        assert(boundary.vertices[2] == vertices[3]);

        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{0, 1}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{2, 0}}));

        assert(vertexIndices.size() == 3);
        assert(vertexIndices[0] == 1);
        assert(vertexIndices[1] == 2);
        assert(vertexIndices[2] == 3);
      }
      // buildBoundaryWithoutPacking
      {
        geom::buildBoundaryWithoutPacking(mesh, &boundary);

        assert(boundary.vertices.size() == 4);
        assert(boundary.indexedSimplices.size() == 3);
        assert(boundary.vertices[0] == vertices[0]);
        assert(boundary.vertices[1] == vertices[1]);
        assert(boundary.vertices[2] == vertices[2]);
        assert(boundary.vertices[3] == vertices[3]);

        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 3}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{3, 1}}));
      }
      // buildBoundaryWithoutPacking
      {
        std::vector<std::size_t> incidentSimplices;
        geom::buildBoundaryWithoutPacking(mesh, &boundary,
                                          std::back_inserter(incidentSimplices));

        assert(boundary.vertices.size() == 4);
        assert(boundary.indexedSimplices.size() == 3);
        assert(boundary.vertices[0] == vertices[0]);
        assert(boundary.vertices[1] == vertices[1]);
        assert(boundary.vertices[2] == vertices[2]);
        assert(boundary.vertices[3] == vertices[3]);

        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 3}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{3, 1}}));

        assert(incidentSimplices.size() == 3);
        assert(incidentSimplices[0] == 0);
        assert(incidentSimplices[1] == 1);
        assert(incidentSimplices[2] == 2);
      }
      // buildBoundaryOfComponents
      {
        std::vector<std::size_t> delimiters;
        geom::buildBoundaryOfComponents(mesh, &boundary,
                                        std::back_inserter(delimiters));

        assert(boundary.vertices.size() == 3);
        assert(boundary.indexedSimplices.size() == 3);
        assert(boundary.vertices[0] == vertices[1]);
        assert(boundary.vertices[1] == vertices[2]);
        assert(boundary.vertices[2] == vertices[3]);

        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{0, 1}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{2, 0}}));

        assert(delimiters.size() == 2);
        assert(delimiters[0] == 0);
        assert(delimiters[1] == 3);
      }
      // buildBoundaryOfComponents
      {
        std::vector<std::size_t> delimiters;
        std::vector<std::size_t> incidentSimplices;
        geom::buildBoundaryOfComponents(mesh, &boundary,
                                        std::back_inserter(delimiters),
                                        std::back_inserter(incidentSimplices));

        assert(boundary.vertices.size() == 3);
        assert(boundary.indexedSimplices.size() == 3);
        assert(boundary.vertices[0] == vertices[1]);
        assert(boundary.vertices[1] == vertices[2]);
        assert(boundary.vertices[2] == vertices[3]);

        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{0, 1}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{2, 0}}));

        assert(delimiters.size() == 2);
        assert(delimiters[0] == 0);
        assert(delimiters[1] == 3);

        assert(incidentSimplices.size() == 3);
        assert(incidentSimplices[0] == 0);
        assert(incidentSimplices[1] == 1);
        assert(incidentSimplices[2] == 2);
      }
      // buildBoundaryOfComponentsWithoutPacking
      {
        std::vector<std::size_t> delimiters;
        geom::buildBoundaryOfComponentsWithoutPacking
        (mesh, &boundary, std::back_inserter(delimiters));

        assert(boundary.vertices.size() == 4);
        assert(boundary.indexedSimplices.size() == 3);
        assert(boundary.vertices[0] == vertices[0]);
        assert(boundary.vertices[1] == vertices[1]);
        assert(boundary.vertices[2] == vertices[2]);
        assert(boundary.vertices[3] == vertices[3]);

        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 3}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{3, 1}}));

        assert(delimiters.size() == 2);
        assert(delimiters[0] == 0);
        assert(delimiters[1] == 3);
      }
      // buildBoundaryOfComponentsWithoutPacking
      {
        std::vector<std::size_t> delimiters;
        std::vector<std::size_t> incidentSimplices;
        geom::buildBoundaryOfComponentsWithoutPacking
        (mesh, &boundary, std::back_inserter(delimiters),
         std::back_inserter(incidentSimplices));

        assert(boundary.vertices.size() == 4);
        assert(boundary.indexedSimplices.size() == 3);
        assert(boundary.vertices[0] == vertices[0]);
        assert(boundary.vertices[1] == vertices[1]);
        assert(boundary.vertices[2] == vertices[2]);
        assert(boundary.vertices[3] == vertices[3]);

        assert(boundary.indexedSimplices[0] == (IndexedSimplexFace{{1, 2}}));
        assert(boundary.indexedSimplices[1] == (IndexedSimplexFace{{2, 3}}));
        assert(boundary.indexedSimplices[2] == (IndexedSimplexFace{{3, 1}}));

        assert(delimiters.size() == 2);
        assert(delimiters[0] == 0);
        assert(delimiters[1] == 3);

        assert(incidentSimplices.size() == 3);
        assert(incidentSimplices[0] == 0);
        assert(incidentSimplices[1] == 1);
        assert(incidentSimplices[2] == 2);
      }
    }
  }

  return 0;
}
