// -*- C++ -*-

#include "stlib/geom/mesh/quadrilateral/QuadMesh.h"

#include "stlib/geom/kernel/BBox.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  //
  // 3-D space.
  //
  {
    typedef geom::QuadMesh<3> Mesh;
    typedef geom::BBox<double, 3> BBox;
    {
      // default constructor
      Mesh x;
      assert(x.getSpaceDimension() == 3);
      assert(x.getVerticesSize() == 0);
      assert(x.getFacesSize() == 0);
    }
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
    const std::size_t numberOfVertices =
      sizeof(vertices) / (3 * sizeof(double));
    std::size_t faces[] = {0, 3, 2, 1,
                           4, 5, 6, 7,
                           1, 2, 6, 5,
                           0, 4, 7, 3,
                           0, 1, 5, 4,
                           2, 3, 7, 6
                          };
    const std::size_t numberOfFaces = sizeof(faces) / (4 * sizeof(std::size_t));

    // Construct from vertices and faces.
    Mesh mesh(numberOfVertices, vertices, numberOfFaces, faces);

    // Print the mesh.
    // CONTINUE
#if 0
    std::cout << "Quadrilateral mesh for a cube:\n";
    geom::writeAscii(std::cout, mesh);
    std::cout << "\n";
#endif

    // copy constructor
    Mesh copyConstructed(mesh);
    //assert(mesh == copyConstructed);

    // assignment operator
    Mesh assigned;
    assigned = mesh;
    //assert(mesh == assigned);

    // Make a bbox
    BBox bb = geom::specificBBox<BBox>(mesh.getVerticesBeginning(),
                                       mesh.getVerticesEnd());
    assert((bb == BBox{{{0., 0., 0.}}, {{1., 1., 1.}}}));
  }

  return 0;
}
