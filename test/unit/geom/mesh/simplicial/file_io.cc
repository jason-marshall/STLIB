// -*- C++ -*-

#include "stlib/geom/mesh/simplicial/file_io.h"
#include "stlib/geom/mesh/iss/build.h"

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
    typedef geom::SimpMeshRed<3, 3> SMR;

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
    {
      // Construct an ISS from vertices and tetrahedra.
      ISS iss;
      build(&iss, numVertices, vertices, numTets, tets);
      SMR mesh(iss);

      // Write to a file in ascii format.
      {
        std::stringstream file;
        geom::writeAscii(file, mesh);
        // Read from a file in ascii format.
        SMR readMesh;
        geom::readAscii(file, &readMesh);
        assert(mesh.computeNodesSize() == readMesh.computeNodesSize());
        assert(mesh.computeCellsSize() == readMesh.computeCellsSize());
      }

      // Write to a file in binary format.
      {
        std::stringstream file;
        geom::writeBinary(file, mesh);
        // Read from a file in binary format.
        SMR readMesh;
        geom::readBinary(file, &readMesh);
        assert(mesh.computeNodesSize() == readMesh.computeNodesSize());
        assert(mesh.computeCellsSize() == readMesh.computeCellsSize());
      }

      // Write to a file in VTK format.
      {
        std::stringstream file;
        geom::writeVtkXml(file, mesh);
        // Write to a file in VTK legacy format.
        geom::writeVtkLegacy(file, mesh);
      }
    }
  }




  //
  // 3-D space.  2-D simplex.
  //
  {
    typedef geom::IndSimpSet<3, 2> ISS;
    typedef geom::SimpMeshRed<3, 2> SMR;

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
    {
      // Construct an ISS from vertices and tetrahedra.
      ISS iss;
      build(&iss, numVertices, vertices, numTriangles, triangles);
      SMR mesh(iss);

      // Write to a file in ascii format.
      {
        std::stringstream file;
        geom::writeAscii(file, mesh);
        // Read from a file in ascii format.
        SMR readMesh;
        geom::readAscii(file, &readMesh);
        assert(mesh.computeNodesSize() == readMesh.computeNodesSize());
        assert(mesh.computeCellsSize() == readMesh.computeCellsSize());
      }

      // Write to a file in binary format.
      {
        std::stringstream file;
        geom::writeBinary(file, mesh);
        // Read from a file in binary format.
        SMR readMesh;
        geom::readBinary(file, &readMesh);
        assert(mesh.computeNodesSize() == readMesh.computeNodesSize());
        assert(mesh.computeCellsSize() == readMesh.computeCellsSize());
      }

      // Write to a file in VTK format.
      {
        std::stringstream file;
        geom::writeVtkXml(file, mesh);
        // Write to a file in VTK legacy format.
        geom::writeVtkLegacy(file, mesh);
      }
    }
  }

  return 0;
}
