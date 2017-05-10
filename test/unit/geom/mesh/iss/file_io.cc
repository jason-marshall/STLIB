// -*- C++ -*-

#include "stlib/geom/mesh/iss/file_io.h"
#include "stlib/geom/mesh/iss/build.h"

#include "stlib/geom/mesh/iss/equality.h"

#include <iostream>
#include <sstream>

#include <cassert>

using namespace stlib;

int
main()
{
  //
  // 2-D space, 2-D simplex.
  //
  {
    typedef geom::IndSimpSet<2, 2> Mesh;

    const std::size_t numVertices = 5;
    double vertices[] = {0, 0,
                         1, 0,
                         0, 1,
                         -1, 0,
                         0, -1
                        };
    const std::size_t numSimplices = 4;
    std::size_t indexedSimplices[] = {0, 1, 2,
                                      0, 2, 3,
                                      0, 3, 4,
                                      0, 4, 1
                                     };

    Mesh mesh;
    build(&mesh, numVertices, vertices, numSimplices, indexedSimplices);
    assert(mesh.vertices.size() == 5);
    assert(mesh.indexedSimplices.size() == 4);

    {
      std::stringstream file;
      writeAscii(file, mesh);
      Mesh x;
      readAscii(file, &x);
      assert(x == mesh);
    }

    {
      std::stringstream file;
      writeBinary(file, mesh);
      Mesh x;
      readBinary(file, &x);
      assert(x == mesh);
    }

    {
      std::cout << "VTK XML\n";
      writeVtkXml(std::cout, mesh);
    }
    {
      std::cout << "VTK Legacy\n";
      writeVtkLegacy(std::cout, mesh);
    }

  }

  return 0;
}
