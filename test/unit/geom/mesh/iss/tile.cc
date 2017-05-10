// -*- C++ -*-

#include "stlib/geom/mesh/iss/tile.h"

#include "stlib/geom/mesh/iss/file_io.h"
#include "stlib/geom/mesh/iss/quality.h"

#include <iostream>

#include <cassert>

using namespace stlib;

#define PT ext::make_array<double>

int
main()
{
  //
  // 2-D
  //

  {
    typedef geom::IndSimpSet<2, 2> ISS;

    {
      geom::BBox<double, 2> domain = {{{0, 0}}, {{1, 1}}};
      double length = 1;
      ISS mesh;

      geom::tile(domain, length, &mesh);
      geom::writeAscii(std::cout, mesh);
      std::cout << "\n";

      geom::printQualityStatistics(std::cout, mesh);
    }

    {
      geom::BBox<double, 2> domain = {{{0, 0}}, {{1, 2}}};
      double length = 1;
      ISS mesh;

      geom::tile(domain, length, &mesh);
      geom::writeAscii(std::cout, mesh);
      std::cout << "\n";
    }
  }

  //
  // 3-D
  //
  {
    typedef geom::IndSimpSet<3, 3> ISS;

    {
      geom::BBox<double, 3> domain = {{{0, 0, 0}}, {{0.1, 0.1, 0.1}}};
      double length = 1;
      ISS mesh;

      geom::tile(domain, length, &mesh);
      geom::writeAscii(std::cout, mesh);
      std::cout << "\n";

      geom::printQualityStatistics(std::cout, mesh);

      {
        std::cout << "VTK Legacy\n";
        geom::writeVtkLegacy(std::cout, mesh);
      }
      {
        std::cout << "VTK XML\n";
        geom::writeVtkXml(std::cout, mesh);
      }
    }

    {
      geom::BBox<double, 3> domain = {{{0, 0, 0}}, {{1, 1, 1}}};
      double length = 1;
      ISS mesh;

      geom::tile(domain, length, &mesh);

      geom::printQualityStatistics(std::cout, mesh);

      std::cout << "VTK XML\n";
      geom::writeVtkXml(std::cout, mesh);
    }
  }

  return 0;
}
