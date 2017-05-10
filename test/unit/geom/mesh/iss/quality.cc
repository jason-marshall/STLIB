// -*- C++ -*-

#include "stlib/geom/mesh/iss/quality.h"
#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/file_io.h"

#include <iostream>
#include <sstream>

#include <cassert>

using namespace stlib;

int
main()
{
  typedef geom::IndSimpSet<3, 3> Mesh;

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
    // Construct an Mesh from vertices and tetrahedra.
    Mesh mesh;
    build(&mesh, numVertices, vertices, numTets, tets);

    // Content.
    std::cout << "Content = " << geom::computeContent(mesh) << '\n';

    {
      double min, max, mean;
      geom::computeContentStatistics(mesh, &min, &max, &mean);
      std::cout << "Content statistics: " << min << " " << max << " "
                << mean << "\n";
    }

    {
      double min, max, mean;
      geom::computeDeterminantStatistics(mesh, &min, &max, &mean);
      std::cout << "Determinant statistics: " << min << " " << max << " "
                << mean << "\n";
    }

    {
      double min, max, mean;
      geom::computeModifiedMeanRatioStatistics(mesh, &min, &max, &mean);
      std::cout << "Mod mean ratio statistics: " << min << " " << max << " "
                << mean << "\n";
    }

    {
      double min, max, mean;
      geom::computeModifiedConditionNumberStatistics(mesh, &min, &max, &mean);
      std::cout << "Mod cond num statistics: " << min << " " << max << " "
                << mean << "\n";
    }

    {
      double min_c, max_c, mean_c;
      double min_d, max_d, mean_d;
      double min_mr, max_mr, mean_mr;
      double min_cn, max_cn, mean_cn;
      geom::computeQualityStatistics(mesh, &min_c, &max_c, &mean_c,
                                     &min_d, &max_d, &mean_d,
                                     &min_mr, &max_mr, &mean_mr,
                                     &min_cn, &max_cn, &mean_cn);
      std::cout << "Content statistics: " << min_c << " " << max_c << " "
                << mean_c << "\n"
                << "Determinant statistics: " << min_d << " " << max_d << " "
                << mean_d << "\n"
                << "Mod mean ratio statistics: " << min_mr << " "
                << max_mr << " " << mean_mr << "\n"
                << "Mod cond num statistics: " << min_cn << " "
                << max_cn << " " << mean_cn << "\n";
    }

    geom::printQualityStatistics(std::cout, mesh);
  }

  // identifyLowQualityWithCondNum()
  {
    Mesh mesh;
    // Equilateral.
    {
      // data/geom/mesh/33/singularities/equilateral.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0.5 0.28867513459481288225457439025098 0.81649658092772603273242802490196\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      // Length 0.
      // data/geom/mesh/33/singularities/equilateral_0.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "0 0 0\n"\
         "0 0 0\n"\
         "0 0 0\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // Inverted.
      // data/geom/mesh/33/singularities/equilateralInverted.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0.5 0.28867513459481288225457439025098 0.81649658092772603273242802490196\n"\
         "1\n"\
         "1 0 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    // Sliver.
    {
      // data/geom/mesh/33/singularities/sliver_0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0 1 0\n"\
         "1 1 0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      // data/geom/mesh/33/singularities/sliver_1e-100.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0 1 0\n"\
         "1 1 1e-100\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/sliver_0.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0 1 0\n"\
         "1 1 0\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/sliver_-0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0 1 0\n"\
         "1 1 -0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    // Cap.
    {
      // data/geom/mesh/33/singularities/cap_0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0.5 0.28867513459481288225457439025098 0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      // data/geom/mesh/33/singularities/cap_1e-100.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0.5 0.28867513459481288225457439025098 1e-100\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/cap_0.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0.5 0.28867513459481288225457439025098 0\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/cap_-0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0.5 0.28867513459481288225457439025098 -0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    // Spade.
    {
      // data/geom/mesh/33/singularities/spade_0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0.5 0 0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      // data/geom/mesh/33/singularities/spade_1e-100.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0.5 0 1e-100\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/spade_0.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0.5 0 0\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/spade_-0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0.5 0 -0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    // Wedge.
    {
      // data/geom/mesh/33/singularities/wedge_0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0 0 0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      // data/geom/mesh/33/singularities/wedge_1e-100.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0 0 1e-100\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/wedge_0.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0 0 0\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/wedge_-0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.86602540378443864676372317075294 0\n"\
         "0 0 -0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    // Splinter.
    {
      // data/geom/mesh/33/singularities/splinter_0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0 0.01 0\n"\
         "1 0.01 0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      // data/geom/mesh/33/singularities/splinter_1e-100.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0 1e-100 0\n"\
         "1 1e-100 1e-100\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/splinter_0.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/splinter_-0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0 0.01 0\n"\
         "1 0.01 -0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    // Spike.
    {
      // data/geom/mesh/33/singularities/spike_0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "0.01 0 0\n"\
         "0.005 0.86602540378443864676372317075294 0\n"\
         "0.005 0.28867513459481288225457439025098 0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      // data/geom/mesh/33/singularities/spike_1e-100.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1e-100 0 0\n"\
         "5e-101 0.86602540378443864676372317075294 0\n"\
         "5e-101 0.28867513459481288225457439025098 1e-100\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/spike_0.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "0 0 0\n"\
         "0 0.86602540378443864676372317075294 0\n"\
         "0 0.28867513459481288225457439025098 0\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/spike_-0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "0.01 0 0\n"\
         "0.005 0.86602540378443864676372317075294 0\n"\
         "0.005 0.28867513459481288225457439025098 -0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    // Spindle.
    {
      // data/geom/mesh/33/singularities/spindle_0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.67 0.01 0\n"\
         "0.33 0 0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      // data/geom/mesh/33/singularities/spindle_1e-100.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.67 1e-100 0\n"\
         "0.33 0 1e-100\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/spindle_0.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.67 0 0\n"\
         "0.33 0 0\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/spindle_-0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.67 0.01 0\n"\
         "0.33 0 -0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    // Spear.
    {
      // data/geom/mesh/33/singularities/spear_0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.01 0\n"\
         "0.5 0 0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      // data/geom/mesh/33/singularities/spear_1e-100.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 1e-100 0\n"\
         "0.5 0 1e-100\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/spear_0.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0 0\n"\
         "0.5 0 0\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/spear_-0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1 0 0\n"\
         "0.5 0.01 0\n"\
         "0.5 0 -0.01\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    // Spire.
    {
      // data/geom/mesh/33/singularities/spire_0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "0.01 0 0\n"\
         "0 0.01 0\n"\
         "0 0 0.81649658092772603273242802490196\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.empty());
    }
    {
      // data/geom/mesh/33/singularities/spire_1e-100.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "1e-100 0 0\n"\
         "0 1e-100 0\n"\
         "0 0 0.81649658092772603273242802490196\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/spire_0.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "0 0 0\n"\
         "0 0 0\n"\
         "0 0 0.81649658092772603273242802490196\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
    {
      // data/geom/mesh/33/singularities/spire_-0.01.txt
      std::istringstream
      in("3 3\n"\
         "4\n"\
         "0 0 0\n"\
         "0.01 0 0\n"\
         "0 0.01 0\n"\
         "0 0 -0.81649658092772603273242802490196\n"\
         "1\n"\
         "0 1 2 3\n");
      geom::readAscii(in, &mesh);
      std::vector<std::size_t> indices;
      geom::identifyLowQualityWithCondNum(mesh, 0.001,
                                          std::back_inserter(indices));
      assert(indices.size() == 1);
    }
  }

  return 0;
}
