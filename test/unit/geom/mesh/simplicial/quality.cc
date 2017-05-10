// -*- C++ -*-

//
// Test the set operations for SimpMeshRed.
//

#include "stlib/geom/mesh/simplicial/quality.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace geom;

  // 3-D space, 3-simplex.
  typedef SimpMeshRed<3> SM;

  //
  // Data for an octahedron
  //
  const std::array<double, 3> vertices[] = {{{0, 0, 0}},
    {{1, 0, 0}},
    {{ -1, 0, 0}},
    {{0, 1, 0}},
    {{0, -1, 0}},
    {{0, 0, 1}},
    {{0, 0, -1}}
  };
  const std::size_t numVertices = sizeof(vertices) /
                                  sizeof(std::array<double, 3>);
  const std::array<std::size_t, 4> simplices[] = {{{0, 1, 3, 5}},
    {{0, 3, 2, 5}},
    {{0, 2, 4, 5}},
    {{0, 4, 1, 5}},
    {{0, 3, 1, 6}},
    {{0, 2, 3, 6}},
    {{0, 4, 2, 6}},
    {{0, 1, 4, 6}}
  };
  const std::size_t numSimplices = sizeof(simplices) /
                                   sizeof(std::array<std::size_t, 4>);

  // Build from an indexed simplex set.
  SM x;
  x.build(vertices, vertices + numVertices,
          simplices, simplices + numSimplices);

  // Tests.
  const double eps = std::numeric_limits<double>::epsilon() * 100;

  double content = geom::computeContent(x);
  std::cout << "Content = " << content << "\n";
  assert(std::abs(content - 4. / 3.) < eps);

  double minContent, maxContent, meanContent;
  geom::computeContentStatistics(x, &minContent, &maxContent, &meanContent);
  std::cout << "Content statistics:\n"
            << minContent << " "
            << maxContent << " "
            << meanContent << "\n";
  assert(std::abs(minContent - 1. / 6.) < eps);
  assert(std::abs(maxContent - 1. / 6.) < eps);
  assert(std::abs(meanContent - 1. / 6.) < eps);

  double minDeterminant, maxDeterminant, meanDeterminant;
  geom::computeDeterminantStatistics(x, &minDeterminant, &maxDeterminant,
                                     &meanDeterminant);
  std::cout << "Determinant statistics:\n"
            << minDeterminant << " "
            << maxDeterminant << " "
            << meanDeterminant << "\n";
  assert(std::abs(minDeterminant - std::sqrt(2.0)) < eps);
  assert(std::abs(maxDeterminant - std::sqrt(2.0)) < eps);
  assert(std::abs(meanDeterminant - std::sqrt(2.0)) < eps);

  double minModifiedMeanRatio, maxModifiedMeanRatio, meanModifiedMeanRatio;
  geom::computeModifiedMeanRatioStatistics(x, &minModifiedMeanRatio,
      &maxModifiedMeanRatio,
      &meanModifiedMeanRatio);
  std::cout << "Mod_Mean_Ratio statistics:\n"
            << minModifiedMeanRatio << " "
            << maxModifiedMeanRatio << " "
            << meanModifiedMeanRatio << "\n";

  double minModifiedConditionNumber, maxModifiedConditionNumber,
         meanModifiedConditionNumber;
  geom::computeModifiedConditionNumberStatistics
  (x, &minModifiedConditionNumber, &maxModifiedConditionNumber,
   &meanModifiedConditionNumber);
  std::cout << "Mod_Cond_Num statistics:\n"
            << minModifiedConditionNumber << " "
            << maxModifiedConditionNumber << " "
            << meanModifiedConditionNumber << "\n";

  geom::computeQualityStatistics
  (x, &minContent, &maxContent, &meanContent,
   &minDeterminant, &maxDeterminant, &meanDeterminant,
   &minModifiedMeanRatio, &maxModifiedMeanRatio, &meanModifiedMeanRatio,
   &minModifiedConditionNumber, &maxModifiedConditionNumber,
   &meanModifiedConditionNumber);
  std::cout << "Statistics:\n"
            << minContent << " "
            << maxContent << " "
            << meanContent << "\n"
            << minDeterminant << " "
            << maxDeterminant << " "
            << meanDeterminant << "\n"
            << minModifiedMeanRatio << " "
            << maxModifiedMeanRatio << " "
            << meanModifiedMeanRatio << "\n"
            << minModifiedConditionNumber << " "
            << maxModifiedConditionNumber << " "
            << meanModifiedConditionNumber << "\n";

  geom::printQualityStatistics(std::cout, x);

  return 0;
}
