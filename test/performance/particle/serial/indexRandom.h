// -*- C++ -*-

#include "stlib/particle/order.h"
#include "stlib/particle/traits.h"

#include "stlib/ads/functor/Identity.h"
#include "stlib/ads/timer/Timer.h"

#include <algorithm>
#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

template<typename _Traits>
class MortonOrder :
  public particle::MortonOrder<_Traits>
{
public:
  typedef typename _Traits::Float Float;
  typedef particle::MortonOrder<_Traits> Base;

  MortonOrder(const geom::BBox<Float, _Traits::Dimension>& domain,
              const Float interactionDistance, const Float padding) :
    Base(domain, interactionDistance, padding)
  {
  }

  using Base::morton;
  using Base::_cellCodes;
  using Base::_indexDirect;
  using Base::_indexForward;
  using Base::_indexBinary;
  using Base::_lookupTable;
};


int
main()
{
  typedef particle::IntegerTypes::Code Code;
  // A particle is just a point.
  typedef std::array<Float, Dimension> Point;
  typedef particle::PlainTraits<Point, ads::Identity<Point>, Dimension, Float>
  Traits;
  typedef MortonOrder<Traits> MortonOrder;

  ads::Timer timer;
  double elapsedTime;
  std::size_t result = 0;

  const geom::BBox<Float, Dimension> Domain = {ext::filled_array<Point>(0),
                                               ext::filled_array<Point>(1)
                                              };

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout << "Times are in nanoseconds."
            << "<table border=\"1\">\n"
            << "<tr>"
            << "<th>Levels</th><th>Nonempty</th><th>Shift</th>"
            << "<th>Direct</th><th>Forward</th><th>Binary</th>"
            << "<th>index()</th>\n";

  // Test trees with varying levels of refinement.
  const std::size_t MinLevel = 4;
  const std::size_t MaxLevel = 8;
  const std::size_t TestCells = 1 << (Dimension * MinLevel);
  Float cellLength = 1. / (1 << MinLevel);
  for (std::size_t cellLevels = MinLevel; cellLevels <= MaxLevel;
       ++cellLevels, cellLength *= 0.5) {
    // Make the orthtree.
    MortonOrder morton(Domain, cellLength, 0);
    // Test varying numbers of nonempty cells.
    for (std::size_t numPart = 1 << (Dimension * MinLevel);
         numPart <= std::size_t(1) << (Dimension * cellLevels);
         numPart *= 2) {

      // The particles are at the centers of cells.
      std::vector<Point> particles;
      // Fill the cells in order to form a continuous block.
      for (Code i = 0; i != numPart - 1; ++i) {
        particles.push_back((ext::convert_array<Float>
                             (morton.morton.coordinates(i)) + Float(0.5)) *
                            cellLength);
      }
      // Put a particle in the last cell.
      particles.push_back(ext::filled_array<Point>(1));
      // Set the particles.
      morton.setParticles(particles.begin(), particles.end());

      // The codes in random order.
      std::vector<Code> codes(morton.cellsSize());
      for (std::size_t i = 0; i != codes.size(); ++i) {
        codes[i] = morton._cellCodes[i];
      }
      std::random_shuffle(codes.begin(), codes.end());

      std::cout << "<tr><td>" << morton.morton.numLevels() << "</td><td>";
      std::cout.precision(8);
      std::cout << double(morton.cellsSize()) /
                (morton.morton.maxCode() + 1) << "</td><td>"
                << morton._lookupTable.shift() << "</td>";
      std::cout.precision(0);

      // Direct.
      timer.tic();
      for (std::size_t i = 0; i != TestCells; ++i) {
        result += morton._indexDirect(codes[i]);
      }
      elapsedTime = timer.toc();
      std::cout << "<td>" << elapsedTime * 1e9 / TestCells << "</td>";

      // Forward.
      timer.tic();
      for (std::size_t i = 0; i != TestCells; ++i) {
        result += morton._indexForward(codes[i]);
      }
      elapsedTime = timer.toc();
      std::cout << "<td>" << elapsedTime * 1e9 / TestCells << "</td>";

      // Binary.
      timer.tic();
      for (std::size_t i = 0; i != TestCells; ++i) {
        result += morton._indexBinary(codes[i]);
      }
      elapsedTime = timer.toc();
      std::cout << "<td>" << elapsedTime * 1e9 / TestCells << "</td>";

      // index().
      timer.tic();
      for (std::size_t i = 0; i != TestCells; ++i) {
        result += morton.index(morton._cellCodes[i]);
      }
      elapsedTime = timer.toc();
      std::cout << "<td>" << elapsedTime * 1e9 / TestCells << "</td>\n";
    }
  }
  std::cout << "</table>\n";

  std::cout << "\nMeaningless result = " << result << "\n";

  return 0;
}
