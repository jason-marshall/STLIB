// -*- C++ -*-

#include "stlib/sfc/sfcOrder.h"
#include "stlib/performance/SimpleTimer.h"

#include <random>

//#include <cstdint>


using Float = float;
std::size_t constexpr D = 3;
using Point = std::array<Float, D>;


template<typename Code>
void
test(std::vector<Point> const& locations)
{
  stlib::performance::SimpleTimer timer;
  timer.start();
  auto const indices = stlib::sfc::sfcOrderSpecific<unsigned, Code>(locations);
  timer.stop();
  std::cout << "\nMeaningless result = " << indices[0] << '\n'
            << "Time to order with " << sizeof(Code) << "-byte codes = "
            << timer.elapsed() << '\n';
}


int
main()
{

  std::mt19937_64 engine;
  std::uniform_real_distribution<Float> dist(0, 1);

  // Make a vector of random points.
  std::vector<Point> locations(10000000);
  for (auto&& p: locations) {
    for (auto&& x: p) {
      x = dist(engine);
    }
  }

  std::cout << "Number of locations = " << locations.size() << '\n';
  test<std::uint8_t>(locations);
  test<std::uint16_t>(locations);
  test<std::uint32_t>(locations);
  test<std::uint64_t>(locations);

  return 0;
}
