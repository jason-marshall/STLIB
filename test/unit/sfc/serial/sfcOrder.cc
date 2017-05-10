// -*- C++ -*-

#include "stlib/sfc/sfcOrder.h"

#include <random>

int
main()
{
  using Float = float;
  std::size_t constexpr D = 3;
  using Point = std::array<Float, D>;

  std::mt19937_64 engine;
  std::uniform_real_distribution<Float> dist(0, 1);

  // Make a vector of random points.
  std::vector<Point> objects(100);
  for (auto&& p: objects) {
    for (auto&& x: p) {
      x = dist(engine);
    }
  }
  auto indices = stlib::sfc::sfcOrder(objects, [](Point const& x){return x;});
  assert(indices.size() == objects.size());
  
  // Check that the indices are a permutation of [0..N).
  std::sort(indices.begin(), indices.end());
  for (std::size_t i = 0; i != indices.size(); ++i) {
    assert(indices[i] == i);
  }

  return 0;
}
