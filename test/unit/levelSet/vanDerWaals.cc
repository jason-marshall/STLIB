// -*- C++ -*-

#include "stlib/levelSet/vanDerWaals.h"

#include "stlib/numerical/equality.h"

using namespace stlib;

int
main()
{
  using numerical::areEqual;

  typedef float T;

  // vanDerWaals()
  {
    // 1-D.
    const std::size_t D = 1;
    const T TargetGridSpacing = 0.1;
    typedef geom::Ball<T, D> Ball;

    std::vector<Ball> balls;
    {
      Ball b = {{{0}}, 1};
      balls.push_back(b);
    }

    const std::pair<T, T> content =
      levelSet::vanDerWaals(balls, TargetGridSpacing);
    assert(std::abs(content.first - 2) <
           2 * TargetGridSpacing * TargetGridSpacing);
    assert(areEqual(content.second, T(1)));
  }

  return 0;
}
