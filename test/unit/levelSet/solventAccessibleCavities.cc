// -*- C++ -*-

#include "stlib/levelSet/solventAccessibleCavities.h"

#include "stlib/numerical/equality.h"

using namespace stlib;

int
main()
{
  using numerical::areEqual;

  typedef float T;

  // solventAccessibleCavities()
  {
    // 3-D.
    const std::size_t D = 3;
    const T ProbeRadius = 1.4;
    const T TargetGridSpacing = 0.1;
    typedef geom::Ball<T, D> Ball;

    std::vector<Ball> balls;
    {
      Ball b = {{{0, 0, 0}}, 1};
      balls.push_back(b);
    }
    {
      Ball b = {{{3, 0, 0}}, 1};
      balls.push_back(b);
    }

    std::vector<T> content, boundary;
    levelSet::solventAccessibleCavities(balls, ProbeRadius,
                                        TargetGridSpacing, &content,
                                        &boundary);
    assert(content.empty());
    assert(boundary.empty());

    std::vector<Ball> seeds;
    levelSet::solventAccessibleCavitySeeds(balls, ProbeRadius,
                                           TargetGridSpacing, &seeds);
    assert(seeds.empty());
  }

  return 0;
}
