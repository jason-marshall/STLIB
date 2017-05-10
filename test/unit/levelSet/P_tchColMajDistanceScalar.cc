// -*- C++ -*-

// Note this file is intentionally misspelled. On Windows, running any 
// program with the word "patch" in its name will bring up a dialog asking,
// "Do you want to allow the following program from an 
// unknown publisher to make changes to this computer?"

#define STLIB_NO_SIMD_INTRINSICS

#include "stlib/levelSet/PatchColMajDistance.h"
#include "stlib/numerical/equality.h"

#include <iostream>

using namespace stlib;

int
main()
{
  using numerical::areEqual;

  typedef levelSet::PatchColMajDistance Patch;
  typedef Patch::Point Point;
  typedef Patch::Ball Ball;

  Patch patch(1);
  patch.initialize(Point{{1, 2, 3}});
  const Ball ball = {{{2, 3, 5}}, 1};

  assert(patch[0] == std::numeric_limits<float>::infinity());
  assert(patch[Patch::NumPoints - 1] == std::numeric_limits<float>::infinity());

  patch.unionEuclidean(ball);

  assert(areEqual(patch[0], std::sqrt(6.f) - ball.radius));
  // (2, 3, 5)
  // (8, 9, 10)
  assert(areEqual(patch[Patch::NumPoints - 1], std::sqrt(97.f) - ball.radius));

  patch.conditionalSetValueGe(0.f, -1.f);
  for (std::size_t i = 0; i != Patch::NumPoints; ++i) {
    assert(patch[i] < 0);
  }

  return 0;
}
