// -*- C++ -*-

// Note this file is intentionally misspelled. On Windows, running any 
// program with the word "patch" in its name will bring up a dialog asking,
// "Do you want to allow the following program from an 
// unknown publisher to make changes to this computer?"

#include "stlib/levelSet/PatchColMajDistance.h"
#include "stlib/numerical/equality.h"

#include <iostream>

using namespace stlib;

int
main()
{
#ifdef __AVX2__
  using numerical::areEqual;

  typedef levelSet::PatchColMajDistance Patch;
  typedef Patch::Ball Ball;

  Patch patch(1);
  patch.initialize(ext::make_array<float>(1, 2, 3));
  const Ball ball = {{{2, 3, 5}}, 1};

  __m128 d = patch[0];
  float x = *reinterpret_cast<const float*>(&d);
  assert(x == std::numeric_limits<float>::infinity());
  d = patch[Patch::Size - 1];
  x = *(reinterpret_cast<const float*>(&d) + 3);
  assert(x == std::numeric_limits<float>::infinity());

  patch.unionEuclidean(ball);

  d = patch[0];
  x = *reinterpret_cast<const float*>(&d);
  assert(areEqual(x, std::sqrt(6.f) - ball.radius));
  // (2, 3, 5)
  // (8, 9, 10)
  d = patch[Patch::Size - 1];
  x = *(reinterpret_cast<const float*>(&d) + 3);
  assert(areEqual(x, std::sqrt(97.f) - ball.radius));

  patch.conditionalSetValueGe(0.f, -1.f);
  for (std::size_t i = 0; i != Patch::NumVectors; ++i) {
    const simd::Vector<float>::Type v = patch[i];
    const float* p = reinterpret_cast<const float*>(&v);
    for (std::size_t j = 0; j != simd::Vector<float>::Size; ++j) {
      assert(p[j] < 0);
    }
  }
#endif

  return 0;
}
