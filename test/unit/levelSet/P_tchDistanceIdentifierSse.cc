// -*- C++ -*-

// Note this file is intentionally misspelled. On Windows, running any 
// program with the word "patch" in its name will bring up a dialog asking,
// "Do you want to allow the following program from an 
// unknown publisher to make changes to this computer?"

#include "stlib/levelSet/PatchDistanceIdentifier.h"
#include "stlib/numerical/equality.h"

#include <iostream>

using namespace stlib;

int
main()
{
#ifdef __SSE__
  using numerical::areEqual;

  typedef levelSet::PatchDistanceIdentifier Patch;
  typedef Patch::Point Point;
  typedef Patch::Ball Ball;

  Patch patch(1);
  patch.initialize(Point{{1, 2, 3}});
  const Ball ball = {{{2, 3, 5}}, 1};

  __m128 d = patch[0];
  float x = *reinterpret_cast<const float*>(&d);
  assert(x == std::numeric_limits<float>::infinity());
  d = patch[Patch::NumVectors - 1];
  x = *(reinterpret_cast<const float*>(&d) + 3);
  assert(x == std::numeric_limits<float>::infinity());

  __m128i idv = patch.identifiers[0];
  unsigned id = *reinterpret_cast<const unsigned*>(&idv);
  assert(id == std::numeric_limits<unsigned>::max());
  idv = patch.identifiers[Patch::NumVectors - 1];
  id = *(reinterpret_cast<const unsigned*>(&idv) + 3);
  assert(id == std::numeric_limits<unsigned>::max());

  patch.unionEuclidean(ball, 0);

  d = patch[0];
  x = *reinterpret_cast<const float*>(&d);
  assert(areEqual(x, std::sqrt(6.f) - ball.radius));
  // (2, 3, 5)
  // (8, 9, 10)
  d = patch[Patch::NumVectors - 1];
  x = *(reinterpret_cast<const float*>(&d) + 3);
  assert(areEqual(x, std::sqrt(97.f) - ball.radius));

  idv = patch.identifiers[0];
  id = *reinterpret_cast<const unsigned*>(&idv);
  assert(id == 0);
  idv = patch.identifiers[Patch::NumVectors - 1];
  id = *(reinterpret_cast<const unsigned*>(&idv) + 3);
  assert(id == 0);

  patch.conditionalSetValueGe(0.f, -std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i != Patch::NumVectors; ++i) {
    const __m128 v = patch[i];
    const float* p = reinterpret_cast<const float*>(&v);
    idv = patch.identifiers[i];
    const unsigned* q = reinterpret_cast<const unsigned*>(&idv);
    for (std::size_t j = 0; j != 4; ++j) {
      assert(p[j] < 0);
      if (p[j] == -std::numeric_limits<float>::infinity()) {
        assert(q[j] == std::numeric_limits<unsigned>::max());
      }
      else {
        assert(q[j] == 0);
      }
    }
  }
#endif

  return 0;
}
