// -*- C++ -*-

// Note this file is intentionally misspelled. On Windows, running any 
// program with the word "patch" in its name will bring up a dialog asking,
// "Do you want to allow the following program from an 
// unknown publisher to make changes to this computer?"

#include "stlib/levelSet/PatchColMajDistComp.h"

#include <iostream>

using namespace stlib;

class PatchColMajDistCompTest :
  public levelSet::PatchColMajDistComp
{
private:
  typedef PatchColMajDistComp Base;

public:

  using Base::_lowerCorner;
  using Base::_dx;
  using Base::_dy;
  using Base::_dz;

  PatchColMajDistCompTest(const float spacing) :
    Base(spacing)
  {
  }

};


int
main()
{
  typedef PatchColMajDistCompTest Patch;
  typedef Patch::Point Point;

  Patch patch(1);
  const Point c = {{1, 2, 3}};
  patch._lowerCorner = c;
  const Point p = {{2, 3, 5}};
  patch.computeDistanceComponents(p);

#ifdef STLIB_NO_SIMD_INTRINSICS

  for (std::size_t i = 0; i != Patch::Extent; ++i) {
    assert(patch._dx[i] == (c[0] + i - p[0]) * (c[0] + i - p[0]));
    assert(patch._dy[i] == (c[1] + i - p[1]) * (c[1] + i - p[1]));
    assert(patch._dz[i] == (c[2] + i - p[2]) * (c[2] + i - p[2]));
  }

#else

#ifdef __AVX2__
  // CONTINUE
#elif defined(__SSE__)
  // CONTINUE
#else
#error SIMD is not supported.
#endif

#endif

  return 0;
}
