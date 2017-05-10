// -*- C++ -*-

// Note this file is intentionally misspelled. On Windows, running any 
// program with the word "patch" in its name will bring up a dialog asking,
// "Do you want to allow the following program from an 
// unknown publisher to make changes to this computer?"

#include "stlib/levelSet/PatchCountNegative.h"

#include <iostream>


using namespace stlib;

int
main()
{
  typedef levelSet::PatchCountNegative Patch;
  typedef Patch::Point Point;
  typedef Patch::Ball Ball;

  Patch patch(1);

  {
    patch.initialize(Point{{0.f, 0.f, 0.f}});
    assert(patch.numNegative() == 0);
    Ball ball = {{{3.5, 3.5, 3.5}}, 7};
    patch.clip(ball);
    assert(patch.numNegative() == 512);
  }
  {
    patch.initialize(Point{{1.f, 2.f, 3.f}});
    assert(patch.numNegative() == 0);
    Ball ball = {{{4.5, 5.5, 6.5}}, 7};
    patch.clip(ball);
    assert(patch.numNegative() == 512);
  }
  {
    patch.initialize(Point{{0.f, 0.f, 0.f}});
    assert(patch.numNegative() == 0);
    Ball ball = {{{0, 0, 0}}, 0.5};
    patch.clip(ball);
    assert(patch.numNegative() == 1);
  }

  return 0;
}
