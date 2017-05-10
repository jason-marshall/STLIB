// -*- C++ -*-

// Note this file is intentionally misspelled. On Windows, running any 
// program with the word "patch" in its name will bring up a dialog asking,
// "Do you want to allow the following program from an 
// unknown publisher to make changes to this computer?"

#include "stlib/levelSet/PatchActive.h"

#include <iostream>


using namespace stlib;

int
main()
{
  typedef levelSet::PatchActive Patch;
  typedef Patch::Point Point;
  typedef Patch::Ball Ball;

  Patch patch(1);

  {
    patch.initialize(Point{{0.f, 0.f, 0.f}}, true);
    assert(patch.numActivePoints() == Patch::NumPoints);
    patch.initialize(Point{{0.f, 0.f, 0.f}}, false);
    assert(patch.numActivePoints() == 0);
  }

  {
    patch.initialize(Point{{0.f, 0.f, 0.f}});
    assert(patch.numActivePoints() == Patch::NumPoints);
    Ball ball = {{{3.5, 3.5, 3.5}}, 7};
    patch.clip(ball);
    assert(patch.numActivePoints() == 0);
  }
  {
    patch.initialize(Point{{1.f, 2.f, 3.f}});
    assert(patch.numActivePoints() == Patch::NumPoints);
    Ball ball = {{{4.5, 5.5, 6.5}}, 7};
    patch.clip(ball);
    assert(patch.numActivePoints() == 0);
  }
  {
    patch.initialize(Point{{0.f, 0.f, 0.f}});
    assert(patch.numActivePoints() == Patch::NumPoints);
    Ball ball = {{{0, 0, 0}}, 0.5};
    patch.clip(ball);
    assert(patch.numActivePoints() == Patch::NumPoints - 1);
  }

  return 0;
}
