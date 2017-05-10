// -*- C++ -*-

#include "stlib/particle/traits.h"
#include "stlib/ads/functor/Identity.h"

#include <array>

using namespace stlib;

// A particle is just a point.
typedef std::array<Float, Dimension> Point;
typedef particle::PlainTraits<Point, ads::Identity<Point>,
        Dimension, Float> Traits;
