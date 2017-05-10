// -*- C++ -*-

#include "stlib/particle/traits.h"
#include "stlib/ads/functor/Identity.h"

#include <array>

using namespace stlib;

// A particle is just a point.
typedef std::array<Float, Dimension> Point;

struct SetPosition :
    public std::binary_function<Point*, Point, void> {
  typedef std::binary_function<Point*, Point, void> Base;
  Base::result_type
  operator()(Base::first_argument_type particle,
             const Base::second_argument_type& point) const
  {
    *particle = point;
  }
};

typedef particle::PeriodicTraits<Point, ads::Identity<Point>, SetPosition,
        Dimension, Float> Traits;
