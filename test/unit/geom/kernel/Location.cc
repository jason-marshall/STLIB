// -*- C++ -*-

#include "stlib/geom/kernel/Location.h"

#include <cassert>


int
main()
{
  using stlib::geom::location;
  {
    std::array<float, 1> const x{{float(2)}};
    assert(location(x) == x);
  }
  {
    std::array<std::array<float, 1>, 1> const x{{{{float(2)}}}};
    assert(location(x) == (std::array<float, 1>{{float(2)}}));
  }
  {
    stlib::geom::Ball<float, 1> const x{{{float(2)}}, float(1)};
    assert(location(x) == (std::array<float, 1>{{float(2)}}));
  }
  {
    stlib::geom::BBox<float, 1> const x{{{float(2)}}, {{float(4)}}};
    assert(location(x) == (std::array<float, 1>{{float(3)}}));
  }
  {
    std::pair<std::array<float, 1>, int> const x{{{float(2)}}, 3};
    assert(location(x) == x.first);
  }

  using stlib::geom::locationForEach;
  {
    std::vector<std::array<float, 1> > const x;
    auto const y = locationForEach(x.begin(), x.end());
    assert(y == x);
  }
  {
    std::vector<std::array<float, 1> > const x{{{float(2)}}};
    auto const y = locationForEach(x.begin(), x.end());
    assert(y == x);
  }

  return 0;
}
