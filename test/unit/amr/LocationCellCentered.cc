// -*- C++ -*-

#include "stlib/amr/LocationCellCentered.h"
#include "stlib/amr/Traits.h"

#include <iostream>

using namespace stlib;

int
main()
{
  {
    const std::size_t Dimension = 1;
    const std::size_t MaximumLevel = 10;
#define __test_amr_LocationCellCentered_ipp__
#include "LocationCellCentered.ipp"
#undef __test_amr_LocationCellCentered_ipp__

    LocationCellCentered x(Point{{0.}}, Point{{1.}}, IndexList{{1}});
    assert(x(IndexList{{0}}) == Point{{0.5}});
    assert(x(IndexList{{-1}}) == Point{{-0.5}});
    assert(x(IndexList{{1}}) == Point{{1.5}});

    x = LocationCellCentered(Point{{-1.}}, Point{{2.}}, IndexList{{1}});
    assert(x(IndexList{{0}}) == Point{{0.}});
    assert(x(IndexList{{-1}}) == Point{{-2.}});
    assert(x(IndexList{{1}}) == Point{{2.}});
  }
  {
    const std::size_t Dimension = 2;
    const std::size_t MaximumLevel = 8;
#define __test_amr_LocationCellCentered_ipp__
#include "LocationCellCentered.ipp"
#undef __test_amr_LocationCellCentered_ipp__

    LocationCellCentered x(Point{{0., 1.}}, Point{{1., 1.}}, IndexList{{1, 1}});
    assert(x(IndexList{{0, 0}}) == (Point{{0.5, 1.5}}));
    assert(x(IndexList{{-1, 0}}) == (Point{{-0.5, 1.5}}));
    assert(x(IndexList{{1, 0}}) == (Point{{1.5, 1.5}}));
    assert(x(IndexList{{0, -1}}) == (Point{{0.5, 0.5}}));
    assert(x(IndexList{{0, 1}}) == (Point{{0.5, 2.5}}));
  }
  {
    const std::size_t Dimension = 3;
    const std::size_t MaximumLevel = 6;
#define __test_amr_LocationCellCentered_ipp__
#include "LocationCellCentered.ipp"
#undef __test_amr_LocationCellCentered_ipp__

    LocationCellCentered x(Point{{0., 1., 2.}}, Point{{1., 2., 4.}},
                           IndexList{{4, 2, 1}});
    assert(x(IndexList{{0, 0, 0}}) == (Point{{0.125, 1.5, 4.}}));
  }
  {
    const std::size_t Dimension = 4;
    const std::size_t MaximumLevel = 4;
#define __test_amr_LocationCellCentered_ipp__
#include "LocationCellCentered.ipp"
#undef __test_amr_LocationCellCentered_ipp__

    LocationCellCentered x(Point{{0., 1., 2., 3.}}, Point{{1., 2., 4., 8.}},
                           IndexList{{4, 2, 1, 2}});
    assert(x(IndexList{{0, 0, 0, 0}}) == (Point{{0.125, 1.5, 4., 5.}}));
  }

  return 0;
}
