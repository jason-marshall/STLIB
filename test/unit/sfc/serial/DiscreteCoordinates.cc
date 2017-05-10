// -*- C++ -*-

#include "stlib/sfc/DiscreteCoordinates.h"
#include "stlib/numerical/constants/Logarithm.h"

#include <limits>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace stlib;

// Check the maximum levels of refinement.
template<std::size_t _Dimension, typename _Code>
void
check()
{
  const std::size_t MaxLevels =
    sfc::BlockMaxLevels<_Dimension, _Code>::Result;
  const std::size_t Digits = std::numeric_limits<_Code>::digits;
  static_assert((1 + _Dimension * MaxLevels +
                 numerical::Logarithm < std::size_t, 2, MaxLevels + 1 >::
                 Result <= Digits), "Too many digits.");
  static_assert((1 + _Dimension * (MaxLevels + 1) +
                 numerical::Logarithm < std::size_t, 2, MaxLevels + 2 >::
                 Result > Digits), "Level too low.");
}

TEST_CASE("Check the maximum levels of refinement for each dimension and unsigned integer type.",
          "[BlockMaxLevels]")
{
  check<1, std::uint8_t>();
  check<1, std::uint16_t>();
  check<1, std::uint32_t>();
  check<1, std::uint64_t>();
  check<2, std::uint8_t>();
  check<2, std::uint16_t>();
  check<2, std::uint32_t>();
  check<2, std::uint64_t>();
  check<3, std::uint8_t>();
  check<3, std::uint16_t>();
  check<3, std::uint32_t>();
  check<3, std::uint64_t>();
}


TEST_CASE("Constructor and equality.", "[DiscreteCoordinates]")
{
  typedef sfc::Traits<1> Traits;
  typedef sfc::DiscreteCoordinates<Traits> DiscreteCoordinates;
  // Default constructor.
  {
    DiscreteCoordinates x;
  }
  // Lower corner constructor.
  typedef DiscreteCoordinates::Point Point;
  DiscreteCoordinates const x(Point{{0}}, Point{{1}}, 7);
  {
    DiscreteCoordinates const y(Point{{0}}, Point{{1}}, 7);
    REQUIRE(x == y);
  }
  {
    DiscreteCoordinates const y(Point{{1}}, Point{{1}}, 7);
    REQUIRE(!(x == y));
  }
  {
    DiscreteCoordinates const y(Point{{0}}, Point{{2}}, 7);
    REQUIRE(!(x == y));
  }
  {
    DiscreteCoordinates const y(Point{{0}}, Point{{1}}, 8);
    REQUIRE(!(x == y));
  }
}
