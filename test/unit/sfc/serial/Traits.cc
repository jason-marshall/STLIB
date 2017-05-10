// -*- C++ -*-

#include "stlib/sfc/Traits.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace stlib;

TEST_CASE("Traits.", "[Traits]")
{
  typedef sfc::Traits<> Traits;
  typedef Traits::Code Code;

  REQUIRE(Code(Traits::GuardCode) ==
          std::numeric_limits<Code>::max());
}
