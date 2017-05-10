// -*- C++ -*-

#include "stlib/sfc/Codes.h"
#include "stlib/sfc/BlockCode.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"


using namespace stlib;

TEST_CASE("1-D, 2 levels", "[Codes]") {
  typedef sfc::Traits<1> Traits;
  typedef sfc::BlockCode<Traits> BlockCode;
  typedef BlockCode::Code Code;
  typedef BlockCode::Point Point;
  Code const GuardCode = Traits::GuardCode;
  BlockCode blockCode(Point{{0}}, Point{{1}}, 2);

  checkValidityOfObjectCodes(blockCode, std::vector<Code>{});
  checkValidityOfObjectCodes(blockCode, std::vector<Code>{0});
  checkValidityOfObjectCodes(blockCode, std::vector<Code>{0, 0});
  checkValidityOfObjectCodes(blockCode, std::vector<Code>{14, 14});
  checkValidityOfObjectCodes(blockCode,
                             std::vector<Code>{2, 6, 10, 14});

  checkValidityOfCellCodes
    (blockCode, std::vector<Code>{GuardCode});
  checkValidityOfCellCodes
    (blockCode, std::vector<Code>{0, GuardCode});
  checkValidityOfCellCodes
    (blockCode, std::vector<Code>{2, 6, 10, 14, GuardCode});
}


