// -*- C++ -*-

#include "stlib/amr/ListOfNodesByLevel.h"

#include "stlib/ads/functor/Identity.h"

using namespace stlib;

int
main()
{
  {
    const int Dimension = 1;
    const int MaximumLevel = 10;
#define __test_amr_ListOfNodesByLevel_ipp__
#include "ListOfNodesByLevel.ipp"
#undef __test_amr_ListOfNodesByLevel_ipp__
  }
  {
    const int Dimension = 2;
    const int MaximumLevel = 8;
#define __test_amr_ListOfNodesByLevel_ipp__
#include "ListOfNodesByLevel.ipp"
#undef __test_amr_ListOfNodesByLevel_ipp__
  }
  {
    const int Dimension = 3;
    const int MaximumLevel = 6;
#define __test_amr_ListOfNodesByLevel_ipp__
#include "ListOfNodesByLevel.ipp"
#undef __test_amr_ListOfNodesByLevel_ipp__
  }
  {
    const int Dimension = 4;
    const int MaximumLevel = 4;
#define __test_amr_ListOfNodesByLevel_ipp__
#include "ListOfNodesByLevel.ipp"
#undef __test_amr_ListOfNodesByLevel_ipp__
  }

  return 0;
}
