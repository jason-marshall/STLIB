// -*- C++ -*-

#include "stlib/performance/AutoTimer.h"

using namespace stlib::performance;

int
main()
{
  AutoTimer _("Print line");
  std::cout << std::endl;

  return 0;
}
