// -*- C++ -*-

#include "stlib/ext/pair.h"

#include <sstream>

#include <cassert>

USING_STLIB_EXT_PAIR_IO_OPERATORS;

int
main()
{
  {
    std::pair<int, double> a(-1, 3);
    std::ostringstream out;
    out << a;
    std::pair<int, double> b;
    std::istringstream in(out.str());
    in >> b;
    assert(a == b);
  }

  return 0;
}
