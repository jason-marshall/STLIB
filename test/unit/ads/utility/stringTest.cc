// -*- C++ -*-

#include "stlib/ads/utility/string.h"

#include <iostream>
#include <string>
#include <vector>

#include <cassert>

using namespace stlib;

int
main()
{
  std::vector<std::string> strings;

  ads::split("", " ", std::back_inserter(strings));
  assert(strings.size() == 0);
  strings.clear();

  ads::split("   ", " ", std::back_inserter(strings));
  assert(strings.size() == 0);
  strings.clear();

  ads::split("a", " ", std::back_inserter(strings));
  assert(strings.size() == 1);
  assert(strings[0] == "a");
  strings.clear();

  ads::split("  a   ", " ", std::back_inserter(strings));
  assert(strings.size() == 1);
  assert(strings[0] == "a");
  strings.clear();

  ads::split("  alpha   ", " ", std::back_inserter(strings));
  assert(strings.size() == 1);
  assert(strings[0] == "alpha");
  strings.clear();

  ads::split("a b c", " ", std::back_inserter(strings));
  assert(strings.size() == 3);
  assert(strings[0] == "a");
  assert(strings[1] == "b");
  assert(strings[2] == "c");
  strings.clear();

  ads::split(" a  b   c  ", " ", std::back_inserter(strings));
  assert(strings.size() == 3);
  assert(strings[0] == "a");
  assert(strings[1] == "b");
  assert(strings[2] == "c");
  strings.clear();

  ads::split("a,bb,ccc", ",", std::back_inserter(strings));
  assert(strings.size() == 3);
  assert(strings[0] == "a");
  assert(strings[1] == "bb");
  assert(strings[2] == "ccc");
  strings.clear();

  return 0;
}
