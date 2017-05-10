// -*- C++ -*-

#include "stlib/ads/functor/select.h"

#include <vector>

#include <cassert>

using namespace stlib;

int
main()
{
  {
    typedef std::pair<char, int> Pair;
    Pair x;
    x.first = 'a';
    x.second = 2;
    {
      ads::Select1st<Pair> s;
      assert(s(x) == 'a');
    }
    {
      ads::Select2nd<Pair> s;
      assert(s(x) == 2);
    }
  }

  {
    typedef std::vector<int> Sequence;
    Sequence x;
    x.push_back(2);
    x.push_back(3);
    x.push_back(5);
    {
      ads::SelectElement<Sequence, 0> s;
      assert(s(x) == x[0]);
    }
    {
      ads::SelectElement<Sequence, 1> s;
      assert(s(x) == x[1]);
    }
    {
      ads::SelectElement<Sequence, 2> s;
      assert(s(x) == x[2]);
    }
  }

  return 0;
}
