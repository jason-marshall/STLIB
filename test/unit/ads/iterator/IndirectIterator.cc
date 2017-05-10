// -*- C++ -*-

#include "stlib/ads/iterator/IndirectIterator.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include <cassert>

using namespace stlib;

int
main()
{
  {
    // IndirectIterator.
    int a = 7;
    int* ap = &a;
    int** app = &ap;

    {
      ads::IndirectIterator<int**> i(app);
      assert(*i == a);
      assert(i[0] == a);
    }
    {
      ads::IndirectIterator<int**> i;
      i = app;
      assert(*i == a);
      assert(i[0] == a);
    }
  }

  {
    // IndirectIterator2.
    int a = 7;
    int* ap = &a;
    int** app = &ap;
    int** * appp = &app;

    {
      ads::IndirectIterator2<int***> i(appp);
      assert(*i == a);
      assert(i[0] == a);
    }
    {
      ads::IndirectIterator2<int***> i;
      i = appp;
      assert(*i == a);
      assert(i[0] == a);
    }
  }

  {
    // IndirectIterator.
    std::vector<int> x;
    x.push_back(4);
    x.push_back(3);
    x.push_back(2);
    x.push_back(1);
    std::vector<int*> xp;
    for (std::vector<int>::iterator i = x.begin(); i != x.end(); ++i) {
      xp.push_back(&*i);
    }
    // This will sort the x vector.
    std::sort(ads::constructIndirectIterator(xp.begin()),
              ads::constructIndirectIterator(xp.end()));
    // Check that it is sorted.
    assert(std::is_sorted(x.begin(), x.end()));
    assert(std::is_sorted(ads::constructIndirectIterator(xp.begin()),
                          ads::constructIndirectIterator(xp.end())));
  }

  {
    // IndirectIterator2.
    std::vector<int> x;
    x.push_back(4);
    x.push_back(3);
    x.push_back(2);
    x.push_back(1);
    std::vector<int*> xp;
    for (std::vector<int>::iterator i = x.begin(); i != x.end(); ++i) {
      xp.push_back(&*i);
    }
    std::vector<int**> xpp;
    for (std::vector<int*>::iterator i = xp.begin(); i != xp.end(); ++i) {
      xpp.push_back(&*i);
    }
    // This will sort the x vector.
    std::sort(ads::constructIndirectIterator2(xpp.begin()),
              ads::constructIndirectIterator2(xpp.end()));
    // Check that it is sorted.
    assert(std::is_sorted(x.begin(), x.end()));
    assert(std::is_sorted(ads::constructIndirectIterator(xp.begin()),
                          ads::constructIndirectIterator(xp.end())));
    assert(std::is_sorted(ads::constructIndirectIterator2(xpp.begin()),
                          ads::constructIndirectIterator2(xpp.end())));
  }

  return 0;
}
