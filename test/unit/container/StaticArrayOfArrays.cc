// -*- C++ -*-

#include "stlib/container/StaticArrayOfArrays.h"

#include <iostream>
#include <sstream>

#include <cassert>

using namespace stlib;

int
main()
{
  {
    // Default constructor.
    container::StaticArrayOfArrays<double> x;
    assert(x.getNumberOfArrays() == 0);
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.max_size() > 0);
    assert(x.getMemoryUsage() > 0);
    assert(x.begin() == x.end());
  }
  {
    // Construct from a container of containers.
    std::vector<std::vector<double> > cc(7);
    cc[1].push_back(1);
    cc[1].push_back(1);
    cc[3].push_back(2);
    cc[3].push_back(3);
    cc[3].push_back(5);
    cc[5].push_back(8);
    cc[5].push_back(13);
    cc[5].push_back(21);
    cc[5].push_back(34);
    cc[5].push_back(55);
    const std::size_t numElements = 10;
    container::StaticArrayOfArrays<double> x(cc);

    assert(x.getNumberOfArrays() == cc.size());
    assert(x.size() == numElements);
    assert(! x.empty());
    assert(x.max_size() > 0);
    assert(x.getMemoryUsage() > 0);
    assert(x.begin() + x.size() == x.end());
    assert(x.rbegin() + x.size() == x.rend());

    for (std::size_t n = 0; n != cc.size(); ++n) {
      assert(x.size(n) == cc[n].size());
      assert(x.empty(n) == cc[n].empty());
      assert(x.begin(n) + x.size(n) == x.end(n));
      assert(x.rbegin(n) + x.size(n) == x.rend(n));
    }

    std::vector<double> values(x.begin(), x.end());
    assert(values.size() == x.size());
    assert(std::equal(x.begin(), x.end(), values.begin()));
    assert(std::equal(x.rbegin(), x.rend(), values.rbegin()));

    assert(x(1, 0) == cc[1][0]);
    assert(x(1, 1) == cc[1][1]);

    assert(x(3, 0) == cc[3][0]);
    assert(x(3, 1) == cc[3][1]);
    assert(x(3, 2) == cc[3][2]);

    assert(x(5, 0) == cc[5][0]);
    assert(x(5, 1) == cc[5][1]);
    assert(x(5, 2) == cc[5][2]);
    assert(x(5, 3) == cc[5][3]);
    assert(x(5, 4) == cc[5][4]);
  }
  {
    // Construct from sizes and values.
    const std::size_t numArrays = 7;
    const std::size_t numElements = 10;
    const std::size_t sizes[numArrays] = { 0, 2, 0, 3, 0, 5, 0 };
    const double values[numElements] = { 1, 1,
                                         2, 3, 5,
                                         8, 13, 21, 34, 55
                                       };
    container::StaticArrayOfArrays<double> x(sizes, sizes + numArrays,
        values, values + numElements);

    assert(x.getNumberOfArrays() == numArrays);
    assert(x.size() == numElements);
    assert(! x.empty());
    assert(x.max_size() > 0);
    assert(x.getMemoryUsage() > 0);
    assert(x.begin() + x.size() == x.end());

    for (std::size_t n = 0; n != numArrays; ++n) {
      assert(x.size(n) == sizes[n]);
      assert(x.empty(n) == ! sizes[n]);
      assert(x.begin(n) + x.size(n) == x.end(n));
    }

    assert(x(1, 0) == values[0]);
    assert(x(1, 1) == values[1]);

    assert(x(3, 0) == values[2]);
    assert(x(3, 1) == values[3]);
    assert(x(3, 2) == values[4]);

    assert(x(5, 0) == values[5]);
    assert(x(5, 1) == values[6]);
    assert(x(5, 2) == values[7]);
    assert(x(5, 3) == values[8]);
    assert(x(5, 4) == values[9]);

    // Copy constructor.
    {
      container::StaticArrayOfArrays<double> y(x);
      assert(x == y);
    }

    // Swap.
    {
      container::StaticArrayOfArrays<double> y(x);
      container::StaticArrayOfArrays<double> z;
      z.swap(y);
      assert(x == z);
    }

    // Assignment operator
    {
      container::StaticArrayOfArrays<double> y;
      y = x;
      assert(x == y);
    }

    // Build
    {
      container::StaticArrayOfArrays<double> y;
      y.rebuild(sizes, sizes + numArrays, values, values + numElements);
      assert(x == y);
    }

    // Build
    {
      container::StaticArrayOfArrays<double> y;
      y.rebuild(sizes, sizes + numArrays);
      std::copy(values, values + numElements, y.begin());
      assert(x == y);
    }

    //
    // Swap.
    //
    {
      container::StaticArrayOfArrays<double> a(x);
      container::StaticArrayOfArrays<double> b;
      a.swap(b);
      assert(x == b);
    }

    // Ascii I/O.
    {
      std::stringstream s;
      s << x;
      container::StaticArrayOfArrays<double> a;
      s >> a;
      assert(x == a);
    }
  }

  return 0;
}
