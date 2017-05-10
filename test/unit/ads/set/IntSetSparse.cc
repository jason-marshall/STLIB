// -*- C++ -*-

//
// Tests for IntSetSparse.
//

#include "stlib/ads/set/IntSetSparse.h"

#include <iostream>

using namespace stlib;

int
main()
{
  using namespace ads;

  typedef IntSetSparse<> ISS;
  typedef ISS::iterator iterator;

  // Default constructor.
  {
    ISS x;
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.begin() == x.end());
    assert(x.subset(x));
    assert(x.is_valid());
  }

  // Upper bound constructor.
  {
    const int ub = 10;
    ISS x(ub);
    assert(x.upper_bound() == ub);
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.begin() == x.end());
    assert(x.subset(x));
    assert(x.is_valid());
  }

  // Range constructor.
  {
    const int ub = 10;
    const int data[] = { 2, 3, 5, 7 };
    const int size = sizeof(data) / sizeof(int);
    ISS x(data, data + size, ub);
    assert(x.upper_bound() == ub);
    assert(x.size() == size);
    assert(! x.empty());
    assert(x.begin() != x.end());
    assert(x.subset(x));
    assert(x.is_valid());

    assert(!x.is_in(0));
    assert(!x.is_in(1));
    assert(x.is_in(2));
    assert(x.is_in(3));
    assert(!x.is_in(4));
    assert(x.is_in(5));
    assert(!x.is_in(6));
    assert(x.is_in(7));
    assert(!x.is_in(8));
    assert(!x.is_in(9));

    x.set_upper_bound(20);
    assert(x.upper_bound() == 20);

    x.clear();
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.begin() == x.end());
    assert(x.subset(x));
    assert(x.is_valid());
  }

  // Insert elements.
  {
    const int ub = 10;
    ISS x(ub);
    x.insert(2);
    x.insert(3);
    x.insert(5);
    x.insert(7);
    assert(x.upper_bound() == ub);
    assert(x.size() == 4);
    assert(! x.empty());
    assert(x.begin() != x.end());
    assert(x.subset(x));
    assert(x.is_valid());
  }

  // Insert a range of elements.
  {
    const int ub = 10;
    const int data[] = { 2, 3, 5, 7 };
    const int size = sizeof(data) / sizeof(int);
    ISS x(ub);
    x.insert(data, data + size);
    assert(x.upper_bound() == ub);
    assert(x.size() == size);
    assert(! x.empty());
    assert(x.begin() != x.end());
    assert(x.subset(x));
    assert(x.is_valid());
  }

  // Erase by iterator.
  {
    const int ub = 10;
    const int data[] = { 2, 3, 5, 7 };
    const int size = sizeof(data) / sizeof(int);
    ISS x(data, data + size, ub);

    iterator i = x.begin();
    while (i != x.end()) {
      x.erase(i++);
    }

    assert(x.size() == 0);
    assert(x.empty());
    assert(x.begin() == x.end());
    assert(x.subset(x));
    assert(x.is_valid());
  }

  // Erase by value.
  {
    const int ub = 10;
    const int data[] = { 2, 3, 5, 7 };
    const int size = sizeof(data) / sizeof(int);
    ISS x(data, data + size, ub);

    for (int i = 0; i != size; ++i) {
      x.erase(data[i]);
    }

    assert(x.size() == 0);
    assert(x.empty());
    assert(x.begin() == x.end());
    assert(x.subset(x));
    assert(x.is_valid());
  }

  // Swap.
  {
    ISS x;
    const int ub = 10;
    const int data[] = { 2, 3, 5, 7 };
    const int size = sizeof(data) / sizeof(int);
    {
      ISS y(data, data + size, ub);
      x.swap(y);
    }
    assert(x.upper_bound() == ub);
    assert(x.size() == size);
    assert(! x.empty());
    assert(x.begin() != x.end());
    assert(x.subset(x));
    assert(x.is_valid());
  }

  // Set operations.
  {
    const int ub = 10;
    ISS x(ub), y(ub);

    // x is the set of primes.
    x.insert(2);
    x.insert(3);
    x.insert(5);
    x.insert(7);

    // y is the set of odd integers.
    y.insert(1);
    y.insert(3);
    y.insert(5);
    y.insert(7);
    y.insert(9);

    // Union.
    {
      const int data[] = { 1, 2, 3, 5, 7, 9 };
      const int size = sizeof(data) / sizeof(int);
      ISS a(data, data + size, ub), b(ub);
      set_union(x, y, b);
      assert(a == b);
    }

    // Intersection.
    {
      const int data[] = { 3, 5, 7 };
      const int size = sizeof(data) / sizeof(int);
      ISS a(data, data + size, ub), b(ub);
      set_intersection(x, y, b);
      assert(a == b);
    }

    // Difference.
    {
      const int data[] = { 2 };
      const int size = sizeof(data) / sizeof(int);
      ISS a(data, data + size, ub), b(ub);
      set_difference(x, y, b);
      assert(a == b);
    }

    // Complement.
    {
      const int data[] = { 0, 1, 4, 6, 8, 9 };
      const int size = sizeof(data) / sizeof(int);
      ISS a(data, data + size, ub), b(ub);
      set_complement(x, b);
      assert(a == b);
    }
  }

  return 0;
}
