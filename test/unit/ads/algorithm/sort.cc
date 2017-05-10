// -*- C++ -*-

#include "stlib/ads/algorithm/sort.h"

#include <cassert>

using namespace stlib;

int
main()
{
  {
    const int size = 4;
    int a[size] = {3, 2, 1, 0};
    double b[size] = {0, 1, 2, 3};
    ads::sortTogether(a, a + size, b, b + size);
    assert(a[0] == 0);
    assert(a[1] == 1);
    assert(a[2] == 2);
    assert(a[3] == 3);
    assert(b[0] == 3);
    assert(b[1] == 2);
    assert(b[2] == 1);
    assert(b[3] == 0);
  }
  {
    const int size = 4;
    int a[size] = {3, 2, 1, 0};
    double b[size] = {0, 1, 2, 3};
    float c[size] = {0, 2, 4, 6};
    ads::sortTogether(a, a + size, b, b + size, c, c + size);
    assert(a[0] == 0);
    assert(a[1] == 1);
    assert(a[2] == 2);
    assert(a[3] == 3);
    assert(b[0] == 3);
    assert(b[1] == 2);
    assert(b[2] == 1);
    assert(b[3] == 0);
    assert(c[0] == 6);
    assert(c[1] == 4);
    assert(c[2] == 2);
    assert(c[3] == 0);
  }
  // Compute order.
  {
    const int size = 4;
    double a[size] = {0, 1, 2, 3};
    int b[size];
    int order[size] = {0, 1, 2, 3};
    ads::computeOrder(a, a + size, b);
    for (int i = 0; i != size; ++i) {
      assert(b[i] == order[i]);
    }
  }
  {
    const int size = 4;
    double a[size] = {3, 2, 1, 0};
    int b[size];
    int order[size] = {3, 2, 1, 0};
    ads::computeOrder(a, a + size, b);
    for (int i = 0; i != size; ++i) {
      assert(b[i] == order[i]);
    }
  }
  {
    const int size = 4;
    double a[size] = {0, 4, 2, 3};
    int b[size];
    int order[size] = {0, 2, 3, 1};
    ads::computeOrder(a, a + size, b);
    for (int i = 0; i != size; ++i) {
      assert(b[i] == order[i]);
    }
  }
}
