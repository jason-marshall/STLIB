// -*- C++ -*-

#include "stlib/container/vector.h"
//#include <iostream>

using namespace stlib;

int
main()
{
  {
    container::vector<float> a;
    assert(a.empty());
  }
  {
    container::vector<float> a(0);
    assert(a.size() == 0);
  }
  {
    container::vector<float> a(10);
    assert(a.size() == 10);
  }
  {
    container::vector<float> a(0, 0.);
    assert(a.size() == 0);
  }
  {
    container::vector<float> a(1, 2.);
    assert(a.size() == 1);
    assert(a[0] == 2);
  }
  {
    // Range constructor.
    const float data[] = {2, 3, 5, 7};
    const std::size_t size = sizeof(data) / sizeof(float);
    container::vector<float> a(data, data + size);
    assert(a.size() == size);
    for (std::size_t i = 0; i != a.size(); ++i) {
      assert(a[i] == data[i]);
    }
    // Copy constructor.
    container::vector<float> b = a;
    assert(b == a);
    // Assignment operator.
    container::vector<float> c;
    c = a;
    assert(c == a);
    // Assign size and value.
    a.assign(2, 3.);
    assert(a.size() == 2);
    assert(a[0] == 3 && a[1] == 3);
    // Assign range.
    a.assign(data, data + size);
    assert(a.size() == size);
    for (std::size_t i = 0; i != a.size(); ++i) {
      assert(a[i] == data[i]);
    }
    // Resize.
    a.resize(3);
    assert(a.size() == 3);
    a.resize(2, 7.);
    assert(a.size() == 2);
    assert(a[0] == 7 && a[1] == 7);
  }
  {
    // Capacity.
    container::vector<float> a;
    assert(a.capacity() == 0);
    // Reserve.
    a.reserve(3);
    assert(a.empty());
    assert(a.capacity() == 3);
  }
  {
    // push_back.
    container::vector<float> a;
    a.push_back(2);
    assert(a.size() == 1);
    assert(a[0] == 2);
    // insert end.
    container::vector<float>::iterator i = a.insert(a.end(), 3);
    assert(*i == 3);
    assert(a.size() == 2);
    assert(a[0] == 2 && a[1] == 3);
    // insert begin.
    i = a.insert(a.begin(), 1);
    assert(*i == 1);
    assert(a.size() == 3);
    assert(a[0] == 1 && a[1] == 2 && a[2] == 3);
    // erase last.
    i = a.erase(a.end() - 1);
    assert(i == a.end());
    assert(a.size() == 2);
    assert(a[0] == 1 && a[1] == 2);
    // erase first.
    i = a.erase(a.begin());
    assert(i == a.begin());
    assert(a.size() == 1);
    assert(a[0] == 2);
  }
  {
    // swap.
    container::vector<float> a;
    a.push_back(2);
    a.push_back(3);
    a.push_back(5);
    container::vector<float> b;
    b.swap(a);
    assert(a.empty());
    assert(b.size() == 3);
    assert(b[0] == 2 && b[1] == 3 && b[2] == 5);
    // Clear.
    b.clear();
    assert(b.empty());
  }

  return 0;
}
