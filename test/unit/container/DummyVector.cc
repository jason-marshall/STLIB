// -*- C++ -*-

#include "stlib/container/DummyVector.h"

#include <cassert>


void
testInt()
{
  typedef stlib::container::DummyVector<int> DummyVector;
  {
    DummyVector a;
    assert(a.empty());
  }
  {
    DummyVector a(DummyVector::allocator_type{});
    assert(a.empty());
  }
  {
    DummyVector a(std::size_t(10));
    assert(a.empty());
  }
  {
    DummyVector a(std::size_t(10), DummyVector::allocator_type{});
    assert(a.empty());
  }
  {
    DummyVector a(std::size_t(10), 23);
    assert(a.empty());
  }
  {
    DummyVector a(std::size_t(10), 23, DummyVector::allocator_type{});
    assert(a.empty());
  }
  {
    int x;
    DummyVector a(&x, &x + 1);
    assert(a.empty());
  }
  {
    int x;
    DummyVector a(&x, &x + 1, DummyVector::allocator_type{});
    assert(a.empty());
  }
  {
    DummyVector a{2, 3, 5 ,7};
    assert(a.empty());
  }
  {
    auto il = {2, 3, 5 ,7};
    DummyVector a(il, DummyVector::allocator_type{});
    assert(a.empty());
  }
  {
    DummyVector a;
    DummyVector b;
    b = a;
    assert(b.empty());
  }
  {
    auto il = {2, 3, 5 ,7};
    DummyVector a;
    a = il;
    assert(a.empty());
  }

  {
    int x;
    DummyVector a;
    a.assign(&x, &x + 1);
    assert(a.empty());
  }
  {
    DummyVector a;
    a.assign(std::size_t(10), int(23));
    assert(a.empty());
  }
  {
    auto il = {2, 3, 5 ,7};
    DummyVector a;
    a.assign(il);
    assert(a.empty());
  }

  {
    DummyVector a;
    assert(a.size() == 0);
    assert(a.capacity() == 0);
    assert(a.empty());
  }
  {
    DummyVector a;
    a.reserve(10);
    assert(a.empty());
    a.shrink_to_fit();
    assert(a.empty());
  }

  {
    DummyVector a;
    a[0] = 5;
  }
  {
    DummyVector const a;
    assert(a[0] == a[0]);
  }

  {
    DummyVector a;
    a.at(0) = 5;
  }
  {
    DummyVector const a;
    assert(a.at(0) == a.at(0));
  }

  {
    DummyVector a;
    a.front() = 5;
  }
  {
    DummyVector const a;
    assert(a.front() == a.front());
  }

  {
    DummyVector a;
    a.back() = 5;
  }
  {
    DummyVector const a;
    assert(a.back() == a.back());
  }

  {
    DummyVector a;
    a.push_back(23);
    assert(a.empty());
  }
  {
    DummyVector a;
    a.pop_back();
    assert(a.empty());
  }

  {
    DummyVector a;
    a.insert(a.end(), 23);
    assert(a.empty());
  }
  {
    DummyVector a;
    a.insert(a.end(), std::size_t(10), 23);
    assert(a.empty());
  }
  {
    int x = 23;
    DummyVector a;
    a.insert(a.end(), &x, &x + 1);
    assert(a.empty());
  }
  {
    auto il = {2, 3, 5, 7};
    DummyVector a;
    a.insert(a.end(), il);
    assert(a.empty());
  }

  {
    DummyVector a;
    a.erase(a.begin());
    assert(a.empty());
  }
  {
    DummyVector a;
    a.erase(a.begin(), a.end());
    assert(a.empty());
  }

  {
    DummyVector a;
    a.clear();
    assert(a.empty());
  }

  {
    DummyVector a;
    a.resize(10);
    assert(a.empty());
  }
  {
    DummyVector a;
    a.resize(10, 23);
    assert(a.empty());
  }

  {
    DummyVector a;
    DummyVector b;
    a.swap(b);
    assert(a.empty());
  }
}


int
main()
{
  testInt();
  
  return 0;
}
