// -*- C++ -*-

#include "stlib/container/VectorFixedCapacity.h"

int
main()
{
  using stlib::container::VectorFixedCapacity;

  // Default constructor.
  {
    VectorFixedCapacity<float, 0> x;
    assert(x.empty());
    assert(x.max_size() == 0);
  }
  {
    VectorFixedCapacity<float, 1> x;
    assert(x.empty());
    assert(x.max_size() == 1);
  }

  // Size constructor.
  {
    VectorFixedCapacity<float, 0> x(0);
    assert(x.empty());
  }
  {
    VectorFixedCapacity<float, 1> x(1);
    assert(x.size() == 1);
    assert(x[0] == 0);
  }
  {
    VectorFixedCapacity<float, 2> x(1);
    assert(x.size() == 1);
    assert(x[0] == 0);
  }

  // Size/value constructor.
  {
    VectorFixedCapacity<float, 0> x(0, float(23));
    assert(x.empty());
  }
  {
    VectorFixedCapacity<float, 1> x(1, float(23));
    assert(x.size() == 1);
    assert(x[0] == 23);
  }
  {
    VectorFixedCapacity<float, 2> x(1, float(23));
    assert(x.size() == 1);
    assert(x[0] == 23);
  }

  // Copy constructor.
  {
    VectorFixedCapacity<float, 0> x;
    VectorFixedCapacity<float, 0> y = x;
    assert(y == x);
  }
  {
    VectorFixedCapacity<float, 1> x(1, float(23));
    VectorFixedCapacity<float, 1> y = x;
    assert(y == x);
  }
  {
    VectorFixedCapacity<float, 2> x(1, float(23));
    VectorFixedCapacity<float, 2> y = x;
    assert(y == x);
  }

  // Construct from an initializer list.
  {
    VectorFixedCapacity<float, 0> x = {};
    assert(x.empty());
  }
  {
    VectorFixedCapacity<float, 1> x = {23};
    assert(x.size() == 1);
    assert(x[0] == 23);
  }
  {
    VectorFixedCapacity<float, 2> x = {23};
    assert(x.size() == 1);
    assert(x[0] == 23);
  }
  {
    VectorFixedCapacity<float, 2> x = {2, 3};
    assert(x.size() == 2);
    assert(x[0] == 2);
    assert(x[1] == 3);
  }

  // Construct from a range.
  {
    float const* p = nullptr;
    VectorFixedCapacity<float, 0> x(p, p);
    assert(x.empty());
  }
  {
    std::array<float, 1> a = {{23}};
    VectorFixedCapacity<float, 1> x(a.begin(), a.end());
    assert(x.size() == 1);
    assert(x[0] == 23);
  }
  {
    std::array<float, 2> a = {{23}};
    VectorFixedCapacity<float, 2> x(a.begin(), a.begin() + 1);
    assert(x.size() == 1);
    assert(x[0] == 23);
  }
  {
    std::array<float, 2> a = {{2, 3}};
    VectorFixedCapacity<float, 2> x(a.begin(), a.end());
    assert(x.size() == 2);
    assert(x[0] == 2);
    assert(x[1] == 3);
  }
  // Dispatched to size/value constructor.
  {
    VectorFixedCapacity<float, 0> x(0, 23);
    assert(x.empty());
  }
  {
    VectorFixedCapacity<float, 1> x(1, 23);
    assert(x.size() == 1);
    assert(x[0] == 23);
  }
  {
    VectorFixedCapacity<float, 2> x(1, 23);
    assert(x.size() == 1);
    assert(x[0] == 23);
  }

  // Copy constructor.
  {
    VectorFixedCapacity<float, 0> x = {};
    VectorFixedCapacity<float, 0> y;
    y = x;
    assert(y == x);
  }
  {
    VectorFixedCapacity<float, 1> x = {23};
    VectorFixedCapacity<float, 1> y;
    y = x;
    assert(y == x);
  }
  {
    VectorFixedCapacity<float, 2> x = {23};
    VectorFixedCapacity<float, 2> y;
    y = x;
    assert(y == x);
  }
  {
    VectorFixedCapacity<float, 2> x = {2, 3};
    VectorFixedCapacity<float, 2> y;
    y = x;
    assert(y == x);
  }

  // Assign from an initializer list.
  {
    VectorFixedCapacity<float, 0> x;
    x = {};
    assert(x.empty());
  }
  {
    VectorFixedCapacity<float, 1> x;
    x = {23};
    assert(x.size() == 1);
    assert(x[0] == 23);
  }
  {
    VectorFixedCapacity<float, 2> x;
    x = {23};
    assert(x.size() == 1);
    assert(x[0] == 23);
  }
  {
    VectorFixedCapacity<float, 2> x;
    x = {2, 3};
    assert(x.size() == 2);
    assert(x[0] == 2);
    assert(x[1] == 3);
  }

  // fill()
  {
    using Vector = VectorFixedCapacity<float, 2>;
    Vector x;
    x.fill(2);
    assert(x.empty());

    x.resize(1);
    x.fill(2);
    assert(x == Vector{2});

    x.resize(2);
    x.fill(2);
    assert(x == (Vector{2, 2}));
  }

  // swap()
  {
    using Vector = VectorFixedCapacity<float, 2>;
    Vector x{2, 3};
    Vector y{5};
    x.swap(y);
    assert(x == Vector{5});
    assert(y == (Vector{2, 3}));
  }

  // resize()
  {
    VectorFixedCapacity<float, 2> x;
    x.resize(0);
    assert(x.empty());
    x.resize(1);
    assert(x.size() == 1);
    assert(x[0] == 0);
    x.resize(2, 3);
    assert(x.size() == 2);
    assert(x[0] == 0);
    assert(x[1] == 3);
  }

  // push_back() and pop_back()
  {
    using Vector = VectorFixedCapacity<float, 2>;
    Vector x;
    x.push_back(2);
    assert(x == Vector{2});
    x.push_back(3);
    assert(x == (Vector{2, 3}));
    x.pop_back();
    assert(x == Vector{2});
    x.pop_back();
    assert(x == Vector{});
  }

  return 0;
}
