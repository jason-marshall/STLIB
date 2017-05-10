// -*- C++ -*-

#include "stlib/ext/array.h"

#include <unordered_map>

#include <cassert>

using namespace stlib;

namespace std
{
template<>
struct hash<array<std::size_t, 2> > {
  std::size_t
  operator()(const array<std::size_t, 2>& x) const
  {
    hash<std::size_t> h;
    return h(x[0] + x[1]);
  }
};
}

int
main()
{
  {
    typedef std::unordered_map<std::size_t, double> HashTable;
    typedef HashTable::value_type Value;
    HashTable x;
    assert(x.empty());
    x.insert(Value(0, 0.));
    assert(! x.empty());
    // CONTINUE: The copy constructor causes a compilation error.
    //HashTable y(x);
    HashTable y(x.begin(), x.end());
    assert(! y.empty());
  }
  {
    typedef std::unordered_map<std::array<std::size_t, 2>, double>
    HashTable;
    typedef HashTable::value_type Value;
    HashTable x;
    assert(x.empty());
    x.insert(Value(std::array<std::size_t, 2>{{0, 0}}, 0.));
    assert(! x.empty());
    // CONTINUE: The copy constructor causes a compilation error.
    //HashTable y(x);
    HashTable y(x.begin(), x.end());
    assert(! y.empty());
  }
  return 0;
}
