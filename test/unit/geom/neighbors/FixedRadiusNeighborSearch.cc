// -*- C++ -*-

#include "stlib/geom/neighbors/FixedRadiusNeighborSearch.h"
#include "stlib/ads/functor/Dereference.h"

#include <iostream>
#include <vector>

#include <cassert>

using namespace stlib;

int
main()
{
  typedef std::array<double, 3> Value;
  typedef std::vector<Value> ValueContainer;
  typedef ValueContainer::const_iterator Record;
  typedef geom::FixedRadiusNeighborSearch<3, ads::Dereference<Record> >
  NeighborSearch;

  //
  // Constructors
  //

  {
    ValueContainer data;
    Value v = {{0, 0, 0}};
    data.push_back(v);
    NeighborSearch x(data.begin(), data.end(), 1.);
    assert(x.size() == 1);
  }
  {
    ValueContainer data;
    Value v = {{0, 0, 0}};
    data.push_back(v);
    v[0] = 1;
    data.push_back(v);
    v[0] = 2;
    data.push_back(v);
    v[0] = 4;
    data.push_back(v);
    NeighborSearch x(data.begin(), data.end(), 1.1);
    assert(x.size() == 4);

    std::vector<std::size_t> records;
    x.findNeighbors(std::back_inserter(records), 0);
    assert(records.size() == 1);
    assert(records[0] == 1);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 1);
    assert(records.size() == 2);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 2);
    assert(records.size() == 1);
    assert(records[0] == 1);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 3);
    assert(records.size() == 0);

    // Rebuild and repeat the tests.
    x.rebuild(data.begin(), data.end());

    records.clear();
    x.findNeighbors(std::back_inserter(records), 0);
    assert(records.size() == 1);
    assert(records[0] == 1);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 1);
    assert(records.size() == 2);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 2);
    assert(records.size() == 1);
    assert(records[0] == 1);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 3);
    assert(records.size() == 0);
  }

  return 0;
}
