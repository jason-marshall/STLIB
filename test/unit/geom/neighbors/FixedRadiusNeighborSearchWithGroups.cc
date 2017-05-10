// -*- C++ -*-

#include "stlib/geom/neighbors/FixedRadiusNeighborSearchWithGroups.h"
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
  typedef geom::FixedRadiusNeighborSearchWithGroups
  <3, ads::Dereference<Record> > NeighborSearch;

  //
  // Constructors
  //

  {
    // 1 record, 1 group.
    ValueContainer data;
    Value v = {{0, 0, 0}};
    data.push_back(v);
    std::vector<std::size_t> groupIndices(data.size(), 0);
    NeighborSearch x(data.begin(), data.end(), groupIndices.begin(), 1.);
    assert(x.size() == 1);

    std::vector<std::size_t> records;
    std::vector<std::size_t> groups;

    records.clear();
    x.findNeighbors(std::back_inserter(records), 0);
    assert(records.size() == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    0);
    assert(records.size() == 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);
  }
  {
    // 2 records, 1 group, radius = 0.5.
    ValueContainer data;
    Value v = {{0, 0, 0}};
    data.push_back(v);
    v[0] = 1;
    data.push_back(v);
    std::vector<std::size_t> groupIndices(data.size(), 0);
    NeighborSearch x(data.begin(), data.end(), groupIndices.begin(), 0.5);
    assert(x.size() == 2);

    std::vector<std::size_t> records;
    std::vector<std::size_t> groups;

    records.clear();
    x.findNeighbors(std::back_inserter(records), 0);
    assert(records.size() == 1);
    assert(records[0] == 1);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 1);
    assert(records.size() == 1);
    assert(records[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    0);
    assert(records.size() == 1);
    assert(records[0] == 1);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    1);
    assert(records.size() == 1);
    assert(records[0] == 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 1);
    assert(groups.size() == 1);
    assert(groups[0] == 0);
  }
  {
    // 2 records, 1 group, radius = 1.5.
    ValueContainer data;
    Value v = {{0, 0, 0}};
    data.push_back(v);
    v[0] = 1;
    data.push_back(v);
    std::vector<std::size_t> groupIndices(data.size(), 0);
    NeighborSearch x(data.begin(), data.end(), groupIndices.begin(), 1.5);
    assert(x.size() == 2);

    std::vector<std::size_t> records;
    std::vector<std::size_t> groups;

    records.clear();
    x.findNeighbors(std::back_inserter(records), 0);
    assert(records.size() == 1);
    assert(records[0] == 1);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 1);
    assert(records.size() == 1);
    assert(records[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    0);
    assert(records.size() == 1);
    assert(records[0] == 1);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    1);
    assert(records.size() == 1);
    assert(records[0] == 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 1);
    assert(groups.size() == 1);
    assert(groups[0] == 0);
  }
  {
    // 2 records, 2 groups, radius = 0.5.
    ValueContainer data;
    Value v = {{0, 0, 0}};
    data.push_back(v);
    v[0] = 1;
    data.push_back(v);
    std::vector<std::size_t> groupIndices(data.size());
    groupIndices[0] = 0;
    groupIndices[1] = 1;
    NeighborSearch x(data.begin(), data.end(), groupIndices.begin(), 0.5);
    assert(x.size() == 2);

    std::vector<std::size_t> records;
    std::vector<std::size_t> groups;

    records.clear();
    x.findNeighbors(std::back_inserter(records), 0);
    assert(records.size() == 0);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 1);
    assert(records.size() == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    0);
    assert(records.size() == 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    1);
    assert(records.size() == 0);
    assert(groups.size() == 1);
    assert(groups[0] == 1);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 1);
    assert(groups.size() == 1);
    assert(groups[0] == 1);
  }
  {
    // 2 records, 2 groups, radius = 1.5.
    ValueContainer data;
    Value v = {{0, 0, 0}};
    data.push_back(v);
    v[0] = 1;
    data.push_back(v);
    std::vector<std::size_t> groupIndices(data.size());
    groupIndices[0] = 0;
    groupIndices[1] = 1;
    NeighborSearch x(data.begin(), data.end(), groupIndices.begin(), 1.5);
    assert(x.size() == 2);

    std::vector<std::size_t> records;
    std::vector<std::size_t> groups;

    records.clear();
    x.findNeighbors(std::back_inserter(records), 0);
    assert(records.size() == 1);
    assert(records[0] == 1);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 1);
    assert(records.size() == 1);
    assert(records[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    0);
    assert(records.size() == 1);
    assert(records[0] == 1);
    assert(groups.size() == 2);
    std::sort(groups.begin(), groups.end());
    assert(groups[0] == 0 && groups[1] == 1);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    1);
    assert(records.size() == 1);
    assert(records[0] == 0);
    assert(groups.size() == 2);
    std::sort(groups.begin(), groups.end());
    assert(groups[0] == 0 && groups[1] == 1);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 0);
    assert(groups.size() == 2);
    std::sort(groups.begin(), groups.end());
    assert(groups[0] == 0 && groups[1] == 1);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 1);
    assert(groups.size() == 2);
    std::sort(groups.begin(), groups.end());
    assert(groups[0] == 0 && groups[1] == 1);
  }
  {
    // 3 records, 1 group, radius = 0.5.
    ValueContainer data;
    Value v = {{0, 0, 0}};
    data.push_back(v);
    v[0] = 1;
    data.push_back(v);
    v[0] = 2;
    data.push_back(v);
    std::vector<std::size_t> groupIndices(data.size(), 0);
    NeighborSearch x(data.begin(), data.end(), groupIndices.begin(), 0.5);
    assert(x.size() == 3);

    std::vector<std::size_t> records;
    std::vector<std::size_t> groups;

    // Records.
    records.clear();
    x.findNeighbors(std::back_inserter(records), 0);
    std::sort(records.begin(), records.end());
    assert(records.size() == 2);
    assert(records[0] == 1);
    assert(records[1] == 2);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 1);
    std::sort(records.begin(), records.end());
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 2);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 2);
    std::sort(records.begin(), records.end());
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 1);

    // Records and groups.
    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    0);
    std::sort(records.begin(), records.end());
    assert(records.size() == 2);
    assert(records[0] == 1);
    assert(records[1] == 2);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    1);
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 2);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    2);
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 1);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    // Groups.
    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 1);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 2);
    assert(groups.size() == 1);
    assert(groups[0] == 0);
  }
  {
    // 3 records, 1 group, radius = 1.5.
    ValueContainer data;
    Value v = {{0, 0, 0}};
    data.push_back(v);
    v[0] = 1;
    data.push_back(v);
    v[0] = 2;
    data.push_back(v);
    std::vector<std::size_t> groupIndices(data.size(), 0);
    NeighborSearch x(data.begin(), data.end(), groupIndices.begin(), 1.5);
    assert(x.size() == 3);

    std::vector<std::size_t> records;
    std::vector<std::size_t> groups;

    // Records.
    records.clear();
    x.findNeighbors(std::back_inserter(records), 0);
    std::sort(records.begin(), records.end());
    assert(records.size() == 2);
    assert(records[0] == 1);
    assert(records[1] == 2);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 1);
    std::sort(records.begin(), records.end());
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 2);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 2);
    std::sort(records.begin(), records.end());
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 1);

    // Records and groups.
    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    0);
    std::sort(records.begin(), records.end());
    assert(records.size() == 2);
    assert(records[0] == 1);
    assert(records[1] == 2);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    1);
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 2);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    2);
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 1);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    // Groups.
    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 1);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 2);
    assert(groups.size() == 1);
    assert(groups[0] == 0);
  }
  {
    // 3 records, 2 groups {0, 1}, {2}, radius = 0.5.
    ValueContainer data;
    Value v = {{0, 0, 0}};
    data.push_back(v);
    v[0] = 1;
    data.push_back(v);
    v[0] = 2;
    data.push_back(v);
    std::vector<std::size_t> groupIndices(data.size(), 0);
    groupIndices[2] = 1;
    NeighborSearch x(data.begin(), data.end(), groupIndices.begin(), 0.5);
    assert(x.size() == 3);

    std::vector<std::size_t> records;
    std::vector<std::size_t> groups;

    // Records.
    records.clear();
    x.findNeighbors(std::back_inserter(records), 0);
    std::sort(records.begin(), records.end());
    assert(records.size() == 1);
    assert(records[0] == 1);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 1);
    std::sort(records.begin(), records.end());
    assert(records.size() == 1);
    assert(records[0] == 0);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 2);
    std::sort(records.begin(), records.end());
    assert(records.size() == 0);

    // Records and groups.
    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    0);
    std::sort(records.begin(), records.end());
    assert(records.size() == 1);
    assert(records[0] == 1);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    1);
    assert(records.size() == 1);
    assert(records[0] == 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    2);
    assert(records.size() == 0);
    assert(groups.size() == 1);
    assert(groups[0] == 1);

    // Groups.
    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 1);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 2);
    assert(groups.size() == 1);
    assert(groups[0] == 1);
  }
  {
    // 3 records, 2 groups {0, 1}, {2}, radius = 1.5.
    ValueContainer data;
    Value v = {{0, 0, 0}};
    data.push_back(v);
    v[0] = 1;
    data.push_back(v);
    v[0] = 2;
    data.push_back(v);
    std::vector<std::size_t> groupIndices(data.size(), 0);
    groupIndices[2] = 1;
    NeighborSearch x(data.begin(), data.end(), groupIndices.begin(), 1.5);
    assert(x.size() == 3);

    std::vector<std::size_t> records;
    std::vector<std::size_t> groups;

    // Records.
    records.clear();
    x.findNeighbors(std::back_inserter(records), 0);
    std::sort(records.begin(), records.end());
    assert(records.size() == 1);
    assert(records[0] == 1);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 1);
    std::sort(records.begin(), records.end());
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 2);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 2);
    std::sort(records.begin(), records.end());
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 1);

    // Records and groups.
    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    0);
    std::sort(records.begin(), records.end());
    std::sort(groups.begin(), groups.end());
    assert(records.size() == 1);
    assert(records[0] == 1);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    1);
    std::sort(records.begin(), records.end());
    std::sort(groups.begin(), groups.end());
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 2);
    assert(groups.size() == 2);
    assert(groups[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    2);
    std::sort(records.begin(), records.end());
    std::sort(groups.begin(), groups.end());
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 1);
    assert(groups.size() == 2);
    assert(groups[0] == 0);
    assert(groups[1] == 1);

    // Groups.
    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 0);
    std::sort(groups.begin(), groups.end());
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 1);
    std::sort(groups.begin(), groups.end());
    assert(groups.size() == 2);
    assert(groups[0] == 0);
    assert(groups[1] == 1);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 2);
    std::sort(groups.begin(), groups.end());
    assert(groups.size() == 2);
    assert(groups[0] == 0);
    assert(groups[1] == 1);
  }
  {
    // 3 records, 3 groups, radius = 0.5.
    ValueContainer data;
    Value v = {{0, 0, 0}};
    data.push_back(v);
    v[0] = 1;
    data.push_back(v);
    v[0] = 2;
    data.push_back(v);
    std::vector<std::size_t> groupIndices(data.size(), 0);
    groupIndices[1] = 1;
    groupIndices[2] = 2;
    NeighborSearch x(data.begin(), data.end(), groupIndices.begin(), 0.5);
    assert(x.size() == 3);

    std::vector<std::size_t> records;
    std::vector<std::size_t> groups;

    // Records.
    records.clear();
    x.findNeighbors(std::back_inserter(records), 0);
    assert(records.size() == 0);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 1);
    assert(records.size() == 0);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 2);
    assert(records.size() == 0);

    // Records and groups.
    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    0);
    assert(records.size() == 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    1);
    assert(records.size() == 0);
    assert(groups.size() == 1);
    assert(groups[0] == 1);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    2);
    assert(records.size() == 0);
    assert(groups.size() == 1);
    assert(groups[0] == 2);

    // Groups.
    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 0);
    assert(groups.size() == 1);
    assert(groups[0] == 0);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 1);
    assert(groups.size() == 1);
    assert(groups[0] == 1);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 2);
    assert(groups.size() == 1);
    assert(groups[0] == 2);
  }
  {
    // 3 records, 3 groups, radius = 1.5.
    ValueContainer data;
    Value v = {{0, 0, 0}};
    data.push_back(v);
    v[0] = 1;
    data.push_back(v);
    v[0] = 2;
    data.push_back(v);
    std::vector<std::size_t> groupIndices(data.size(), 0);
    groupIndices[1] = 1;
    groupIndices[2] = 2;
    NeighborSearch x(data.begin(), data.end(), groupIndices.begin(), 1.5);
    assert(x.size() == 3);

    std::vector<std::size_t> records;
    std::vector<std::size_t> groups;

    // Records.
    records.clear();
    x.findNeighbors(std::back_inserter(records), 0);
    std::sort(records.begin(), records.end());
    assert(records.size() == 1);
    assert(records[0] == 1);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 1);
    std::sort(records.begin(), records.end());
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 2);

    records.clear();
    x.findNeighbors(std::back_inserter(records), 2);
    std::sort(records.begin(), records.end());
    assert(records.size() == 1);
    assert(records[0] == 1);

    // Records and groups.
    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    0);
    std::sort(records.begin(), records.end());
    std::sort(groups.begin(), groups.end());
    assert(records.size() == 1);
    assert(records[0] == 1);
    assert(groups.size() == 2);
    assert(groups[0] == 0);
    assert(groups[1] == 1);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    1);
    std::sort(records.begin(), records.end());
    std::sort(groups.begin(), groups.end());
    assert(records.size() == 2);
    assert(records[0] == 0);
    assert(records[1] == 2);
    assert(groups.size() == 3);
    assert(groups[0] == 0);
    assert(groups[1] == 1);
    assert(groups[2] == 2);

    records.clear();
    groups.clear();
    x.findNeighbors(std::back_inserter(records), std::back_inserter(groups),
                    2);
    std::sort(records.begin(), records.end());
    std::sort(groups.begin(), groups.end());
    assert(records.size() == 1);
    assert(records[0] == 1);
    assert(groups.size() == 2);
    assert(groups[0] == 1);
    assert(groups[1] == 2);

    // Groups.
    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 0);
    std::sort(groups.begin(), groups.end());
    assert(groups.size() == 2);
    assert(groups[0] == 0);
    assert(groups[1] == 1);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 1);
    std::sort(groups.begin(), groups.end());
    assert(groups.size() == 3);
    assert(groups[0] == 0);
    assert(groups[1] == 1);
    assert(groups[2] == 2);

    groups.clear();
    x.findNeighboringGroups(std::back_inserter(groups), 2);
    std::sort(groups.begin(), groups.end());
    assert(groups.size() == 2);
    assert(groups[0] == 1);
    assert(groups[1] == 2);
  }

  return 0;
}
