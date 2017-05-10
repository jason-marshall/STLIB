// -*- C++ -*-

#include "stlib/geom/orq/CellArrayAllNeighbors.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <iostream>
#include <vector>

#include <cassert>

using namespace stlib;

template<typename _Float>
struct Node {
  _Float* coords;
};

template<typename _Float>
struct Location :
    public std::unary_function<Node<_Float>*, std::array<_Float, 3> > {
  typedef std::unary_function<Node<_Float>*, std::array<_Float, 3> > Base;
  typename Base::result_type
  operator()(typename Base::argument_type r)
  {
    typename Base::result_type location =
    {{r->coords[0], r->coords[1], r->coords[2]}};
    return location;
  }
};

template<typename _Float>
void
test()
{
  // Test the location functor.
  {
    geom::CellArrayAllNeighbors<3, Node<_Float>*, Location<_Float> > ns(1.);
    Node<_Float>* dummy = 0;
    ns.allNeighbors(dummy, dummy);
    assert(ns.records.empty());
    assert(ns.packedNeighbors.empty());
    assert(ns.neighborDelimiters.empty());
  }

  typedef std::array<_Float, 3> Value;
  typedef std::vector<Value> ValueContainer;
  typedef typename ValueContainer::const_iterator Record;
  typedef geom::CellArrayAllNeighbors<3, Record> NS;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  {
    // No records.
    NS ns(1.);
    ValueContainer values;
    ns.allNeighbors(values.begin(), values.end());
    assert(ns.records.empty());
    assert(ns.packedNeighbors.empty());
    assert(ns.neighborDelimiters.empty());
    // One record.
    values.push_back(Value{{0., 0., 0.}});
    ns.allNeighbors(values.begin(), values.end());
    assert(ns.records.size() == values.size());
    assert(ns.packedNeighbors.empty());
    assert(ns.neighborDelimiters.size() == values.size() + 1);
    // Two records, no neighbors.
    values.push_back(Value{{1., 1., 1.}});
    ns.allNeighbors(values.begin(), values.end());
    assert(ns.records.size() == values.size());
    assert(ns.packedNeighbors.empty());
    assert(ns.neighborDelimiters.size() == values.size() + 1);
    // Two records, one neighbor each.
    values[1] = Value{{0., 0., 0.}};
    ns.allNeighbors(values.begin(), values.end());
    assert(ns.records.size() == values.size());
    assert(ns.packedNeighbors.size() == 2);
    assert(ns.neighborDelimiters.size() == values.size() + 1);
    assert(ns.neighborDelimiters[0] == 0);
    assert(ns.neighborDelimiters[1] == 1);
    assert(ns.neighborDelimiters[2] == 2);
    // Two records, one neighbor each.
    values[1] = Value{{0.99, 0., 0.}};
    ns.allNeighbors(values.begin(), values.end());
    assert(ns.records.size() == values.size());
    assert(ns.packedNeighbors.size() == 2);
    assert(ns.neighborDelimiters.size() == values.size() + 1);
    assert(ns.neighborDelimiters[0] == 0);
    assert(ns.neighborDelimiters[1] == 1);
    assert(ns.neighborDelimiters[2] == 2);
    // Three records, two neighbors each.
    values.push_back(Value{{0., 0., 0.}});
    ns.allNeighbors(values.begin(), values.end());
    assert(ns.records.size() == values.size());
    assert(ns.packedNeighbors.size() == 3 * 2);
    assert(ns.neighborDelimiters.size() == values.size() + 1);
    assert(ns.neighborDelimiters[0] == 0);
    assert(ns.neighborDelimiters[1] == 2);
    assert(ns.neighborDelimiters[2] == 4);
    assert(ns.neighborDelimiters[3] == 6);
  }

  {
    // Bo's test data that used to cause a seg fault.
    const Value data[] = {
      {{ -3.2, 0, 19.44}},
      {{3.2, 0, 19.44}},
      {{1.6, 2.77128, 19.44}},
      {{ -1.6, 2.77128, 19.44}},
      {{ -1.6, -2.77128, 19.44}},
      {{1.6, -2.77128, 19.44}},
      {{ -3.2, 0, 25.92}},
      {{3.2, 0, 25.92}},
      {{1.6, 2.77128, 25.92}},
      {{ -1.6, 2.77128, 25.92}},
      {{ -1.6, -2.77128, 25.92}},
      {{1.6, -2.77128, 25.92}},
      {{0, 0, 32.4}}
    };
    ValueContainer values;
    for (std::size_t i = 0; i != sizeof(data) / sizeof(Value); ++i) {
      values.push_back(data[i]);
    }
    NS ns(1.5);
    ns.allNeighbors(values.begin(), values.end());
  }
}

int
main()
{
  test<float>();
  test<double>();

  return 0;
}
