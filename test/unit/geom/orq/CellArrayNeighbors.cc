// -*- C++ -*-

#include "stlib/geom/orq/CellArrayNeighbors.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <iostream>
#include <vector>

#include <cassert>

using namespace stlib;

template<typename _Float>
void
test2()
{
  const std::size_t D = 2;
  typedef std::array<_Float, D> Value;
  typedef std::vector<Value> ValueContainer;
  typedef typename ValueContainer::const_iterator Record;
  typedef geom::CellArrayNeighbors<_Float, D, Record> NS;
  typedef typename NS::Point Point;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  {
    std::vector<Record> neighbors;
    // No records.
    NS ns;
    ValueContainer values;
    ns.initialize(values.begin(), values.end());
    ns.neighborQuery(Point{{0., 0.}}, 1., &neighbors);
    assert(neighbors.empty());
    // One record.
    values.push_back(Value{{_Float(0), _Float(0)}});
    ns.initialize(values.begin(), values.end());
    ns.neighborQuery(Point{{1.1, 0.}}, 1., &neighbors);
    assert(neighbors.empty());
    ns.neighborQuery(Point{{0., 0.}}, 1., &neighbors);
    assert(neighbors.size() == 1);
    // Two records.
    values.push_back(Value{{_Float(1.), _Float(1.)}});
    ns.initialize(values.begin(), values.end());
    ns.neighborQuery(Point{{-1.1, 0.}}, 1., &neighbors);
    assert(neighbors.empty());
    ns.neighborQuery(Point{{0., 0.}}, 1., &neighbors);
    assert(neighbors.size() == 1);
    ns.neighborQuery(Point{{1., 1.}}, 1., &neighbors);
    assert(neighbors.size() == 1);
    ns.neighborQuery(Point{{0.5, 0.5}}, 1., &neighbors);
    assert(neighbors.size() == 2);
  }
}

template<typename _Float>
void
test3()
{
  typedef std::array<_Float, 3> Value;
  typedef std::vector<Value> ValueContainer;
  typedef typename ValueContainer::const_iterator Record;
  typedef geom::CellArrayNeighbors<_Float, 3, Record> NS;
  typedef typename NS::Point Point;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  {
    std::vector<Record> neighbors;
    // No records.
    NS ns;
    ValueContainer values;
    ns.initialize(values.begin(), values.end());
    ns.neighborQuery(Point{{0., 0., 0.}}, 1., &neighbors);
    assert(neighbors.empty());
    // One record.
    values.push_back(Value{{_Float(0.), _Float(0.), _Float(0.)}});
    ns.initialize(values.begin(), values.end());
    ns.neighborQuery(Point{{1.1, 0., 0.}}, 1., &neighbors);
    assert(neighbors.empty());
    ns.neighborQuery(Point{{0., 0., 0.}}, 1., &neighbors);
    assert(neighbors.size() == 1);
    // Two records.
    values.push_back(Value{{_Float(1.), _Float(1.), _Float(1.)}});
    ns.initialize(values.begin(), values.end());
    ns.neighborQuery(Point{{-1.1, 0., 0.}}, 1., &neighbors);
    assert(neighbors.empty());
    ns.neighborQuery(Point{{0., 0., 0.}}, 1., &neighbors);
    assert(neighbors.size() == 1);
    ns.neighborQuery(Point{{1., 1., 1.}}, 1., &neighbors);
    assert(neighbors.size() == 1);
    ns.neighborQuery(Point{{0.5, 0.5, 0.5}}, 1., &neighbors);
    assert(neighbors.size() == 2);
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
    NS ns;
    ns.initialize(values.begin(), values.end());
    std::vector<Record> neighbors;
    for (std::size_t i = 0; i != values.size(); ++i) {
      ns.neighborQuery(values[i], 1., &neighbors);
    }
  }
}

int
main()
{
  test2<float>();
  test2<double>();
  test3<float>();
  test3<double>();

  return 0;
}
