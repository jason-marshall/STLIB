// -*- C++ -*-

#include "stlib/geom/orq/SequentialScan.h"

#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/ads/functor/Dereference.h"

#include <iostream>
#include <vector>

#include <cassert>

using namespace stlib;

template<typename _Float>
void
test()
{
  typedef std::array<_Float, 3> Value;
  typedef std::vector<Value> ValueContainer;
  typedef typename ValueContainer::const_iterator Record;
  typedef std::vector<Record> RecordContainer;
  typedef geom::SequentialScan<3, ads::Dereference<Record> > ORQDS;
  typedef typename ORQDS::BBox BBox;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  //
  // Constructors
  //

  {
    // Default constructor.
    ORQDS x;
    std::cout << "ORQDS() = \n"
              << x << '\n';
  }
  {
    // Range constructor.
    ValueContainer v;
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    ORQDS x(v.begin(), v.end());
    std::cout << "Use a range of records.\n"
              << x << '\n';
  }

  //
  // Add records.
  //

  {
    std::cout << "Add one at a time." << '\n';
    ORQDS orq;
    ValueContainer v;
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    for (Record i = v.begin(); i != v.end(); ++i) {
      orq.insert(i);
    }
    std::cout << orq << '\n';
  }
  {
    std::cout << "Add a range." << '\n';
    ORQDS orq;
    ValueContainer v;
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    orq.insert(v.begin(), v.end());
    std::cout << orq << '\n';
  }

  //
  // Accesors: size
  //

  {
    ORQDS orq;
    assert(orq.empty());

    ValueContainer v;
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(1 + random()), _Float(2 + random()),
              _Float(3 + 2 * random())}});
    }
    orq.insert(v.begin(), v.end());

    assert(! orq.empty());
    assert(orq.size() == 8);
  }

  //
  // Mathematical member functions
  //

  {
    ValueContainer v;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 4; ++k) {
          v.push_back(Value{{_Float(1. + i),
                  _Float(2 + j / 2.0),
                  _Float(3 + 2 * k / 3.0)}});
        }
      }
    }
    ORQDS orq(v.begin(), v.end());

    RecordContainer vec;

    assert(orq.computeWindowQuery(std::back_inserter(vec),
                                  BBox{{{1., 2., 3.}}, {{2., 3., 5.}}}) == 24);
    assert(vec.size() == 24);
    vec.clear();

    assert(orq.computeWindowQuery(std::back_inserter(vec),
                                  BBox{{{1., 2., 3.}}, {{2., 3., 3.}}}) == 6);
    assert(vec.size() == 6);
    vec.clear();

    assert(orq.computeWindowQuery(std::back_inserter(vec),
                                  BBox{{{1., 2., 3.}}, {{1., 2., 3.}}}) == 1);
    assert(vec.size() == 1);
    vec.clear();

    assert(orq.computeWindowQuery(std::back_inserter(vec),
                                  BBox{{{1., 2., 0.}}, {{2., 3., 1.}}}) == 0);
    assert(vec.size() == 0);
    vec.clear();
  }
}

int
main()
{
  test<float>();
  test<double>();

  return 0;
}
