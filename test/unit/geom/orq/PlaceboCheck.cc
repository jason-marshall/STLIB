// -*- C++ -*-

#include "stlib/geom/orq/PlaceboCheck.h"

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
  typedef geom::PlaceboCheck<3, ads::Dereference<Record> > ORQDS;
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
    assert(x.size() == 0);
    assert(x.empty());
    std::cout << "PlaceboCheck() = \n" << x << '\n';
  }
  {
    // Range constructor.
    ValueContainer v;
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    ORQDS x(v.begin(), v.end());
    assert(x.size() == 8);
    assert(!x.empty());
    std::cout << "Use a range of records.\n" << x << '\n';
  }

  //
  // Add records.
  //

  {
    std::cout << "Add one at a time." << '\n';
    ORQDS x;
    ValueContainer v;
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    for (Record i = v.begin(); i != v.end(); ++i) {
      x.insert(i);
    }
    std::cout << x << '\n';
  }
  {
    std::cout << "Add a range." << '\n';
    ORQDS x;
    ValueContainer v;
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    x.insert(v.begin(), v.end());
    std::cout << x << '\n';
  }

  //
  // Accesors: size
  //

  {
    ORQDS x;
    assert(x.empty());

    ValueContainer v;
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(1 + random()), _Float(2 + random()),
              _Float(3 + 2 * random())}});
    }
    x.insert(v.begin(), v.end());

    assert(! x.empty());
    assert(x.size() == 8);

    x.querySize = 4;
    assert(x.querySize == 4);
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

    ORQDS x(v.begin(), v.end());
    x.querySize = 4;

    std::vector<Record> vec;

    assert(x.computeWindowQuery(std::back_inserter(vec),
                                BBox{{{1., 2., 3.}}, {{2., 3., 5.}}}) == 4);
    assert(vec.size() == 4);
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
