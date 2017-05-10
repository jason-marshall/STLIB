// -*- C++ -*-

#include "stlib/geom/orq/SparseCellArray.h"

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
  typedef geom::SparseCellArray<3, ads::Dereference<Record> > ORQDS;
  typedef typename ORQDS::Point Point;
  typedef typename ORQDS::BBox BBox;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  //
  // Constructors
  //

  {
    // Construct from grid dimensions and Cartesian domain.
    ORQDS x(Point{{0.5, 0.34, 0.25}},
            BBox{{{0., 0., 0.}}, {{1., 1., 1.}}});
    std::cout << "SparseCellArray((0.5,0.34,0.25), (0,0,0,1,1,1)) = \n"
              << x << '\n';
  }
  {
    ValueContainer v;
    // Unit, uniform distribution.
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    {
      ORQDS x(Point{{0.5, 0.34, 0.25}},
              BBox{{{0., 0., 0.}}, {{1., 1., 1.}}},
              v.begin(), v.end());
      std::cout << "Use the domain and a range of initial values.\n"
                << x << '\n';
    }
    {
      ORQDS x(Point{{0.5, 0.34, 0.25}}, v.begin(), v.end());
      std::cout << "Use a range of initial values.\n" << x << '\n';
    }
  }

  //
  // Add grid elements.
  //

  {
    std::cout << "Add one at a time.\n";
    ValueContainer v;
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    ORQDS x(Point{{0.5, 0.34, 0.25}},
            BBox{{{0., 0., 0.}}, {{1., 1., 1.}}});
    for (Record i = v.begin(); i != v.end(); ++i) {
      x.insert(i);
    }
    std::cout << x << '\n';
  }
  {
    std::cout << "Add a range.\n";
    ValueContainer v;
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    ORQDS x(Point{{0.5, 0.34, 0.25}},
            BBox{{{0., 0., 0.}}, {{1., 1., 1.}}});
    x.insert(v.begin(), v.end());
    std::cout << x << '\n';
  }

  //
  // Accesors: grid size
  //

  {
    ORQDS x(Point{{0.5, 0.34, 0.25}},
            BBox{{{1., 2., 3.}}, {{2., 3., 5.}}});
    assert(x.empty());

    ValueContainer v;
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(1 + random()), _Float(2 + random()),
              _Float(3 + 2 * random())}});
    }
    x.insert(v.begin(), v.end());

    assert(! x.empty());
    assert(x.size() == 8);
  }

  //
  // Accesors: Cartesian box
  //

  {
    ORQDS x(Point{{0.5, 0.34, 0.25}},
            BBox{{{1., 2., 3.}}, {{2., 3., 5.}}});
    assert(x.getDomain() ==
           (BBox{{{1., 2., 3.}}, {{2., 3., 5.}}}));
  }

  //
  // Mathematical member functions
  //

  {
    ORQDS x(Point{{0.5, 0.34, 0.25}},
            BBox{{{1., 2., 3.}}, {{2.0001, 3.0001, 5.0001}}});
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
    x.insert(v.begin(), v.end());

    std::vector<Record> vec;

    assert(x.computeWindowQuery(std::back_inserter(vec),
                                BBox{{{1., 2., 3.}}, {{2., 3., 5.}}}) == 24);
    assert(vec.size() == 24);
    vec.clear();

    assert(x.computeWindowQuery(std::back_inserter(vec),
                                BBox{{{1., 2., 3.}}, {{2., 3., 3.}}}) == 6);
    assert(vec.size() == 6);
    vec.clear();

    assert(x.computeWindowQuery(std::back_inserter(vec),
                                BBox{{{1., 2., 3.}}, {{1., 2., 3.}}}) == 1);
    assert(vec.size() == 1);
    vec.clear();

    assert(x.computeWindowQuery(std::back_inserter(vec),
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
