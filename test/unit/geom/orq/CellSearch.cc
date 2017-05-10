// -*- C++ -*-

#include "stlib/geom/orq/CellSearch.h"

#include "stlib/ads/functor/Dereference.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

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
  typedef geom::Search<3, ads::Dereference<Record> > Search;
  typedef geom::CellSearch<3, ads::Dereference<Record>, Search> ORQDS;
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
    // Construct from cell dimensions and Cartesian domain.
    ORQDS x(Point{{0.5, 0.34, 0.25}},
            BBox{{{0., 0., 0.}}, {{1., 1., 1.}}});
    std::cout << "CellSearch((0.5,0.34,0.25), (0,0,0,1,1,1)) = \n"
              << x << '\n';
  }
  {
    // A single point at the origin.
    ValueContainer v;
    v.push_back(Value{{_Float(0.), _Float(0.), _Float(0.)}});
    {
      ORQDS x(Point{{0.5, 0.34, 0.25}},
              BBox{{{0., 0., 0.}}, {{1., 1., 1.}}},
              v.begin(), v.end());
      std::cout << "Use a domain and a range of initial values.\n"
                << x << '\n';
      // CONTINUE: Check that each record is in the domain.
    }
    {
      ORQDS x(Point{{0.5, 0.34, 0.25}}, v.begin(), v.end());
      std::cout << "Use a range of initial values.\n"
                << x << '\n';
      // CONTINUE: Check that each record is in the domain.
    }
  }
  {
    ValueContainer v;
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    {
      ORQDS x(Point{{0.5, 0.34, 0.25}},
              BBox{{{0., 0., 0.}}, {{1., 1., 1.}}},
              v.begin(), v.end());
      std::cout << "Use a domain and a range of initial values.\n"
                << x << '\n';
      // CONTINUE: Check that each record is in the domain.
    }
    {
      ORQDS x(Point{{0.5, 0.34, 0.25}}, v.begin(), v.end());
      std::cout << "Use a range of initial values.\n"
                << x << '\n';
      // CONTINUE: Check that each record is in the domain.
    }
  }

  //
  // Add elements.
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
    std::cout << "Add a range." << '\n';
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
  // Accesors: size
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
    for (Record i = v.begin(); i != v.end(); ++i) {
      x.insert(i);
    }

    assert(! x.empty());
    assert(x.size() == 8);
  }
}

int
main()
{
  test<float>();
  test<double>();

  return 0;
}
