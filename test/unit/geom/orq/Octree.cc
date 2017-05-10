// -*- C++ -*-

#include "stlib/geom/orq/Octree.h"

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
  typedef geom::Octree<ads::Dereference<Record>,
          std::back_insert_iterator<RecordContainer> > ORQDS;
  typedef typename ORQDS::BBox BBox;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  //
  // Constructors
  //

  {
    // Construct from a Cartesian domain.
    BBox unitCube = {{{0., 0., 0.}}, {{1., 1., 1.}}};
    ORQDS oct(unitCube);
    std::cout << "ORQDS(BBox{0,0,0,1,1,1}) = "
              << '\n'
              << oct
              << '\n';
    BBox domain = {{{0., 0., 0.}}, {{1., 1., 1.}}};
    ORQDS emptyOctree(domain);
    assert(emptyOctree.isValid());

    ValueContainer v;
    for (std::size_t i = 0; i != 16; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    ORQDS octr(unitCube, v.begin(), v.end());
    std::cout << "Use a range of initial values."
              << '\n'
              << octr
              << '\n';

    ORQDS octree(domain, v.begin(), v.end());
    octree.print(std::cout);
    std::cout << "Memory usage = " << octree.getMemoryUsage() << '\n';
    assert(octree.isValid());
  }
  {
    // Records that cannot be separated.
    BBox unitCube = {{{0., 0., 0.}}, {{1., 1., 1.}}};
    ValueContainer v;
    for (std::size_t i = 0; i != 8; ++i) {
      v.push_back(Value{{_Float(0), _Float(0), _Float(0)}});
    }
    // If we don't exceed the leaf size, we can store them.
    ORQDS oct(unitCube, v.begin(), v.end());

    // Now exceed the leaf size.
    v.push_back(Value{{_Float(0), _Float(0), _Float(0)}});
    bool caught = false;
    try {
      // Make sure that this throws a runtime_error.
      ORQDS oct(unitCube, v.begin(), v.end());
    }
    catch (const std::runtime_error& error) {
      caught = true;
    }
    assert(caught);
  }

  //
  // Add grid elements.
  //

  {
    std::cout << "Add one at a time." << '\n';
    BBox domain = {{{0., 0., 0.}}, {{1., 1., 1.}}};
    ORQDS octree(domain);
    ValueContainer v;
    for (std::size_t i = 0; i != 16; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    for (Record i = v.begin(); i != v.end(); ++i) {
      octree.insert(i);
    }
    std::cout << octree << '\n';
  }
  {
    std::cout << "Add a range." << '\n';
    BBox domain = {{{0., 0., 0.}}, {{1., 1., 1.}}};
    ORQDS octree(domain);
    ValueContainer v;
    for (std::size_t i = 0; i != 16; ++i) {
      v.push_back(Value{{_Float(random()), _Float(random()),
              _Float(random())}});
    }
    octree.insert(v.begin(), v.end());
    std::cout << octree << '\n';
  }

  //
  // Accesors: grid size
  //

  {
    BBox domain = {{{1., 2., 3.}}, {{2., 3., 5.}}};
    ORQDS octree(domain);
    assert(octree.empty());

    ValueContainer v;
    for (std::size_t i = 0; i != 100; ++i) {
      v.push_back(Value{{_Float(1 + random()), _Float(2 + random()),
              _Float(3 + 2 * random())}});
    }
    for (Record i = v.begin(); i != v.end(); ++i) {
      octree.insert(i);
    }

    assert(octree.isValid());
    assert(! octree.empty());
    assert(octree.size() == 100);
  }

  //
  // Accesors: Cartesian box
  //

  {
    BBox domain = {{{1., 2., 3.}}, {{2., 3., 5.}}};
    ORQDS octree(domain);
    assert(octree.getDomain() ==
           (BBox{{{1., 2., 3.}}, {{2., 3., 5.}}}));
  }

  //
  // Mathematical member functions
  //

  {
    BBox domain = {{{1., 2., 3.}}, {{2.0001, 3.0001, 5.0001}}};
    ORQDS octree(domain);
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
    octree.insert(v.begin(), v.end());

    RecordContainer vec;

    assert(octree.computeWindowQuery(back_inserter(vec),
                                     BBox{{{1., 2., 3.}}, {{2., 3., 5.}}}) ==
           24);
    assert(vec.size() == 24);
    vec.clear();

    assert(octree.computeWindowQuery(back_inserter(vec),
                                     BBox{{{1., 2., 3.}}, {{2., 3., 3.}}}) ==
           6);
    assert(vec.size() == 6);
    vec.clear();

    assert(octree.computeWindowQuery(back_inserter(vec),
                                     BBox{{{1., 2., 3.}}, {{1., 2., 3.}}}) ==
           1);
    assert(vec.size() == 1);
    vec.clear();

    assert(octree.computeWindowQuery(back_inserter(vec),
                                     BBox{{{1., 2., 0.}}, {{2., 3., 1.}}}) ==
           0);
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
