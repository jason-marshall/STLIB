// -*- C++ -*-

#include "stlib/geom/orq/SpatialIndexMortonUniform.h"

#include <iostream>

using namespace stlib;

template<typename _Float>
void
test()
{
  {
    // 1-D
    typedef geom::SpatialIndexMortonUniform<_Float, 1> SpatialIndex;
    typedef typename SpatialIndex::argument_type argument_type;
    typedef geom::BBox<_Float, 1> BBox;
    // Zero levels of refinement.
    {
      const BBox domain = {{{0}}, {{1}}};
      const SpatialIndex f(domain, 1);
      assert(f(argument_type{{0.}}) == 0);
      assert(f(argument_type{{-1.}}) == 0);
      assert(f(argument_type{{10.}}) == 0);
    }
    // One level of refinement.
    {
      const BBox domain = {{{0}}, {{1}}};
      const SpatialIndex f(domain, 0.5);
      assert(f(argument_type{{0.}}) == 0);
      assert(f(argument_type{{-1.}}) == 0);
      assert(f(argument_type{{0.5}}) == 1);
      assert(f(argument_type{{10.}}) == 1);
    }
    {
      const BBox domain = {{{1}}, {{2}}};
      const SpatialIndex f(domain, 0.5);
      assert(f(argument_type{{1.}}) == 0);
      assert(f(argument_type{{0.}}) == 0);
      assert(f(argument_type{{1.5}}) == 1);
      assert(f(argument_type{{10.}}) == 1);
    }
    // Two levels of refinement.
    {
      const BBox domain = {{{0}}, {{1}}};
      const SpatialIndex f(domain, 0.25);
      assert(f(argument_type{{-1.}}) == 0);
      assert(f(argument_type{{0.}}) == 0);
      assert(f(argument_type{{0.25}}) == 1);
      assert(f(argument_type{{0.51}}) == 2);
      assert(f(argument_type{{0.751}}) == 3);
      assert(f(argument_type{{10.}}) == 3);
    }
  }
  {
    // 2-D
    typedef geom::SpatialIndexMortonUniform<_Float, 2> SpatialIndex;
    typedef typename SpatialIndex::argument_type argument_type;
    typedef geom::BBox<_Float, 2> BBox;
    // Zero levels of refinement.
    {
      const BBox domain = {{{0, 0}}, {{1, 1}}};
      const SpatialIndex f(domain, 1);
      assert(f(argument_type{{0., 0.}}) == 0);
      assert(f(argument_type{{-1., -1.}}) == 0);
      assert(f(argument_type{{10., 10.}}) == 0);
    }
    {
      const BBox domain = {{{0, 0}}, {{1, 0}}};
      const SpatialIndex f(domain, 1);
      assert(f(argument_type{{0., 0.}}) == 0);
      assert(f(argument_type{{-1., -1.}}) == 0);
      assert(f(argument_type{{10., 10.}}) == 0);
    }
    // One level of refinement.
    {
      const BBox domain = {{{0, 0}}, {{1, 1}}};
      const SpatialIndex f(domain, 0.5);
      assert(f(argument_type{{0., 0.}}) == 0);
      assert(f(argument_type{{0.5, 0.}}) == 1);
      assert(f(argument_type{{0., 0.5}}) == 2);
      assert(f(argument_type{{0.5, 0.5}}) == 3);
      assert(f(argument_type{{-1., -1.}}) == 0);
      assert(f(argument_type{{10., -10.}}) == 1);
      assert(f(argument_type{{-10., 0.5}}) == 2);
      assert(f(argument_type{{10., 10.}}) == 3);
    }
    {
      const BBox domain = {{{0, 0}}, {{0, 1}}};
      const SpatialIndex f(domain, 0.5);
      assert(f(argument_type{{0., 0.}}) == 0);
      assert(f(argument_type{{0.5, 0.}}) == 1);
      assert(f(argument_type{{0., 0.5}}) == 2);
      assert(f(argument_type{{0.5, 0.5}}) == 3);
    }
    // Two levels of refinement.
    {
      const BBox domain = {{{0, 0}}, {{1, 1}}};
      const SpatialIndex f(domain, 0.25);
      assert(f(argument_type{{0., 0.}}) == 0);
      assert(f(argument_type{{0.25, 0.}}) == 1);
      assert(f(argument_type{{0., 0.25}}) == 2);
      assert(f(argument_type{{0.25, 0.25}}) == 3);
      assert(f(argument_type{{0.5, 0.}}) == 4);
      assert(f(argument_type{{0.75, 0.}}) == 5);
      assert(f(argument_type{{0.5, 0.25}}) == 6);
      assert(f(argument_type{{0.75, 0.25}}) == 7);
      assert(f(argument_type{{0., 0.5}}) == 8);
      assert(f(argument_type{{0.25, 0.5}}) == 9);
      assert(f(argument_type{{0., 0.75}}) == 10);
      assert(f(argument_type{{0.25, 0.75}}) == 11);
      assert(f(argument_type{{0.5, 0.5}}) == 12);
      assert(f(argument_type{{0.75, 0.5}}) == 13);
      assert(f(argument_type{{0.5, 0.75}}) == 14);
      assert(f(argument_type{{0.75, 0.75}}) == 15);
    }
  }
}

void
jeffsTest()
{
  const float magicNumber = -5.414999;
  const geom::BBox<float, 3> domain = {
    {{magicNumber, magicNumber, magicNumber}},
    {{ -magicNumber, -magicNumber, -magicNumber}}
  };
  float magicNumber2 = 1;
  geom::SpatialIndexMortonUniform<float, 3> f(domain,
      double(magicNumber2));


  std::array<float, 3> point = {{ -5.414999,  -3.609999,  -2.38e-7}};
  std::cout << f(point) << '\n';
}

int
main()
{
  test<float>();
  test<double>();
  jeffsTest();

  return 0;
}
