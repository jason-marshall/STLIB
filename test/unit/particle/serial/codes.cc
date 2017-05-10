// -*- C++ -*-

#include "stlib/particle/codes.h"
#include "stlib/ext/array.h"

using namespace stlib;

template<std::size_t _Dimension, bool _Periodic>
void
testMorton(std::size_t numLevels)
{
  typedef particle::IntegerTypes::Code Code;
  typedef typename particle::TemplatedTypes<double, _Dimension>::Point Point;

  const geom::BBox<double, _Dimension> Domain = {ext::filled_array<Point>(0),
                                                 ext::filled_array<Point>(1)
                                                };
  double cellLength = 1;
  for (std::size_t i = 0; i != numLevels; ++i) {
    cellLength *= 0.5;
  }
  {
    particle::Morton<double, _Dimension, _Periodic>
    morton(Domain, cellLength);
    for (Code i = 0; i != Code(1) << (_Dimension * numLevels);
         ++i) {
      assert(morton.code(morton.coordinates(i)) == i);
    }
    for (std::size_t i = 0; i <= numLevels; ++i) {
      morton.setLevels(i);
      assert(morton.numLevels() == i);
      for (std::size_t j = 0; j != _Dimension; ++j) {
        assert(morton.cellLengths()[j] == 1. / (Code(1) << i));
      }
    }
  }
  {
    particle::Morton<double, _Dimension, _Periodic> morton;
    morton.initialize(Domain, cellLength);
    for (Code i = 0; i != Code(1) << (_Dimension * numLevels);
         ++i) {
      assert(morton.code(morton.coordinates(i)) == i);
    }
  }
}

#if 0
template<std::size_t _Dimension>
void
testMortonMaxLevels()
{
  const std::size_t numLevels =
    (std::numeric_limits<std::size_t>::digits - 1) / _Dimension;
  particle::Morton<double, _Dimension> morton(numLevels);
  const std::size_t n = (std::size_t(1) << (_Dimension * numLevels)) - 1;
  assert(morton.code(morton.coordinates(n)) == n);
}
#endif

#if 0
// Use this function for valid codes.
template<std::size_t _Dimension>
bool
alignCollapse(const std::array<std::size_t, _Dimension>& a,
              const std::array<std::size_t, _Dimension>& b,
              const std::size_t level)
{
  return particle::mortonAlign(a, level) == b &&
         particle::mortonCollapse(b, level) == a;
}
#endif

int
main()
{
#if 0
  testMortonMaxLevels<1>();
  testMortonMaxLevels<2>();
  testMortonMaxLevels<3>();
#endif

  for (std::size_t i = 0; i != 6; ++i) {
    testMorton<1, false>(i);
    testMorton<2, false>(i);
    testMorton<3, false>(i);
    testMorton<1, true>(i);
    testMorton<2, true>(i);
    testMorton<3, true>(i);
  }

#if 0
  // mortonAlign

  // 1-D.
  // 0 levels.
  assert(alignCollapse(ext::make_array<std::size_t>(0x0),
                       ext::make_array<std::size_t>(0x0), 0));

  // 1-D.
  {
    const std::size_t D = 1;
    typedef std::array<std::size_t, D> Coords;
    // 0 levels.
    {
      const std::size_t level = 0;
      // Valid.
      {
        const Coords a = {{0}};
        const Coords b = {{0}};
        assert(particle::mortonAlign(a, level) == b);
        assert(particle::mortonCollapse(b, level) == a);
      }
      // Invalid.
      {
        const Coords a = {{1}};
        const Coords b = {{0}};
        assert(particle::mortonAlign(a, level) == b);
      }
    }
    // 1 levels.
    {
      const std::size_t level = 1;
      // Valid.
      {
        const Coords a = {{0}};
        const Coords b = {{0}};
        assert(particle::mortonAlign(a, level) == b);
        assert(particle::mortonCollapse(b, level) == a);
      }
      {
        const Coords a = {{1}};
        const Coords b = {{1}};
        assert(particle::mortonAlign(a, level) == b);
        assert(particle::mortonCollapse(b, level) == a);
      }
      // Invalid.
      {
        const Coords a = {{2}};
        const Coords b = {{0}};
        assert(particle::mortonAlign(a, level) == b);
      }
      {
        const Coords a = {{3}};
        const Coords b = {{1}};
        assert(particle::mortonAlign(a, level) == b);
      }
    }
    // 2 levels.
    {
      const std::size_t level = 2;
      // Valid.
      for (std::size_t i = 0; i != 1 << level; ++i) {
        const Coords a = {{i}};
        const Coords b = {{i}};
        assert(particle::mortonAlign(a, level) == b);
        assert(particle::mortonCollapse(b, level) == a);
      }
      // Invalid.
      {
        const Coords a = {{4}};
        const Coords b = {{0}};
        assert(particle::mortonAlign(a, level) == b);
      }
    }
  }


  // 2-D.
  {
    const std::size_t D = 2;
    typedef std::array<std::size_t, D> Coords;
    // 0 levels.
    {
      const std::size_t level = 0;
      // Valid.
      {
        const Coords a = {{0, 0}};
        const Coords b = {{0, 0}};
        assert(particle::mortonAlign(a, level) == b);
        assert(particle::mortonCollapse(b, level) == a);
      }
      // Invalid.
      {
        const Coords a = {{1, 1}};
        const Coords b = {{0, 0}};
        assert(particle::mortonAlign(a, level) == b);
      }
    }
    // 1 levels.
    {
      const std::size_t level = 1;
      // Valid.
      {
        const Coords a = {{0, 0}};
        const Coords b = {{0, 0}};
        assert(particle::mortonAlign(a, level) == b);
        assert(particle::mortonCollapse(b, level) == a);
      }
      {
        const Coords a = {{1, 0}};
        const Coords b = {{1, 0}};
        assert(particle::mortonAlign(a, level) == b);
        assert(particle::mortonCollapse(b, level) == a);
      }
      {
        const Coords a = {{0, 1}};
        const Coords b = {{0, 2}};
        assert(particle::mortonAlign(a, level) == b);
        assert(particle::mortonCollapse(b, level) == a);
      }
      {
        const Coords a = {{1, 1}};
        const Coords b = {{1, 2}};
        assert(particle::mortonAlign(a, level) == b);
        assert(particle::mortonCollapse(b, level) == a);
      }
    }
    // 2 levels.
    {
      const std::size_t level = 2;
      // Valid.
      {
        const Coords a = {{0, 0}};
        const Coords b = {{0, 0}};
        assert(particle::mortonAlign(a, level) == b);
        assert(particle::mortonCollapse(b, level) == a);
      }
      {
        const Coords a = {{1, 1}};
        const Coords b = {{1, 2}};
        assert(particle::mortonAlign(a, level) == b);
        assert(particle::mortonCollapse(b, level) == a);
      }
      {
        const Coords a = {{2, 2}};
        const Coords b = {{4, 8}};
        assert(particle::mortonAlign(a, level) == b);
        assert(particle::mortonCollapse(b, level) == a);
      }
      {
        const Coords a = {{3, 3}};
        const Coords b = {{5, 10}};
        assert(particle::mortonAlign(a, level) == b);
        assert(particle::mortonCollapse(b, level) == a);
      }
      // Invalid.
      {
        const Coords a = {{4, 4}};
        const Coords b = {{0, 0}};
        assert(particle::mortonAlign(a, level) == b);
      }
    }
  }


  // mortonCode
  // 2-D.
  // 2 levels.
  assert(particle::mortonCode(ext::make_array<std::size_t>(0, 0), 2) == 0);
  assert(particle::mortonCode(ext::make_array<std::size_t>(1, 1), 2) == 1 | 2);
  assert(particle::mortonCode(ext::make_array<std::size_t>(2, 2), 2) == 4 | 8);
  assert(particle::mortonCode(ext::make_array<std::size_t>(3, 3), 2) ==
         5 | 10);
  assert(particle::mortonCode(ext::make_array<std::size_t>(4, 4), 2) == 0);
#endif

  return 0;
}
