// -*- C++ -*-

#include "stlib/geom/orq/MortonCoordinates.h"
#include "stlib/numerical/integer/print.h"

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

template<typename _Float>
void
test()
{
  const _Float Eps = std::numeric_limits<_Float>::epsilon();
  {
    // 1-D
    typedef geom::MortonCoordinates<_Float, 1> MortonCoordinates;
    typedef typename MortonCoordinates::argument_type argument_type;
    typedef typename MortonCoordinates::result_type result_type;
    typedef geom::BBox<_Float, 1> BBox;
    const BBox domain = {{{0}}, {{1}}};
    const MortonCoordinates f(domain);
    assert(f(argument_type{{0.}}) == (result_type{{0}}));
    assert(geom::mortonCode(f(argument_type{{0.}})) == 0);
    std::cout << "1-D\n";
    const std::array<_Float, 5> values = {{ -1, 0, 0.5, 1 - Eps, 2}};
    for (std::size_t i = 0; i != values.size(); ++i) {
      std::array<_Float, 1> x = {{values[i]}};
      std::size_t code = geom::mortonCode(f(x));
      std::cout << x << ' ';
      numerical::printBits(std::cout, code);
      std::cout << '\n';
    }
    assert(f.level(1) == 0);
    assert(f.level(0.75) == 0);
    assert(f.level(0.5) == 1);
    assert(f.level(0.4) == 1);
    assert(f.level(0.25) == 2);
    assert(f.level(0) == f.Levels);
  }
  {
    // 2-D
    const std::size_t D = 2;
    typedef geom::MortonCoordinates<_Float, D> MortonCoordinates;
    typedef typename MortonCoordinates::argument_type argument_type;
    typedef typename MortonCoordinates::result_type result_type;
    typedef geom::BBox<_Float, D> BBox;
    const BBox domain = {{{0, 0}}, {{1, 1}}};
    const MortonCoordinates f(domain);
    assert(f(argument_type{{0, 0}}) == (result_type{{0, 0}}));
    assert(geom::mortonCode(f(argument_type{{0, 0}})) == 0);
    std::cout << "2-D\n";
    const std::array<std::array<_Float, D>, 5> values =
    {{{{ -1, -1}}, {{0, 0}}, {{0.5, 0.5}}, {{1 - Eps, 1 - Eps}}, {{2, 2}}}};
    for (std::size_t i = 0; i != values.size(); ++i) {
      for (std::size_t level = 0; level != 3; ++level) {
        std::size_t code = geom::mortonCode(f(values[i]), level);
        std::cout << values[i] << ' ' << level << ' ';
        numerical::printBits(std::cout, code);
        std::cout << '\n';
      }
      std::size_t code = geom::mortonCode(f(values[i]));
      std::cout << values[i] << ' ';
      numerical::printBits(std::cout, code);
      std::cout << '\n';
    }
  }
}

int
main()
{
  std::cout << "float\n";
  test<float>();
  std::cout << "double\n";
  test<double>();

  return 0;
}
