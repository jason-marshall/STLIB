// -*- C++ -*-

#include "stlib/geom/kernel/Simplex.h"

#include <cassert>


template<typename _Float, std::size_t _D, std::size_t _K>
inline
void
test()
{
  // CONTINUE Sizes are different for zero-element arrays on a mac.
#if 0
  static_assert(sizeof(stlib::geom::Simplex<_Float, _D, _K>) ==
                (_K + 1) * (_D == 0 ? 1 : _D * sizeof(_Float)), "Size check.");
#else
  static_assert(sizeof(stlib::geom::Simplex<_Float, _D, _K>) <=
                (_K + 1) * (_D == 0 ? 1 : _D) * sizeof(_Float), "Size check.");
#endif
}

template<typename _Float, std::size_t _D>
inline
void
testK()
{
  test<_Float, _D, 0>();
  test<_Float, _D, 1>();
  test<_Float, _D, 2>();
  test<_Float, _D, 3>();
  test<_Float, _D, 4>();
}

template<typename _Float>
inline
void
testD()
{
  testK<_Float, 0>();
  testK<_Float, 1>();
  testK<_Float, 2>();
  testK<_Float, 3>();
  testK<_Float, 4>();
}

int
main()
{
  testD<float>();
  testD<double>();

  return 0;
}
