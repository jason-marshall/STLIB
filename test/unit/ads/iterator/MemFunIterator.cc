// -*- C++ -*-

#include "stlib/ads/iterator/MemFunIterator.h"

#include "stlib/ads/array/FixedArray.h"

#include <iostream>
#include <vector>
#include <functional>

#include <cassert>


using namespace stlib;

class C
{
public:

  int
  size()
  {
    return 1;
  }

  int
  const_size() const
  {
    return 1;
  }
};

int
main()
{
  using namespace ads;

  typedef ads::FixedArray<1> FA;
  typedef MemFunIterator<FA*, FA, int> MFI;
  {
    std::mem_fun_ref_t<int, C> x(&C::size);
    C p;
    std::cout << x(p) << '\n';
  }
  {
    std::const_mem_fun_ref_t<int, FA> x(&FA::min_index);
    FA p;
    std::cout << x(p) << '\n';
  }
  {
    MFI x;
    FA fa;
    x = &FA::min_index;
    x = &fa;
    FA* p = x;
    assert(p->min_index() == *x);
  }
  {
    MemFunIterator<C*, C, int, false> x(&C::size);
  }
  {
    C c;
    MemFunIterator<C*, C, int, false> x(&C::size, &c);
    assert(*x == c.size());
  }
  {
    C c;
    MemFunIterator<C*, C, int, false> x(&C::size);
    x = &c;
    assert(*x == c.size());
  }
  {
    MemFunIterator<const C*, C, int, true> x(&C::const_size);
  }
  /*
  {
    MFI x(&FA::size);
  }
  */
  return 0;
}
