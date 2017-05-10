// -*- C++ -*-

#define STLIB_PERFORMANCE

#include "stlib/container/VectorFixedCapacity.h"
#include "stlib/performance/PerformanceSerial.h"

int
main()
{
  using stlib::performance::Scope;
  using stlib::performance::record;

  constexpr std::size_t Capacity = 8;
  constexpr std::size_t Size = 10000000;
  
  {
    Scope _("push_back() and pop_back() with std::vector");
    std::vector<std::size_t> x;
    std::size_t result = 0;
    for (std::size_t n = 0; n != Size; ) {
      for (std::size_t i = 0; i != Capacity && n != Size; ++i, ++n) {
        x.push_back(i);
      }
      result += x.back();
      for (std::size_t i = 0; i != Capacity && n != Size; ++i, ++n) {
        x.pop_back();
      }
      result += x.size();
    }
    record("Meaningless result", result);
  }

  {
    Scope _("push_back() and pop_back() with VectorFixedCapacity");
    stlib::container::VectorFixedCapacity<std::size_t, Capacity> x;
    std::size_t result = 0;
    for (std::size_t n = 0; n != Size; ) {
      for (std::size_t i = 0; i != Capacity && n != Size; ++i, ++n) {
        x.push_back(i);
      }
      result += x.back();
      for (std::size_t i = 0; i != Capacity && n != Size; ++i, ++n) {
        x.pop_back();
      }
      result += x.size();
    }
    record("Meaningless result", result);
  }

  stlib::performance::print(std::cout);

  return 0;
}
