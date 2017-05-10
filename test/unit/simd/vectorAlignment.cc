// -*- C++ -*-

#include <vector>

#include <cassert>

int
main()
{
  // Note that the allocated memory may not be 16-byte aligned.
#if 0
  {
    std::vector<float> data(4);
    assert((reinterpret_cast<std::size_t>(&data[0]) & 0xF) == 0);
    assert((reinterpret_cast<std::size_t>(&data[1]) & 0xF) == 4);
    assert((reinterpret_cast<std::size_t>(&data[2]) & 0xF) == 8);
    assert((reinterpret_cast<std::size_t>(&data[3]) & 0xF) == 12);
  }
  for (std::size_t i = 0; i != 100; ++i) {
    std::vector<float> data(i + 1);
    assert((reinterpret_cast<std::size_t>(&data[0]) & 0xF) == 0);
  }
  for (std::size_t i = 0; i != 100; ++i) {
    std::vector<char> data(i + 1);
    assert((reinterpret_cast<std::size_t>(&data[0]) & 0xF) == 0);
  }
#endif

  return 0;
}
