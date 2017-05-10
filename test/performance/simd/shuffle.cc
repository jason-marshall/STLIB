// -*- C++ -*-

#include "stlib/simd/shuffle.h"
#include "stlib/ads/timer.h"

#include <iostream>


using namespace stlib;

template<typename _Float, std::size_t _D>
void
time()
{
  // Compatible with up to 6-D.
  std::vector<_Float, simd::allocator<_Float> > data((1 << 12) * 3);
  for (std::size_t i = 0; i != data.size(); ++i) {
    data[i] = i;
  }

  // AOS to hybrid SOA.
  ads::Timer timer;
  timer.tic();
  simd::aosToHybridSoa<_D>(&data);
  const double aosToHybridSoaTime = timer.toc();

  _Float result = 0;
  for (std::size_t i = 0; i != data.size(); ++i) {
    result += data[i];
  }

  // Hybrid SOA to AOS.
  timer.tic();
  simd::hybridSoaToAos<_D>(&data);
  const double hybridSoaToAosTime = timer.toc();

  for (std::size_t i = 0; i != data.size(); ++i) {
    result += data[i];
  }

  std::cout << "Meaningless result = " << result << '\n'
            << "Dimension = " << _D << '\n'
            << "Size of floating point number = " << sizeof(_Float)
            << " bytes.\n"
            << "Size of SIMD vector = " << simd::Vector<_Float>::Size << '\n'
            << "Time per element for AOS to hybrid SOA = "
            << aosToHybridSoaTime / data.size() * 1e9 << " ns.\n"
            << "Time per element for hybrid SOA to AOS = "
            << hybridSoaToAosTime / data.size() * 1e9 << " ns.\n";
}

int
main()
{
  time<float, 1>();
  time<float, 2>();
  time<float, 3>();
  time<float, 4>();
  time<double, 1>();
  time<double, 2>();
  time<double, 3>();
  time<double, 4>();

  return 0;
}
