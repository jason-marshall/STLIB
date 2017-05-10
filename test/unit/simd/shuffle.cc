// -*- C++ -*-

#include "stlib/simd/simd.h"

#include <vector>

#include <cassert>


using namespace stlib;

template<typename _Float, std::size_t _D>
inline
void
test()
{
  const std::size_t VectorSize = simd::Vector<_Float>::Size;
  {
    std::vector<_Float, simd::allocator<_Float> > data(_D * VectorSize);
    // 0 1 2 3 ...
    for (std::size_t i = 0; i != data.size(); ++i) {
      data[i] = i;
    }
    // 0 _D 2*_D ...
    simd::aosToHybridSoa<_D>(&data);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      for (std::size_t j = 0; j != _D; ++j) {
        assert(data[i + VectorSize * j] == _D * i + j);
      }
    }
    // Back to AOS.
    simd::hybridSoaToAos<_D>(&data);
    for (std::size_t i = 0; i != data.size(); ++i) {
      assert(data[i] == i);
    }
  }
  {
    // No padding necessary.
    std::vector<std::array<_Float, _D> > input(VectorSize);
    // 0 1 2 3 ...
    for (std::size_t i = 0; i != input.size(); ++i) {
      for (std::size_t j = 0; j != _D; ++j) {
        input[i][j] = _D * i + j;
      }
    }
    // 0 _D 2*_D ...
    std::vector<_Float, simd::allocator<_Float> > data;
    simd::aosToHybridSoa(input, &data);
    assert(data.size() == input.size() * _D);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      for (std::size_t j = 0; j != _D; ++j) {
        assert(data[i + VectorSize * j] == _D * i + j);
      }
    }
    simd::aosToHybridSoa<_D>(input.begin(), input.end(), &data);
    assert(data.size() == input.size() * _D);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      for (std::size_t j = 0; j != _D; ++j) {
        assert(data[i + VectorSize * j] == _D * i + j);
      }
    }
  }
  {
    // Empty input.
    std::vector<std::array<_Float, _D> > input;
    std::vector<_Float, simd::allocator<_Float> > data;
    simd::aosToHybridSoa(input, &data);
    assert(data.empty());
    simd::aosToHybridSoa<_D>(input.begin(), input.end(), &data);
    assert(data.empty());
  }
  {
    // Padding is necessary.
    std::vector<std::array<_Float, _D> > input(1);
    // 0 1 2 3 ...
    for (std::size_t i = 0; i != input.size(); ++i) {
      for (std::size_t j = 0; j != _D; ++j) {
        input[i][j] = _D * i + j;
      }
    }
    // 0 _D 2*_D ...
    std::vector<_Float, simd::allocator<_Float> > data;
    simd::aosToHybridSoa(input, &data);
    assert(data.size() == VectorSize * _D);
    for (std::size_t j = 0; j != _D; ++j) {
      assert(data[VectorSize * j] == j);
    }
    for (std::size_t i = 1; i != VectorSize; ++i) {
      for (std::size_t j = 0; j != _D; ++j) {
        assert(data[i + VectorSize * j] != data[i + VectorSize * j]);
      }
    }
    simd::aosToHybridSoa<_D>(input.begin(), input.end(), &data);
    assert(data.size() == VectorSize * _D);
    for (std::size_t j = 0; j != _D; ++j) {
      assert(data[VectorSize * j] == j);
    }
    for (std::size_t i = 1; i != VectorSize; ++i) {
      for (std::size_t j = 0; j != _D; ++j) {
        assert(data[i + VectorSize * j] != data[i + VectorSize * j]);
      }
    }
  }
}


int
main()
{
  test<float, 1>();
  test<float, 2>();
  test<float, 3>();
  test<float, 4>();
  test<double, 1>();
  test<double, 2>();
  test<double, 3>();
  test<double, 4>();



  //-------------------------------------------------------------------------
  // CONTINUE: Old interface.
  // A single block of 12.
  {
    std::vector<float, simd::allocator<float, 16> > data(12);
    // 0 1 2 3 4 5 6 7 8 9 10 11
    for (std::size_t i = 0; i != data.size(); ++i) {
      data[i] = i;
    }
    // 0 3 6 9 1 4 7 10 2 5 8 11
    simd::aos4x3ToSoa3x4(&data[0]);
    for (std::size_t i = 0; i != 4; ++i) {
      for (std::size_t j = 0; j != 3; ++j) {
        assert(data[i + 4 * j] == 3 * i + j);
      }
    }
  }

  // Multiple blocks of 12.
  {
    const std::size_t NumBlocks = 3;
    const std::size_t BlockSize = 12;
    std::vector<float, simd::allocator<float, 16> > data(NumBlocks * BlockSize);
    for (std::size_t i = 0; i != NumBlocks; ++i) {
      for (std::size_t j = 0; j != BlockSize; ++j) {
        data[i * BlockSize + j] = j;
      }
    }
    simd::aos4x3ToSoa3x4(&data[0], data.size() / 3);
    for (std::size_t i = 0; i != NumBlocks; ++i) {
      for (std::size_t j = 0; j != 4; ++j) {
        for (std::size_t k = 0; k != 3; ++k) {
          assert(data[j + 4 * k] == 3 * j + k);
        }
      }
    }
  }

  // std::vector interface.
  {
    const std::size_t NumBlocks = 3;
    const std::size_t BlockSize = 12;
    std::vector<float, simd::allocator<float, 16> > data(NumBlocks * BlockSize);
    for (std::size_t i = 0; i != NumBlocks; ++i) {
      for (std::size_t j = 0; j != BlockSize; ++j) {
        data[i * BlockSize + j] = j;
      }
    }
    simd::aos4x3ToSoa3x4(&data);
    for (std::size_t i = 0; i != NumBlocks; ++i) {
      for (std::size_t j = 0; j != 4; ++j) {
        for (std::size_t k = 0; k != 3; ++k) {
          assert(data[j + 4 * k] == 3 * j + k);
        }
      }
    }
  }

  // 3D point interface.
  {
    const std::size_t NumBlocks = 3;
    const std::size_t BlockSize = 12;
    std::vector<std::array<float, 3> > input(BlockSize);
    {
      float* data = &input[0][0];
      for (std::size_t i = 0; i != NumBlocks; ++i) {
        for (std::size_t j = 0; j != BlockSize; ++j) {
          data[i * BlockSize + j] = j;
        }
      }
    }
    std::vector<float, simd::allocator<float, 16> > shuffled;
    simd::aos4x3ToSoa3x4(input, &shuffled);
    for (std::size_t i = 0; i != NumBlocks; ++i) {
      for (std::size_t j = 0; j != 4; ++j) {
        for (std::size_t k = 0; k != 3; ++k) {
          assert(shuffled[j + 4 * k] == 3 * j + k);
        }
      }
    }
  }

  return 0;
}
