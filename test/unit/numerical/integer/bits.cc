// -*- C++ -*-

#include "stlib/numerical/integer/bits.h"

#include <algorithm>
#include <iostream>

using namespace stlib;


template<typename _UnsignedInteger, typename _Index>
void
indicesToBitArrayToIndices(std::vector<_Index> const& input)
{
  assert(! input.empty());
  std::vector<_Index> result = input;
  std::sort(result.begin(), result.end());
  _Index const size = result.back() + 1;
  std::vector<_UnsignedInteger> bitArray =
    numerical::convertIndicesToBitArray<_UnsignedInteger>(size, input);
  std::vector<_Index> output =
    numerical::convertBitArrayToIndices<_Index>(bitArray);
  assert(output == result);
}


template<typename _Index, typename _UnsignedInteger>
void
indicesAndBitArray()
{
  {
    std::vector<_Index> indices;
    std::vector<_UnsignedInteger> bitArray =
      numerical::convertIndicesToBitArray<_UnsignedInteger>(0, indices);
    assert(bitArray.empty());
  }
  {
    std::vector<_UnsignedInteger> bitArray;
    std::vector<_Index> indices =
      numerical::convertBitArrayToIndices<_Index>(bitArray);
    assert(indices.empty());
  }
  indicesToBitArrayToIndices<_UnsignedInteger>(std::vector<_Index>{0});
  indicesToBitArrayToIndices<_UnsignedInteger>(std::vector<_Index>{127});
  indicesToBitArrayToIndices<_UnsignedInteger>
    (std::vector<_Index>{0, 2, 3, 42, 511});
  indicesToBitArrayToIndices<_UnsignedInteger>
    (std::vector<_Index>{511, 42, 3, 2, 0});
}


template<typename _Index>
void
indicesToBitVectorToIndices(std::vector<_Index> const& input)
{
  assert(! input.empty());
  std::vector<_Index> result = input;
  std::sort(result.begin(), result.end());
  _Index const size = result.back() + 1;
  std::vector<bool> bitArray =
    numerical::convertIndicesToBitVector(size, input);
  std::vector<_Index> output =
    numerical::convertBitVectorToIndices<_Index>(bitArray);
  assert(output == result);
}


template<typename _Index>
void
indicesAndBitVector()
{
  {
    std::vector<_Index> indices;
    std::vector<bool> bitArray =
      numerical::convertIndicesToBitVector(0, indices);
    assert(bitArray.empty());
  }
  {
    std::vector<bool> bitArray;
    std::vector<_Index> indices =
      numerical::convertBitVectorToIndices<_Index>(bitArray);
    assert(indices.empty());
  }
  indicesToBitVectorToIndices(std::vector<_Index>{0});
  indicesToBitVectorToIndices(std::vector<_Index>{127});
  indicesToBitVectorToIndices(std::vector<_Index>{0, 2, 3, 42, 511});
  indicesToBitVectorToIndices(std::vector<_Index>{511, 42, 3, 2, 0});
}


int
main()
{
  // This only works on 64-bit architectures.
  //static_assert(sizeof(numerical::Integer<64>::Type) * 8 == 64, "Error.");
  static_assert(sizeof(numerical::Integer<32>::Type) * 8 == 32, "Error.");
  static_assert(sizeof(numerical::Integer<17>::Type) * 8 == 32, "Error.");
  static_assert(sizeof(numerical::Integer<16>::Type) * 8 == 16, "Error.");
  static_assert(sizeof(numerical::Integer<9>::Type) * 8 == 16, "Error.");
  static_assert(sizeof(numerical::Integer<8>::Type) * 8 == 8, "Error.");
  static_assert(sizeof(numerical::Integer<1>::Type) * 8 == 8, "Error.");

  // This only works on 64-bit architectures.
  //static_assert(sizeof(numerical::UnsignedInteger<64>::Type) * 8 == 64, "Error.");
  static_assert(sizeof(numerical::UnsignedInteger<32>::Type) * 8 == 32,
                    "Error.");
  static_assert(sizeof(numerical::UnsignedInteger<16>::Type) * 8 == 16,
                    "Error.");
  static_assert(sizeof(numerical::UnsignedInteger<8>::Type) * 8 == 8, "Error.");

  std::cout << 8 * sizeof(long) << "\n"
            << 8 * sizeof(int) << "\n"
            << 8 * sizeof(short) << "\n"
            << 8 * sizeof(char) << "\n";

  // Population count.
  using numerical::popCount;
  //assert(popCount(char(0)) == 0);
  assert(popCount((unsigned char)(0)) == 0);
  //assert(popCount(short(0)) == 0);
  assert(popCount((unsigned short)(0)) == 0);
  //assert(popCount(int(0)) == 0);
  assert(popCount((unsigned int)(0)) == 0);
  assert(popCount((unsigned long)(0)) == 0);
  assert(popCount((unsigned long long)(0)) == 0);

  assert(popCount((unsigned char)(0xFF)) == 8);
  assert(popCount((unsigned short)(0xFFFF)) == 16);
  assert(popCount((unsigned int)(0xFFFFFFFF)) == 32);

  assert(popCount(std::uint8_t(0)) == 0);
  assert(popCount(std::uint16_t(0)) == 0);
  assert(popCount(std::uint32_t(0)) == 0);
  assert(popCount(std::uint64_t(0)) == 0);
  
  assert(popCount(std::numeric_limits<std::size_t>::max()) ==
         8 * sizeof(std::size_t));

  // Highest bit set.
  assert(numerical::highestBitPosition(0x1) == 0);
  assert(numerical::highestBitPosition(0x2) == 1);
  assert(numerical::highestBitPosition(std::numeric_limits
                                       <unsigned char>::max()) ==
         std::numeric_limits<unsigned char>::digits - 1);
  assert(numerical::highestBitPosition(std::numeric_limits
                                       <unsigned short>::max()) ==
         std::numeric_limits<unsigned short>::digits - 1);
  assert(numerical::highestBitPosition(std::numeric_limits
                                       <unsigned int>::max()) ==
         std::numeric_limits<unsigned int>::digits - 1);
  assert(numerical::highestBitPosition(std::numeric_limits
                                       <unsigned long>::max()) ==
         std::numeric_limits<unsigned long>::digits - 1);

  // 1-D.
  {
    const std::size_t Dimension = 1;
    const std::size_t Length = 8;
    typedef numerical::Integer<Length>::Type Coordinate;
    typedef numerical::Integer<Dimension* Length>::Type Code;
    std::array<Coordinate, Dimension> coordinates;
    coordinates[0] = 3;
    Code code = numerical::interlaceBits<Code>(coordinates, Length);
    assert(code == coordinates[0]);
    std::array<Coordinate, Dimension> copy;
    numerical::unlaceBits(code, Length, &copy);
    assert(copy == coordinates);
  }
  // 2-D.
  {
    const std::size_t Dimension = 2;
    const std::size_t Length = 8;
    typedef numerical::Integer<Length>::Type Coordinate;
    typedef numerical::Integer<Dimension* Length>::Type Code;
    std::array<Coordinate, Dimension> coordinates;
    coordinates[0] = 3;
    coordinates[1] = 5;
    Code code = numerical::interlaceBits<Code>(coordinates, Length);
    std::array<Coordinate, Dimension> copy;
    numerical::unlaceBits(code, Length, &copy);
    assert(copy == coordinates);
  }
  // 3-D.
  {
    const std::size_t Dimension = 3;
    const std::size_t Length = 8;
    typedef numerical::Integer<Length>::Type Coordinate;
    typedef numerical::Integer<Dimension* Length>::Type Code;
    std::array<Coordinate, Dimension> coordinates;
    coordinates[0] = 3;
    coordinates[1] = 5;
    coordinates[2] = 7;
    Code code = numerical::interlaceBits<Code>(coordinates, Length);
    std::array<Coordinate, Dimension> copy;
    numerical::unlaceBits(code, Length, &copy);
    assert(copy == coordinates);
  }
  // 4-D.
  {
    const std::size_t Dimension = 4;
    const std::size_t Length = 8;
    typedef numerical::Integer<Length>::Type Coordinate;
    typedef numerical::Integer<Dimension* Length>::Type Code;
    std::array<Coordinate, Dimension> coordinates;
    coordinates[0] = 3;
    coordinates[1] = 5;
    coordinates[2] = 7;
    coordinates[3] = 11;
    Code code = numerical::interlaceBits<Code>(coordinates, Length);
    std::array<Coordinate, Dimension> copy;
    numerical::unlaceBits(code, Length, &copy);
    assert(copy == coordinates);
    std::fill(copy.begin(), copy.end(), -1);
    numerical::unlaceBits(code, Length, &copy);
    assert(copy == coordinates);
  }

  // Convert between indices and a bit array.
  indicesAndBitArray<int, unsigned char>();
  indicesAndBitArray<std::size_t, unsigned char>();
  indicesAndBitArray<int, std::size_t>();
  indicesAndBitArray<std::size_t, std::size_t>();
  
  // Convert between indices and a bit vector.
  indicesAndBitVector<int>();
  indicesAndBitVector<std::size_t>();
  
  return 0;
}
