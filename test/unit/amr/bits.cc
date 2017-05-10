// -*- C++ -*-

#include "stlib/amr/bits.h"

#include <iostream>

using namespace stlib;

int
main()
{
  // This only works on 64-bit architectures.
  //static_assert(sizeof(amr::Integer<64>::Type) * 8 == 64, "Error.");
  static_assert(sizeof(amr::Integer<32>::Type) * 8 == 32, "Error.");
  static_assert(sizeof(amr::Integer<17>::Type) * 8 == 32, "Error.");
  static_assert(sizeof(amr::Integer<16>::Type) * 8 == 16, "Error.");
  static_assert(sizeof(amr::Integer<9>::Type) * 8 == 16, "Error.");
  static_assert(sizeof(amr::Integer<8>::Type) * 8 == 8, "Error.");
  static_assert(sizeof(amr::Integer<1>::Type) * 8 == 8, "Error.");

  // This only works on 64-bit architectures.
  //static_assert(sizeof(amr::UnsignedInteger<64>::Type) * 8 == 64, "Error.");
  static_assert(sizeof(amr::UnsignedInteger<32>::Type) * 8 == 32, "Error.");
  static_assert(sizeof(amr::UnsignedInteger<16>::Type) * 8 == 16, "Error.");
  static_assert(sizeof(amr::UnsignedInteger<8>::Type) * 8 == 8, "Error.");

  std::cout << 8 * sizeof(long) << "\n"
            << 8 * sizeof(int) << "\n"
            << 8 * sizeof(short) << "\n"
            << 8 * sizeof(char) << "\n";

  // 1-D.
  {
    const int Dimension = 1;
    const int Length = 8;
    typedef amr::UnsignedInteger<Length>::Type Coordinate;
    typedef amr::UnsignedInteger<Dimension* Length>::Type Code;
    std::array<Coordinate, Dimension> coordinates;
    coordinates[0] = 3;
    Code code = amr::interlaceBits<Code>(coordinates, Length);
    assert(code == coordinates[0]);
    std::array<Coordinate, Dimension> copy;
    amr::unlaceBits(code, Length, &copy);
    assert(copy == coordinates);
  }
  // 2-D.
  {
    const int Dimension = 2;
    const int Length = 8;
    typedef amr::UnsignedInteger<Length>::Type Coordinate;
    typedef amr::UnsignedInteger<Dimension* Length>::Type Code;
    std::array<Coordinate, Dimension> coordinates;
    coordinates[0] = 3;
    coordinates[1] = 5;
    Code code = amr::interlaceBits<Code>(coordinates, Length);
    std::array<Coordinate, Dimension> copy;
    amr::unlaceBits(code, Length, &copy);
    assert(copy == coordinates);
  }
  // 3-D.
  {
    const int Dimension = 3;
    const int Length = 8;
    typedef amr::UnsignedInteger<Length>::Type Coordinate;
    typedef amr::UnsignedInteger<Dimension* Length>::Type Code;
    std::array<Coordinate, Dimension> coordinates;
    coordinates[0] = 3;
    coordinates[1] = 5;
    coordinates[2] = 7;
    Code code = amr::interlaceBits<Code>(coordinates, Length);
    std::array<Coordinate, Dimension> copy;
    amr::unlaceBits(code, Length, &copy);
    assert(copy == coordinates);
  }
  // 4-D.
  {
    const int Dimension = 4;
    const int Length = 8;
    typedef amr::UnsignedInteger<Length>::Type Coordinate;
    typedef amr::UnsignedInteger<Dimension* Length>::Type Code;
    std::array<Coordinate, Dimension> coordinates;
    coordinates[0] = 3;
    coordinates[1] = 5;
    coordinates[2] = 7;
    coordinates[3] = 11;
    Code code = amr::interlaceBits<Code>(coordinates, Length);
    std::array<Coordinate, Dimension> copy;
    amr::unlaceBits(code, Length, &copy);
    assert(copy == coordinates);
    // CONTINUE: assign is broken in GCC 4.0.
    //copy.assign(Coordinate(-1));
    std::fill(copy.begin(), copy.end(), Coordinate(-1));
    amr::unlaceBits(code, Length, &copy);
    assert(copy == coordinates);
  }

  return 0;
}
