// -*- C++ -*-

#include "stlib/amr/MessageOutputStreamChecked.h"
#include "stlib/amr/MessageInputStream.h"

using namespace stlib;

class A
{
public:
  char a;
  unsigned char b;
  int c;

  bool
  operator==(const A& other)
  {
    return a == other.a && b == other.b && c == other.c;
  }
};

class B : public A
{
public:
  double d;

  bool
  operator==(const B& other)
  {
    return A::operator==(other) && d == other.d;
  }
};

int
main()
{
  //
  // Output.
  //

  amr::MessageOutputStreamChecked out;

  // Boolean.

  const bool boolean = true;
  out << boolean;
  std::size_t size = sizeof(bool);
  assert(out.getSize() == size);

  // Signed integer.

  const char character = 'a';
  out << character;
  size += sizeof(char);
  assert(out.getSize() == size);

  const short shortInteger = 2;
  out << shortInteger;
  size += sizeof(short);
  assert(out.getSize() == size);

  const int integer = 3;
  out << integer;
  size += sizeof(int);
  assert(out.getSize() == size);

  const long longInteger = 4;
  out << longInteger;
  size += sizeof(long);
  assert(out.getSize() == size);

  // Unsigned integer.

  const unsigned char unsignedCharacter = 2;
  out << unsignedCharacter;
  size += sizeof(unsigned char);
  assert(out.getSize() == size);

  const unsigned short unsignedShortInteger = 3;
  out << unsignedShortInteger;
  size += sizeof(unsigned short);
  assert(out.getSize() == size);

  const unsigned int unsignedInteger = 5;
  out << unsignedInteger;
  size += sizeof(unsigned int);
  assert(out.getSize() == size);

  const unsigned long unsignedLongInteger = 7;
  out << unsignedLongInteger;
  size += sizeof(unsigned long);
  assert(out.getSize() == size);

  // Floating point.

  const float singleFloat = 1.5;
  out << singleFloat;
  size += sizeof(float);
  assert(out.getSize() == size);

  const double doubleFloat = 3.14;
  out << doubleFloat;
  size += sizeof(double);
  assert(out.getSize() == size);

  // Classes.
  assert(sizeof(A) == sizeof(double));
  A a;
  a.a = 1;
  a.b = 2;
  a.c = 3;
  out.write(a);
  size += sizeof(A);
  assert(out.getSize() == size);

  assert(sizeof(B) == 2 * sizeof(double));
  B b;
  b.a = 1;
  b.b = 2;
  b.c = 3;
  b.d = 3.14;
  out.write(b);
  size += sizeof(B);
  assert(out.getSize() == size);

  // Array of signed integer.

  const char characterArray1[1] = {'a'};
  out.write(characterArray1, sizeof(characterArray1) / sizeof(char));
  size += sizeof(characterArray1);
  assert(out.getSize() == size);

  const char characterArray2[2] = {'a', 'b'};
  out.write(characterArray2, sizeof(characterArray2) / sizeof(char));
  size += sizeof(characterArray2);
  assert(out.getSize() == size);

  const char characterArray3[3] = {'a', 'b', 'c'};
  out.write(characterArray3, sizeof(characterArray3) / sizeof(char));
  size += sizeof(characterArray3);
  assert(out.getSize() == size);

  const char characterArray4[4] = {'a', 'b', 'c', 'd'};
  out.write(characterArray4, sizeof(characterArray4) / sizeof(char));
  size += sizeof(characterArray4);
  assert(out.getSize() == size);

  const char characterArray8[8] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
  out.write(characterArray8, sizeof(characterArray8) / sizeof(char));
  size += sizeof(characterArray8);
  assert(out.getSize() == size);

  const char characterArray9[9] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'};
  out.write(characterArray9, sizeof(characterArray9) / sizeof(char));
  size += sizeof(characterArray9);
  assert(out.getSize() == size);

  //
  // Input.
  //

  amr::MessageInputStream in(out);
  assert(in.getSize() == size);
  assert(in == out);

  // Boolean.

  {
    bool x;
    in >> x;
    assert(x == boolean);
  }

  // Signed integer.

  {
    char x;
    in >> x;
    assert(x == character);
  }
  {
    short x;
    in >> x;
    assert(x == shortInteger);
  }
  {
    int x;
    in >> x;
    assert(x == integer);
  }
  {
    long x;
    in >> x;
    assert(x == longInteger);
  }

  // Unsigned integer.

  {
    unsigned char x;
    in >> x;
    assert(x == unsignedCharacter);
  }
  {
    unsigned short x;
    in >> x;
    assert(x == unsignedShortInteger);
  }
  {
    unsigned int x;
    in >> x;
    assert(x == unsignedInteger);
  }
  {
    unsigned long x;
    in >> x;
    assert(x == unsignedLongInteger);
  }

  // Floating point.

  {
    float x;
    in >> x;
    assert(x == singleFloat);
  }
  {
    double x;
    in >> x;
    assert(x == doubleFloat);
  }

  // Classes.
  {
    A x;
    in.read(x);
    assert(x == a);
  }
  {
    B x;
    in.read(x);
    assert(x == b);
  }

  // Array of signed integer.

  {
    const int length = sizeof(characterArray1) / sizeof(char);
    char x[length];
    in.read(x, length);
    for (int n = 0; n != length; ++n) {
      assert(x[n] = characterArray1[n]);
    }
  }

  {
    const int length = sizeof(characterArray2) / sizeof(char);
    char x[length];
    in.read(x, length);
    for (int n = 0; n != length; ++n) {
      assert(x[n] = characterArray2[n]);
    }
  }

  {
    const int length = sizeof(characterArray3) / sizeof(char);
    char x[length];
    in.read(x, length);
    for (int n = 0; n != length; ++n) {
      assert(x[n] = characterArray3[n]);
    }
  }

  {
    const int length = sizeof(characterArray4) / sizeof(char);
    char x[length];
    in.read(x, length);
    for (int n = 0; n != length; ++n) {
      assert(x[n] = characterArray4[n]);
    }
  }

  {
    const int length = sizeof(characterArray8) / sizeof(char);
    char x[length];
    in.read(x, length);
    for (int n = 0; n != length; ++n) {
      assert(x[n] = characterArray8[n]);
    }
  }

  {
    const int length = sizeof(characterArray9) / sizeof(char);
    char x[length];
    in.read(x, length);
    for (int n = 0; n != length; ++n) {
      assert(x[n] = characterArray9[n]);
    }
  }

  return 0;
}
