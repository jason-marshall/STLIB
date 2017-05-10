// -*- C++ -*-

#include "stlib/ext/vector.h"

#include <sstream>
#include <limits>

int
main()
{
  USING_STLIB_EXT_VECTOR_MATH_OPERATORS;
  
  //----------------------------------------------------------------------------
  // Vector Assignment Operators with a Scalar Operand.
  {
    std::vector<int> a;
    a.push_back(2);
    a.push_back(3);
    a.push_back(5);
    // +=
    {
      std::vector<int> b = a;
      b += 1;
      for (std::size_t n = 0; n != b.size(); ++n) {
        assert(b[n] == a[n] + 1);
      }
    }
    // +=
    {
      std::vector<int> b = a;
      b += char(1);
      for (std::size_t n = 0; n != b.size(); ++n) {
        assert(b[n] == a[n] + char(1));
      }
    }
    // -=
    {
      std::vector<int> b = a;
      b -= 1;
      for (std::size_t n = 0; n != b.size(); ++n) {
        assert(b[n] == a[n] - 1);
      }
    }
    // *=
    {
      std::vector<int> b = a;
      b *= 2;
      for (std::size_t n = 0; n != b.size(); ++n) {
        assert(b[n] == a[n] * 2);
      }
    }
    // /=
    {
      std::vector<int> b = a;
      b /= 2;
      for (std::size_t n = 0; n != b.size(); ++n) {
        assert(b[n] == a[n] / 2);
      }
    }
    // %=
    {
      std::vector<int> b = a;
      b %= 2;
      for (std::size_t n = 0; n != b.size(); ++n) {
        assert(b[n] == a[n] % 2);
      }
    }
    // <<=
    {
      std::vector<int> b = a;
      b <<= 1;
      for (std::size_t n = 0; n != b.size(); ++n) {
        assert(b[n] == a[n] << 1);
      }
    }
    // >>=
    {
      std::vector<int> b = a;
      b >>= 1;
      for (std::size_t n = 0; n != b.size(); ++n) {
        assert(b[n] == a[n] >> 1);
      }
    }
  }

  //---------------------------------------------------------------------------
  // Vector Assignment Operators with a Vector Operand.
  {
    std::vector<int> a;
    a.push_back(2);
    a.push_back(3);
    a.push_back(5);
    std::vector<int> b;
    b.push_back(1);
    b.push_back(2);
    b.push_back(3);
    // +=
    {
      std::vector<int> c = a;
      c += b;
      for (std::size_t n = 0; n != c.size(); ++n) {
        assert(c[n] == a[n] + b[n]);
      }
    }
    // -=
    {
      std::vector<int> c = a;
      c -= b;
      for (std::size_t n = 0; n != c.size(); ++n) {
        assert(c[n] == a[n] - b[n]);
      }
    }
    // *=
    {
      std::vector<int> c = a;
      c *= b;
      for (std::size_t n = 0; n != c.size(); ++n) {
        assert(c[n] == a[n] * b[n]);
      }
    }
    // /=
    {
      std::vector<int> c = a;
      c /= b;
      for (std::size_t n = 0; n != c.size(); ++n) {
        assert(c[n] == a[n] / b[n]);
      }
    }
    // %=
    {
      std::vector<int> c = a;
      c %= b;
      for (std::size_t n = 0; n != c.size(); ++n) {
        assert(c[n] == a[n] % b[n]);
      }
    }
    // <<=
    {
      std::vector<int> c = a;
      c <<= b;
      for (std::size_t n = 0; n != c.size(); ++n) {
        assert(c[n] == a[n] << b[n]);
      }
    }
    // >>=
    {
      std::vector<int> c = a;
      c >>= b;
      for (std::size_t n = 0; n != c.size(); ++n) {
        assert(c[n] == a[n] >> b[n]);
      }
    }
  }

  //---------------------------------------------------------------------------
  // File I/O.
  {
    USING_STLIB_EXT_VECTOR_IO_OPERATORS;
    {
      std::vector<int> a;
      a.push_back(2);
      a.push_back(3);
      a.push_back(5);
      std::cout << a << '\n';
      std::ostringstream out;
      out << a;
      std::vector<int> b;
      std::istringstream in(out.str());
      in >> b;
      assert(a == b);
    }
    {
      std::vector<int> a;
      a.push_back(2);
      a.push_back(3);
      a.push_back(5);
      std::ostringstream out;
      stlib::ext::writeElements(out, a);
      std::vector<int> b;
      std::istringstream in(out.str());
      stlib::ext::readElements(in, &b);
      assert(a == b);
    }
    {
      std::vector<double> a = {2., 3., 5., std::numeric_limits<double>::max(),
                               std::numeric_limits<double>::infinity()};
      std::cout << a << '\n';
      std::ostringstream out;
      stlib::ext::write(out, a);
      std::vector<double> b;
      std::istringstream in(out.str());
      stlib::ext::read(in, &b);
      assert(a == b);
    }
    {
      std::vector<int> a;
      a.push_back(2);
      a.push_back(3);
      a.push_back(5);
      std::vector<unsigned char> buffer(stlib::ext::serializedSize(a));
      stlib::ext::write(&buffer.front(), a);
      std::vector<int> b;
      stlib::ext::read(&buffer.front(), &b);
      assert(a == b);
    }
    {
      std::vector<int> const a = {2, 3, 5};
      std::vector<int> const b = {1, 1, 2};
      std::vector<unsigned char> buffer;
      stlib::ext::write(&buffer, a);
      stlib::ext::write(&buffer, b);
      std::vector<int> x, y;
      std::size_t const pos = stlib::ext::read(buffer, &x);
      stlib::ext::read(buffer, &y, pos);
      assert(x == a);
      assert(y == b);
    }
  }
  {
    //---------------------------------------------------------------------------
    // Vector Mathematical Functions
    {
      std::vector<int> a;
      a.push_back(2);
      a.push_back(3);
      a.push_back(5);
      assert(stlib::ext::sum(a) == 10);
      assert(stlib::ext::product(a) == 30);
      assert(stlib::ext::min(a) == 2);
      assert(stlib::ext::max(a) == 5);
      assert(stlib::ext::dot(a, a) == 38);
      assert(stlib::ext::sum(a) == 10);
    }
  }
  
  return 0;
}
