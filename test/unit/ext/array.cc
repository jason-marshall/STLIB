// -*- C++ -*-

#include "stlib/ext/array.h"

#include <sstream>

int
main()
{
  //---------------------------------------------------------------------------
  // Size.
  {
    std::cout << "Size of <int,1> = " << sizeof(std::array<int, 1>)
              << '\n'
              << "Size of <int,2> = " << sizeof(std::array<int, 2>)
              << '\n';
  }
  //---------------------------------------------------------------------------
  // Aggregate initializer.
  {
    std::array<int, 3> a = {{}};
    assert(a[0] == 0 && a[1] == 0 && a[2] == 0);
  }
  {
    std::array<int, 3> a = {{0}};
    assert(a[0] == 0 && a[1] == 0 && a[2] == 0);
  }
  {
    std::array<int, 3> a = {{2}};
    assert(a[0] == 2 && a[1] == 0 && a[2] == 0);
  }
  {
    int a[3] = {2};
    assert(a[0] == 2 && a[1] == 0 && a[2] == 0);
  }

  
  {
    USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
    
    //---------------------------------------------------------------------------
    // Array Assignment Operators with a Scalar Operand.
    {
      std::array<int, 3> a = {{2, 3, 5}};
      // +=
      {
        using stlib::ext::operator+=;
        std::array<int, 3> b = a;
        b += 1;
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] + 1);
        }
      }
      // -=
      {
        std::array<int, 3> b = a;
        b -= 1;
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] - 1);
        }
      }
      // *=
      {
        std::array<int, 3> b = a;
        b *= 2;
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] * 2);
        }
      }
      // /=
      {
        std::array<int, 3> b = a;
        b /= 2;
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] / 2);
        }
      }
      // %=
      {
        std::array<int, 3> b = a;
        b %= 2;
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] % 2);
        }
      }
      // <<=
      {
        std::array<int, 3> b = a;
        b <<= 1;
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] << 1);
        }
      }
      // >>=
      {
        std::array<int, 3> b = a;
        b >>= 1;
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] >> 1);
        }
      }
    }
    {
      std::array<unsigned, 3> const a = {{2, 3, 5}};
      // +=
      {
        using stlib::ext::operator+=;
        std::array<unsigned, 3> b = a;
        b += unsigned(1);
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] + 1);
        }
      }
      // -=
      {
        std::array<unsigned, 3> b = a;
        b -= unsigned(1);
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] - 1);
        }
      }
      // *=
      {
        std::array<unsigned, 3> b = a;
        b *= unsigned(2);
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] * 2);
        }
      }
      // /=
      {
        std::array<unsigned, 3> b = a;
        b /= unsigned(2);
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] / 2);
        }
      }
    }
    {
      std::array<float, 3> const a = {{2, 3, 5}};
      // +=
      {
        using stlib::ext::operator+=;
        std::array<float, 3> b = a;
        b += float(1);
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] + 1);
        }
      }
      // -=
      {
        std::array<float, 3> b = a;
        b -= float(1);
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] - 1);
        }
      }
      // *=
      {
        std::array<float, 3> b = a;
        b *= float(2);
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] * 2);
        }
      }
      // /=
      {
        std::array<float, 3> b = a;
        b /= float(2);
        for (std::size_t n = 0; n != b.size(); ++n) {
          assert(b[n] == a[n] / 2);
        }
      }
    }

    //---------------------------------------------------------------------------
    // Array Assignment Operators with a Array Operand.
    {
      std::array<int, 3> a = {{2, 3, 5}};
      std::array<int, 3> b = {{1, 2, 3}};
      // +=
      {
        std::array<int, 3> c = a;
        c += b;
        for (std::size_t n = 0; n != c.size(); ++n) {
          assert(c[n] == a[n] + b[n]);
        }
      }
      // -=
      {
        std::array<int, 3> c = a;
        c -= b;
        for (std::size_t n = 0; n != c.size(); ++n) {
          assert(c[n] == a[n] - b[n]);
        }
      }
      // *=
      {
        std::array<int, 3> c = a;
        c *= b;
        for (std::size_t n = 0; n != c.size(); ++n) {
          assert(c[n] == a[n] * b[n]);
        }
      }
      // /=
      {
        std::array<int, 3> c = a;
        c /= b;
        for (std::size_t n = 0; n != c.size(); ++n) {
          assert(c[n] == a[n] / b[n]);
        }
      }
      // %=
      {
        std::array<int, 3> c = a;
        c %= b;
        for (std::size_t n = 0; n != c.size(); ++n) {
          assert(c[n] == a[n] % b[n]);
        }
      }
      // <<=
      {
        std::array<int, 3> c = a;
        c <<= b;
        for (std::size_t n = 0; n != c.size(); ++n) {
          assert(c[n] == a[n] << b[n]);
        }
      }
      // >>=
      {
        std::array<int, 3> c = a;
        c >>= b;
        for (std::size_t n = 0; n != c.size(); ++n) {
          assert(c[n] == a[n] >> b[n]);
        }
      }
    }
    {
      std::array<std::array<double, 2>, 2> simplex = {{{{1, 2}}, {{3, 4}}}};
      std::array<double, 2> point = {{1, 2}};
      simplex -= point;
      assert(simplex == (std::array<std::array<double, 2>, 2>
        {{{{0, 0}}, {{2, 2}}}}));
    }

    //---------------------------------------------------------------------------
    // Unary Operators
    {
      std::array<double, 3> a = {{1, 2, 3}};
      assert(a == +a);
      assert(a == -(-a));
      std::array<double, 3> b;
      b = -a;
      for (std::size_t i = 0; i != b.size(); ++i) {
        assert(b[i] == -a[i]);
      }
    }
    {
      std::array<std::array<double, 3>, 3> a = {{{{1, 2, 3}},
                                                 {{4, 5, 6}},
                                                 {{7, 8, 9}}
        }
      };
      assert(a == +a);
      assert(a == -(-a));
      std::array<std::array<double, 3>, 3> b;
      b = -a;
      for (std::size_t i = 0; i != b.size(); ++i) {
        assert(b[i] == -a[i]);
      }
    }

    //---------------------------------------------------------------------------
    // Binary Operators
    {
      // Array-scalar.
      {
        // int, int
        const std::array<int, 1> a = {{1}};
        const int b = 2;
        std::array<int, 1> c;
        c = a + b;
        assert(c[0] == a[0] + b);
        c = a - b;
        assert(c[0] == a[0] - b);
        c = a * b;
        assert(c[0] == a[0] * b);
        c = a / b;
        assert(c[0] == a[0] / b);
        c = a % b;
        assert(c[0] == a[0] % b);
      }
      // Scalar-array.
      {
        // int, int
        const int a = 2;
        const std::array<int, 1> b = {{1}};
        std::array<int, 1> c;
        c = a + b;
        assert(c[0] == a + b[0]);
        c = a - b;
        assert(c[0] == a - b[0]);
        c = a * b;
        assert(c[0] == a * b[0]);
        c = a / b;
        assert(c[0] == a / b[0]);
        c = a % b;
        assert(c[0] == a % b[0]);
      }
      // Array-array.
      {
        // int, int
        const std::array<int, 1> a = {{2}};
        const std::array<int, 1> b = {{1}};
        std::array<int, 1> c;
        c = a + b;
        assert(c[0] == a[0] + b[0]);
        c = a - b;
        assert(c[0] == a[0] - b[0]);
        c = a * b;
        assert(c[0] == a[0] * b[0]);
        c = a / b;
        assert(c[0] == a[0] / b[0]);
        c = a % b;
        assert(c[0] == a[0] % b[0]);
      }
    }
  } // USING_STLIB_EXT_ARRAY_OPERATORS

  
  {
    USING_STLIB_EXT_ARRAY_IO_OPERATORS;
    //---------------------------------------------------------------------------
    // File I/O.
    {
      std::array<int, 3> a = {{2, 3, 5}};
      std::ostringstream out;
      out << a;
      std::array<int, 3> b;
      std::istringstream in(out.str());
      in >> b;
      assert(a == b);
    }
    {
      // Here we use the fact that we injected the I/O operators into std.
      std::array<std::array<int, 1>, 3> a = {{{{2}}, {{3}}, {{5}}}};
      std::ostringstream out;
      out << a;
      std::array<std::array<int, 1>, 3> b;
      std::istringstream in(out.str());
      in >> b;
      assert(a == b);
    }
    {
      std::array<int, 3> a = {{2, 3, 5}};
      std::ostringstream out;
      stlib::ext::write(a, out);
      std::array<int, 3> b;
      std::istringstream in(out.str());
      stlib::ext::read(&b, in);
      assert(a == b);
    }
  } // USING_STLIB_EXT_ARRAY_IO
  
  
  //---------------------------------------------------------------------------
  // Make an array.
  {
    // ConvertArray
    {
      std::array<char, 1> x = {{2}};
      std::array<int, 1> y = {{2}};
      assert(stlib::ext::ConvertArray<int>::convert(x) == y);
    }
    {
      std::array<char, 1> x = {{2}};
      std::array<char, 1> y = {{2}};
      assert(stlib::ext::ConvertArray<char>::convert(x) == y);
    }

    // convert_array
    {
      std::array<char, 1> x = {{2}};
      std::array<int, 1> y = {{2}};
      assert(stlib::ext::convert_array<int>(x) == y);
    }
    {
      std::array<char, 1> x = {{2}};
      std::array<char, 1> y = {{2}};
      assert(stlib::ext::convert_array<char>(x) == y);
    }

    // filled_array
    {
      std::array<int, 1> x = {{2}};
      assert((stlib::ext::filled_array<std::array<int, 1> >(2) == x));
    }
    {
      std::array<int, 2> x = {{2, 2}};
      assert((stlib::ext::filled_array<std::array<int, 2> >(2) == x));
    }
    {
      std::array<int, 3> x = {{2, 2, 2}};
      assert((stlib::ext::filled_array<std::array<int, 3> >(2) == x));
    }

    // copy_array
    {
      int data[] = {2};
      std::array<int, 1> x = {{2}};
      assert((stlib::ext::copy_array<std::array<int, 1> >(data) == x));
    }
    {
      int data[] = {2, 3};
      std::array<int, 2> x = {{2, 3}};
      assert((stlib::ext::copy_array<std::array<int, 2> >(data) == x));
    }
    {
      int data[] = {2, 3, 5};
      std::array<int, 3> x = {{2, 3, 5}};
      assert((stlib::ext::copy_array<std::array<int, 3> >(data) == x));
    }
  }


  {
    //---------------------------------------------------------------------------
    // Array Mathematical Functions
    {
      const std::array<int, 3> a = {{2, 3, 5}};
      assert(stlib::ext::sum(a) == 10);
      assert(stlib::ext::product(a) == 30);
      assert(stlib::ext::min(a) == 2);
      assert(stlib::ext::max(a) == 5);
      const std::array<int, 3> b = {{7, 11 , 13}};
      assert(stlib::ext::min(a, b) == a);
      assert(stlib::ext::max(a, b) == b);
      assert(stlib::ext::dot(a, b) == 112);
      {
        const std::array<int, 3> r = {{ -16, 9, 1}};
        assert(stlib::ext::cross(a, b) == r);
      }
      {
        std::array<int, 3> c;
        stlib::ext::cross(a, b, &c);
        const std::array<int, 3> r = {{ -16, 9, 1}};
        assert(c == r);
      }
      const std::array<int, 3> c = {{17, 19, 23}};
      assert(stlib::ext::tripleProduct(a, b, c) == -78);
    }
    {
      const std::array<int, 2> a = {{2, 3}};
      const std::array<int, 2> b = {{5, 7}};
      assert(stlib::ext::discriminant(a, b) == 2 * 7 - 3 * 5);
    }
    {
      std::array<double, 2> a = {{3, 4}};
      assert(stlib::ext::squaredMagnitude(a) == 25.);
      assert(std::abs(stlib::ext::magnitude(a) - 5.) <
             5. * std::numeric_limits<double>::epsilon());
      stlib::ext::normalize(&a);
      assert(std::abs(stlib::ext::magnitude(a) - 1.) <
             std::numeric_limits<double>::epsilon());
    }
    {
      std::array<double, 2> a = {{1., -1.}};
      stlib::ext::negateElements(&a);
      assert(a == (std::array<double, 2>{{-1., 1.}}));
    }
    {
      std::array<bool, 2> a = {{false, true}};
      stlib::ext::negateElements(&a);
      assert(a == (std::array<bool, 2>{{true, false}}));
    }
    {
      const std::array<double, 2> a = {{1, 2}};
      const std::array<double, 2> b = {{4, 6}};
      assert(stlib::ext::squaredDistance(a, b) == 25.);
      assert(std::abs(stlib::ext::euclideanDistance(a, b) - 5.) <
             5. * std::numeric_limits<double>::epsilon());
    }
    //---------------------------------------------------------------------------
    // SIMD implementations.
    {
      assert(stlib::ext::dot(std::array<float, 3>{{1, 2, 3}},
                             std::array<float, 3>{{2, 3, 5}}) == 23);
      assert(stlib::ext::dot(std::array<float, 4>{{1, 2, 3, 4}},
                             std::array<float, 4>{{2, 3, 5, 7}}) == 51);
      assert(stlib::ext::dot(std::array<double, 4>{{1, 2, 3, 4}},
                             std::array<double, 4>{{2, 3, 5, 7}}) == 51);
      assert(stlib::ext::squaredDistance(std::array<float, 3>{{1, 2, 3}},
                                         std::array<float, 3>{{2, 3, 5}}) == 6);
      assert(stlib::ext::squaredDistance(std::array<float, 4>{{1, 2, 3, 4}},
                                         std::array<float, 4>{{2, 3, 5, 7}}) == 15);
      assert(stlib::ext::squaredDistance(std::array<double, 4>{{1, 2, 3, 4}},
                                         std::array<double, 4>{{2, 3, 5, 7}}) == 15);
    }
  } // USING_STLIB_EXT_ARRAY_MATH

  return 0;
}
