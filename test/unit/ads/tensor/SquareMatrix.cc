// -*- C++ -*-

#include "stlib/ads/tensor.h"

#include <iostream>
#include <sstream>

using namespace stlib;

int
main()
{
  using namespace ads;

  //
  // 1x1 matrices.
  //
  {
    // sizeof
    std::cout << "sizeof(double) = " << sizeof(double) << '\n';
    std::cout << "sizeof(SquareMatrix<1>) = "
              << sizeof(SquareMatrix<1>) << '\n';
  }
  {
    // default constructor
    std::cout << "Default Constructor:\n"
              << SquareMatrix<1>() << '\n';
    std::cout << "Element Constructor:\n"
              << SquareMatrix<1>(3) << '\n';
    const double a[] = {3};
    std::cout << "Array Constructor:\n"
              << SquareMatrix<1>(a) << '\n';
    std::cout << "Value Constructor:\n"
              << SquareMatrix<1>(7) << '\n';
  }
  {
    // Copy.
    const double a[] = {3};
    SquareMatrix<1> x(a);
    SquareMatrix<1> y(x);
    assert(x == y);
  }
  {
    // Assignment
    const double a[] = {3};
    SquareMatrix<1> x(a);
    SquareMatrix<1> y;
    y = x;
    assert(x == y);
  }
  {
    SquareMatrix<1> m(7);

    //
    // Accessors.
    //
    assert(m.size() == 1);
    assert(m.end() - m.begin() == 1);

    //
    // Subscripting.
    //
    assert(m[0] == 7);
    assert(m(0) == 7);

    //
    // Indexing.
    //
    assert(m(0, 0) == 7);
  }
  {
    SquareMatrix<1> m;

    //
    // Subscripting.
    //
    m[0] = 7;
    assert(m[0] == 7);
  }
  {
    SquareMatrix<1> m;

    //
    // Indexing.
    //
    m(0, 0) = 7;
    assert(m(0, 0) == 7);
  }
  {
    // ==
    SquareMatrix<1, int> a(3);
    SquareMatrix<1, int> b(3);
    assert(a == b);
  }
  {
    // !=
    SquareMatrix<1, int> a(3);
    SquareMatrix<1, int> b(4);
    assert(a != b);
  }
  {
    // Addition.
    SquareMatrix<1, int> a(3);
    SquareMatrix<1, int> b(1);
    SquareMatrix<1, int> c(4);
    assert(a + b == c);
    a += b;
    assert(a == c);
  }
  {
    // Subtraction.
    SquareMatrix<1, int> a(3);
    SquareMatrix<1, int> b(1);
    SquareMatrix<1, int> c(4);
    assert(c - a == b);
    assert(c - b == a);
    c -= b;
    assert(c == a);
  }
  //
  // Binary operators
  //
  {
    // SquareMatrix-scalar addition.
    assert((SquareMatrix<1, int>(3) + 2 ==
            SquareMatrix<1, int>(5)));
    // Scalar-SquareMatrix addition.
    assert((2 + SquareMatrix<1, int>(3) ==
            SquareMatrix<1, int>(5)));
    // SquareMatrix-SquareMatrix addition.
    assert((SquareMatrix<1, int>(3) +
            SquareMatrix<1, int>(3) ==
            SquareMatrix<1, int>(6)));

    // SquareMatrix-scalar subtraction.
    assert((SquareMatrix<1, int>(3) - 2 ==
            SquareMatrix<1, int>(1)));
    // Scalar-SquareMatrix subtraction.
    assert((2 - SquareMatrix<1, int>(3) ==
            SquareMatrix<1, int>(-1)));
    // SquareMatrix-SquareMatrix subtraction.
    assert((SquareMatrix<1, int>(4) -
            SquareMatrix<1, int>(3) ==
            SquareMatrix<1, int>(1)));

    // SquareMatrix-scalar product.
    assert((SquareMatrix<1, int>(3) * 2 ==
            SquareMatrix<1, int>(6)));
    // Scalar-SquareMatrix product.
    assert((2 * SquareMatrix<1, int>(3) ==
            SquareMatrix<1, int>(6)));
    // SquareMatrix-SquareMatrix product.
    assert((SquareMatrix<1, int>(3) *
            SquareMatrix<1, int>(3) ==
            SquareMatrix<1, int>(9)));

    // SquareMatrix-scalar division.
    assert((SquareMatrix<1>(3) / 2. ==
            SquareMatrix<1>(1.5)));
    // Scalar-SquareMatrix division.
    assert((1. / SquareMatrix<1>(4) ==
            SquareMatrix<1>(1. / 4)));
  }
  //
  // Math operators.
  //
  {
    // sum
    SquareMatrix<1, int> a(4);
    assert(computeSum(a) == 4);
  }
  {
    // product
    SquareMatrix<1, int> a(4);
    assert(computeProduct(a) == 4);
  }
  {
    // min
    SquareMatrix<1, int> a(4);
    assert(computeMinimum(a) == 4);
  }
  {
    // max
    SquareMatrix<1, int> a(4);
    assert(computeMaximum(a) == 4);
  }
  {
    // negate
    SquareMatrix<1, int> a(4);
    SquareMatrix<1, int> b(-4);
    b.negate();
    assert(a == b);
  }
  {
    SquareMatrix<1, int> a(4);
    // determinant
    assert(computeDeterminant(a) == 4);
    // trace
    assert(computeTrace(a) == 4);
    // transpose
    assert((computeTranspose(a) == SquareMatrix<1, int>(4)));
  }
  {
    // inverse
    SquareMatrix<1> a(4);
    assert((computeInverse(a) == SquareMatrix<1>(1. / 4)));
    // Frobenius norm
    assert(computeFrobeniusNorm(a) == std::abs(4.0));
  }
  //
  // I/O
  //
  {
    std::stringstream file;
    SquareMatrix<1, int> a(4);
    file << a << '\n';
    SquareMatrix<1, int> b;
    file >> b;
    assert(a == b);
  }

  //
  // 2x2 matrices.
  //
  {
    // sizeof
    std::cout << "sizeof(double) = " << sizeof(double) << '\n';
    std::cout << "sizeof(SquareMatrix<2>) = "
              << sizeof(SquareMatrix<2>) << '\n';
  }
  {
    // default constructor
    std::cout << "Default Constructor:\n"
              << SquareMatrix<2>() << '\n';
    std::cout << "Element Constructor:\n"
              << SquareMatrix<2>(0, 1, 2, 3) << '\n';
    const double a[] = {0, 1, 2, 3};
    std::cout << "Array Constructor:\n"
              << SquareMatrix<2>(a) << '\n';
    std::cout << "Value Constructor:\n"
              << SquareMatrix<2>(7) << '\n';
  }
  {
    // Copy.
    const double a[] = {0, 1, 2, 3};
    SquareMatrix<2> x(a);
    SquareMatrix<2> y(x);
    assert(x == y);
  }
  {
    // Assignment
    const double a[] = {0, 1, 2, 3};
    SquareMatrix<2> x(a);
    SquareMatrix<2> y;
    y = x;
    assert(x == y);
  }
  {
    SquareMatrix<2> m(0, 1, 2, 3);

    //
    // Accessors.
    //
    assert(m.size() == 4);
    assert(m.end() - m.begin() == 4);

    //
    // Subscripting.
    //
    for (std::size_t i = 0; i != m.size(); ++i) {
      assert(m[i] == i);
      assert(m(i) == i);
    }

    //
    // Indexing.
    //
    for (std::size_t i = 0; i != 2; ++i) {
      for (std::size_t j = 0; j != 2; ++j) {
        assert(m(i, j) == i * 2 + j);
      }
    }

  }
  {
    SquareMatrix<2> m;

    //
    // Subscripting.
    //
    for (std::size_t i = 0; i != m.size(); ++i) {
      m[i] = i;
    }
    for (std::size_t i = 0; i != m.size(); ++i) {
      assert(m[i] == i);
    }
  }
  {
    SquareMatrix<2> m;

    //
    // Indexing.
    //
    for (std::size_t i = 0; i != 2; ++i) {
      for (std::size_t j = 0; j != 2; ++j) {
        m(i, j) = i * 2 + j;
      }
    }
    for (std::size_t i = 0; i != 2; ++i) {
      for (std::size_t j = 0; j != 2; ++j) {
        assert(m(i, j) == i * 2 + j);
      }
    }
  }
  {
    // ==
    SquareMatrix<2, int> a(0, 1, 2, 3);
    SquareMatrix<2, int> b(0, 1, 2, 3);
    assert(a == b);
  }
  {
    // !=
    SquareMatrix<2, int> a(0, 1, 2, 3);
    SquareMatrix<2, int> b(0, 1, 2, 4);
    assert(a != b);
  }
  {
    // Addition.
    SquareMatrix<2, int> a(0, 1, 2, 3);
    SquareMatrix<2, int> b(1, 2, 3, 1);
    SquareMatrix<2, int> c(1, 3, 5, 4);
    assert(a + b == c);
    a += b;
    assert(a == c);
  }
  {
    // Subtraction.
    SquareMatrix<2, int> a(0, 1, 2, 3);
    SquareMatrix<2, int> b(1, 2, 3, 1);
    SquareMatrix<2, int> c(1, 3, 5, 4);
    assert(c - a == b);
    assert(c - b == a);
    c -= b;
    assert(c == a);
  }
  //
  // Binary operators
  //
  {
    // SquareMatrix-scalar addition.
    assert((SquareMatrix<2, int>(0, 1, 2, 3) + 2 ==
            SquareMatrix<2, int>(2, 3, 4, 5)));
    // Scalar-SquareMatrix addition.
    assert((2 + SquareMatrix<2, int>(0, 1, 2, 3) ==
            SquareMatrix<2, int>(2, 3, 4, 5)));
    // SquareMatrix-SquareMatrix addition.
    assert((SquareMatrix<2, int>(0, 1, 2, 3) +
            SquareMatrix<2, int>(0, 1, 2, 3) ==
            SquareMatrix<2, int>(0, 2, 4, 6)));

    // SquareMatrix-scalar subtraction.
    assert((SquareMatrix<2, int>(0, 1, 2, 3) - 2 ==
            SquareMatrix<2, int>(-2, -1, 0, 1)));
    // Scalar-SquareMatrix subtraction.
    assert((2 - SquareMatrix<2, int>(0, 1, 2, 3) ==
            SquareMatrix<2, int>(2, 1, 0, -1)));
    // SquareMatrix-SquareMatrix subtraction.
    assert((SquareMatrix<2, int>(1, 2, 3, 4) -
            SquareMatrix<2, int>(0, 1, 2, 3) ==
            SquareMatrix<2, int>(1)));

    // SquareMatrix-scalar product.
    assert((SquareMatrix<2, int>(0, 1, 2, 3) * 2 ==
            SquareMatrix<2, int>(0, 2, 4, 6)));
    // Scalar-SquareMatrix product.
    assert((2 * SquareMatrix<2, int>(0, 1, 2, 3) ==
            SquareMatrix<2, int>(0, 2, 4, 6)));
    // SquareMatrix-SquareMatrix product.
    assert((SquareMatrix<2, int>(0, 1, 2, 3) *
            SquareMatrix<2, int>(0, 1, 2, 3) ==
            SquareMatrix<2, int>(2, 3, 6, 11)));

    // SquareMatrix-scalar division.
    assert((SquareMatrix<2>(0, 1, 2, 3) / 2. ==
            SquareMatrix<2>(0, .5, 1, 1.5)));
    // Scalar-SquareMatrix division.
    assert((1. / SquareMatrix<2>(1, 2, 3, 4) ==
            SquareMatrix<2>(1. / 1, 1. / 2, 1. / 3, 1. / 4)));
  }
  //
  // Math operators.
  //
  {
    // sum
    SquareMatrix<2, int> a(1, 2, 3, 4);
    assert(computeSum(a) == 10);
  }
  {
    // product
    SquareMatrix<2, int> a(1, 2, 3, 4);
    assert(computeProduct(a) == 24);
  }
  {
    // min
    SquareMatrix<2, int> a(1, 2, 3, 4);
    assert(computeMinimum(a) == 1);
  }
  {
    // max
    SquareMatrix<2, int> a(1, 2, 3, 4);
    assert(computeMaximum(a) == 4);
  }
  {
    // negate
    SquareMatrix<2, int> a(1, 2, 3, 4);
    SquareMatrix<2, int> b(-1, -2, -3, -4);
    b.negate();
    assert(a == b);
  }
  {
    SquareMatrix<2, int> a(1, 2, 3, 4);
    // determinant
    assert(computeDeterminant(a) == -2);
    // trace
    assert(computeTrace(a) == 5);
    // transpose
    assert((computeTranspose(a) == SquareMatrix<2, int>(1, 3, 2, 4)));
  }
  {
    // inverse
    SquareMatrix<2> a(1, 2, 3, 4);
    assert((computeInverse(a) == SquareMatrix<2>(-2., 1.,
            3. / 2., -1. / 2.)));
    double d = computeDeterminant(a);
    assert(computeInverse(a, d) == computeInverse(a));
    // Frobenius norm
    assert(std::abs(computeFrobeniusNorm(a) - std::sqrt(30.)) <
           10 * std::numeric_limits<double>::epsilon());
  }
  //
  // I/O
  //
  {
    std::stringstream file;
    SquareMatrix<2, int> a(1, 2, 3, 4);
    file << a << '\n';
    SquareMatrix<2, int> b;
    file >> b;
    assert(a == b);
  }

  //
  // 3x3 matrices.
  //
  {
    // sizeof
    std::cout << "sizeof(double) = " << sizeof(double) << '\n';
    std::cout << "sizeof(SquareMatrix<3>) = "
              << sizeof(SquareMatrix<3>) << '\n';
  }
  {
    // default constructor
    std::cout << "Default Constructor:\n"
              << SquareMatrix<3>() << '\n';
    std::cout << "Element Constructor:\n"
              << SquareMatrix<3>(0, 1, 2, 3, 4, 5, 6, 7, 8) << '\n';
    const double a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::cout << "Array Constructor:\n"
              << SquareMatrix<3>(a) << '\n';
    std::cout << "Value Constructor:\n"
              << SquareMatrix<3>(7) << '\n';
  }
  {
    // Copy.
    const double a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    SquareMatrix<3> x(a);
    SquareMatrix<3> y(x);
    assert(x == y);
  }
  {
    // Assignment
    const double a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    SquareMatrix<3> x(a);
    SquareMatrix<3> y;
    y = x;
    assert(x == y);
  }
  {
    SquareMatrix<3> m(0, 1, 2, 3, 4, 5, 6, 7, 8);

    //
    // Accessors.
    //
    assert(m.size() == 9);
    assert(m.end() - m.begin() == 9);

    //
    // Subscripting.
    //
    for (std::size_t i = 0; i != m.size(); ++i) {
      assert(m[i] == i);
      assert(m(i) == i);
    }

    //
    // Indexing.
    //
    for (std::size_t i = 0; i != 3; ++i) {
      for (std::size_t j = 0; j != 3; ++j) {
        assert(m(i, j) == i * 3 + j);
      }
    }

  }
  {
    SquareMatrix<3> m;

    //
    // Subscripting.
    //
    for (std::size_t i = 0; i != m.size(); ++i) {
      m[i] = i;
    }
    for (std::size_t i = 0; i != m.size(); ++i) {
      assert(m[i] == i);
    }
  }
  {
    SquareMatrix<3> m;

    //
    // Indexing.
    //
    for (std::size_t i = 0; i != 3; ++i) {
      for (std::size_t j = 0; j != 3; ++j) {
        m(i, j) = i * 3 + j;
      }
    }
    for (std::size_t i = 0; i != 3; ++i) {
      for (std::size_t j = 0; j != 3; ++j) {
        assert(m(i, j) == i * 3 + j);
      }
    }
  }
  {
    // ==
    SquareMatrix<3, int> a(0, 1, 2, 3, 4, 5, 6, 7, 8);
    SquareMatrix<3, int> b(0, 1, 2, 3, 4, 5, 6, 7, 8);
    assert(a == b);
  }
  {
    // !=
    SquareMatrix<3, int> a(0, 1, 2, 3, 4, 5, 6, 7, 8);
    SquareMatrix<3, int> b(0, 1, 2, 3, 4, 5, 6, 7, 9);
    assert(a != b);
  }
  {
    // Addition.
    SquareMatrix<3, int> a(0, 1, 2, 3, 4, 3, 2, 1, 0);
    SquareMatrix<3, int> b(1, 2, 3, 1, 2, 3, 1, 2, 3);
    SquareMatrix<3, int> c(1, 3, 5, 4, 6, 6, 3, 3, 3);
    assert(a + b == c);
    a += b;
    assert(a == c);
  }
  {
    // Subtraction.
    SquareMatrix<3, int> a(0, 1, 2, 3, 4, 3, 2, 1, 0);
    SquareMatrix<3, int> b(1, 2, 3, 1, 2, 3, 1, 2, 3);
    SquareMatrix<3, int> c(1, 3, 5, 4, 6, 6, 3, 3, 3);
    assert(c - a == b);
    assert(c - b == a);
    c -= b;
    assert(c == a);
  }
  //
  // Binary operators
  //
  {
    // SquareMatrix-scalar addition.
    assert((SquareMatrix<3, int>(0, 1, 2, 3, 4, 5, 6, 7, 8) + 2 ==
            SquareMatrix<3, int>(2, 3, 4, 5, 6, 7, 8, 9, 10)));
    // Scalar-SquareMatrix addition.
    assert((2 + SquareMatrix<3, int>(0, 1, 2, 3, 4, 5, 6, 7, 8) ==
            SquareMatrix<3, int>(2, 3, 4, 5, 6, 7, 8, 9, 10)));
    // SquareMatrix-SquareMatrix addition.
    assert((SquareMatrix<3, int>(0, 1, 2, 3, 4, 5, 6, 7, 8) +
            SquareMatrix<3, int>(0, 1, 2, 3, 4, 5, 6, 7, 8) ==
            SquareMatrix<3, int>(0, 2, 4, 6, 8, 10, 12, 14, 16)));

    // SquareMatrix-scalar subtraction.
    assert((SquareMatrix<3, int>(0, 1, 2, 3, 4, 5, 6, 7, 8) - 2 ==
            SquareMatrix<3, int>(-2, -1, 0, 1, 2, 3, 4, 5, 6)));
    // Scalar-SquareMatrix subtraction.
    assert((2 - SquareMatrix<3, int>(0, 1, 2, 3, 4, 5, 6, 7, 8) ==
            SquareMatrix<3, int>(2, 1, 0, -1, -2, -3, -4, -5, -6)));
    // SquareMatrix-SquareMatrix subtraction.
    assert((SquareMatrix<3, int>(1, 2, 3, 4, 5, 6, 7, 8, 9) -
            SquareMatrix<3, int>(0, 1, 2, 3, 4, 5, 6, 7, 8) ==
            SquareMatrix<3, int>(1)));

    // SquareMatrix-scalar product.
    assert((SquareMatrix<3, int>(0, 1, 2, 3, 4, 5, 6, 7, 8) * 2 ==
            SquareMatrix<3, int>(0, 2, 4, 6, 8, 10, 12, 14, 16)));
    // Scalar-SquareMatrix product.
    assert((2 * SquareMatrix<3, int>(0, 1, 2, 3, 4, 5, 6, 7, 8) ==
            SquareMatrix<3, int>(0, 2, 4, 6, 8, 10, 12, 14, 16)));
    // SquareMatrix-SquareMatrix product.
    assert((SquareMatrix<3, int>(0, 1, 2, 3, 4, 5, 6, 7, 8) *
            SquareMatrix<3, int>(0, 1, 2, 3, 4, 5, 6, 7, 8) ==
            SquareMatrix<3, int>(15, 18, 21, 42, 54, 66, 69, 90, 111)));

    // SquareMatrix-scalar division.
    assert((SquareMatrix<3>(0, 1, 2, 3, 4, 5, 6, 7, 8) / 2. ==
            SquareMatrix<3>(0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4)));
    // Scalar-SquareMatrix division.
    assert((1. / SquareMatrix<3>(1, 2, 3, 4, 5, 6, 7, 8, 9) ==
            SquareMatrix<3>(1. / 1, 1. / 2, 1. / 3, 1. / 4, 1. / 5, 1. / 6,
                            1. / 7, 1. / 8, 1. / 9)));
  }
  //
  // Math operators.
  //
  {
    // sum
    SquareMatrix<3, int> a(1, 2, 3, 4, 5, 6, 7, 8, 9);
    assert(computeSum(a) == 45);
  }
  {
    // product
    SquareMatrix<3, int> a(1, 2, 3, 4, 5, 6, 7, 8, 9);
    assert(computeProduct(a) == 362880);
  }
  {
    // min
    SquareMatrix<3, int> a(1, 2, 3, 4, 5, 6, 7, 8, 9);
    assert(computeMinimum(a) == 1);
  }
  {
    // max
    SquareMatrix<3, int> a(1, 2, 3, 4, 5, 6, 7, 8, 9);
    assert(computeMaximum(a) == 9);
  }
  {
    // negate
    SquareMatrix<3, int> a(1, 2, 3, 4, 5, 6, 7, 8, 9);
    SquareMatrix<3, int> b(-1, -2, -3, -4, -5, -6, -7, -8, -9);
    b.negate();
    assert(a == b);
  }
  {
    SquareMatrix<3, int> a(1, 2, 3, 4, 5, 6, 2, 3, 5);
    // determinant
    assert(computeDeterminant(a) == -3);
    // trace
    assert(computeTrace(a) == 11);
    // transpose
    assert((computeTranspose(a) == SquareMatrix<3, int>(1, 4, 2, 2, 5, 3, 3, 6,
            5)));
  }
  {
    // inverse
    SquareMatrix<3> a(1, 2, 3, 4, 5, 6, 2, 3, 5);
    assert((computeInverse(a) == SquareMatrix<3>(-7. / 3, 1. / 3, 1,
            8. / 3, 1. / 3, -2,
            -2. / 3, -1. / 3, 1)));
    double d = computeDeterminant(a);
    assert(computeInverse(a, d) == computeInverse(a));
    // Frobenius norm
    assert(std::abs(computeFrobeniusNorm(a) - std::sqrt(129.)) <
           10 * std::numeric_limits<double>::epsilon());
  }
  //
  // I/O
  //
  {
    std::stringstream file;
    SquareMatrix<3, int> a(1, 2, 3, 4, 5, 6, 7, 8, 9);
    file << a << '\n';
    SquareMatrix<3, int> b;
    file >> b;
    assert(a == b);
  }

  //
  // 4x4 matrices.
  //
  {
    // sizeof
    std::cout << "sizeof(double) = " << sizeof(double) << '\n';
    std::cout << "sizeof(SquareMatrix<4>) = "
              << sizeof(SquareMatrix<4>) << '\n';
  }
  {
    // default constructor
    std::cout << "Default Constructor:\n"
              << SquareMatrix<4>() << '\n';
    const double a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::cout << "Array Constructor:\n"
              << SquareMatrix<4>(a) << '\n';
    std::cout << "Value Constructor:\n"
              << SquareMatrix<4>(7) << '\n';
  }
  {
    // Copy.
    const double a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    SquareMatrix<4> x(a);
    SquareMatrix<4> y(x);
    assert(x == y);
  }
  {
    // Assignment
    const double a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    SquareMatrix<4> x(a);
    SquareMatrix<4> y;
    y = x;
    assert(x == y);
  }
  {
    const double a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    SquareMatrix<4> m(a);

    //
    // Accessors.
    //
    assert(m.size() == 4 * 4);
    assert(m.end() - m.begin() == 4 * 4);

    //
    // Subscripting.
    //
    for (std::size_t i = 0; i != m.size(); ++i) {
      assert(m[i] == i);
      assert(m(i) == i);
    }

    //
    // Indexing.
    //
    for (std::size_t i = 0; i != 4; ++i) {
      for (std::size_t j = 0; j != 4; ++j) {
        assert(m(i, j) == i * 4 + j);
      }
    }
  }
  {
    SquareMatrix<4> m;

    //
    // Subscripting.
    //
    for (std::size_t i = 0; i != m.size(); ++i) {
      m[i] = i;
    }
    for (std::size_t i = 0; i != m.size(); ++i) {
      assert(m[i] == i);
    }
  }
  {
    SquareMatrix<4> m;

    //
    // Indexing.
    //
    for (std::size_t i = 0; i != 4; ++i) {
      for (std::size_t j = 0; j != 4; ++j) {
        m(i, j) = i * 4 + j;
      }
    }
    for (std::size_t i = 0; i != 4; ++i) {
      for (std::size_t j = 0; j != 4; ++j) {
        assert(m(i, j) == i * 4 + j);
      }
    }
  }
  {
    // ==
    const int array[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    SquareMatrix<4, int> a(array);
    SquareMatrix<4, int> b(array);
    assert(a == b);
  }
  {
    // !=
    const int array[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    SquareMatrix<4, int> a(array);
    SquareMatrix<4, int> b(array);
    b[0] = 1;
    assert(a != b);
  }
  {
    // Addition.
    const int aa[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    const int bb[] = {1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0};
    const int cc[] = {1, 3, 5, 3, 1, 3, 5, 3, 1, 3, 5, 3, 1, 3, 5, 3};
    SquareMatrix<4, int> a(aa);
    SquareMatrix<4, int> b(bb);
    SquareMatrix<4, int> c(cc);
    assert(a + b == c);
    a += b;
    assert(a == c);
  }
  {
    // Subtraction.
    const int aa[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    const int bb[] = {1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0};
    const int cc[] = {1, 3, 5, 3, 1, 3, 5, 3, 1, 3, 5, 3, 1, 3, 5, 3};
    SquareMatrix<4, int> a(aa);
    SquareMatrix<4, int> b(bb);
    SquareMatrix<4, int> c(cc);
    assert(c - a == b);
    assert(c - b == a);
    c -= b;
    assert(c == a);
  }
  //
  // Binary operators
  //
  {
    const int a[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    const int ap2[] = {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5};
    const int apa[] = {0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6};
    // SquareMatrix-scalar addition.
    assert((SquareMatrix<4, int>(a) + 2 ==
            SquareMatrix<4, int>(ap2)));
    // Scalar-SquareMatrix addition.
    assert((2 + SquareMatrix<4, int>(a) ==
            SquareMatrix<4, int>(ap2)));
    // SquareMatrix-SquareMatrix addition.
    assert((SquareMatrix<4, int>(a) +
            SquareMatrix<4, int>(a) ==
            SquareMatrix<4, int>(apa)));
  }
  {
    const int a[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    const int amt[] = { -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1};
    const int tma[] = {2, 1, 0, -1, 2, 1, 0, -1, 2, 1, 0, -1, 2, 1, 0, -1};
    // SquareMatrix-scalar subtraction.
    assert((SquareMatrix<4, int>(a) - 2 ==
            SquareMatrix<4, int>(amt)));
    // Scalar-SquareMatrix subtraction.
    assert((2 - SquareMatrix<4, int>(a) ==
            SquareMatrix<4, int>(tma)));
    // SquareMatrix-SquareMatrix subtraction.
    assert((SquareMatrix<4, int>(a) -
            SquareMatrix<4, int>(amt) ==
            SquareMatrix<4, int>(2)));
  }
  {
    const int a[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    const int at2[] = {0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6};
    const int ata[] = {0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18};
    // SquareMatrix-scalar product.
    assert((SquareMatrix<4, int>(a) * 2 ==
            SquareMatrix<4, int>(at2)));
    // Scalar-SquareMatrix product.
    assert((2 * SquareMatrix<4, int>(a) ==
            SquareMatrix<4, int>(at2)));
    // SquareMatrix-SquareMatrix product.
    assert((SquareMatrix<4, int>(a) *
            SquareMatrix<4, int>(a) ==
            SquareMatrix<4, int>(ata)));
  }
  {
    const double a[] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    const double adt[] = {0.5, 1, 1.5, 2, 0.5, 1, 1.5, 2, 0.5, 1, 1.5, 2, 0.5, 1, 1.5, 2};
    const double oda[] = {1. / 1, 1. / 2, 1. / 3, 1. / 4, 1. / 1, 1. / 2, 1. / 3, 1. / 4,
                          1. / 1, 1. / 2, 1. / 3, 1. / 4, 1. / 1, 1. / 2, 1. / 3, 1. / 4
                         };
    // SquareMatrix-scalar division.
    assert((SquareMatrix<4>(a) / 2. == SquareMatrix<4>(adt)));
    // Scalar-SquareMatrix division.
    assert((1. / SquareMatrix<4>(a) == SquareMatrix<4>(oda)));
  }
  //
  // Math operators.
  //
  {
    const int data[] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    SquareMatrix<4, int> a(data);
    assert(computeSum(a) == 40);
    assert(computeProduct(a) == 331776);
    assert(computeMinimum(a) == 1);
    assert(computeMaximum(a) == 4);
    a.negate();
    const int mdata[] = { -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -4};
    SquareMatrix<4, int> ma(mdata);
    assert(a == ma);
  }
  {
    const double data[] = {1, 2, 3, 4, 2, 3, 5, 8, 2, 3, 5, 7, 11, 13, 17, 19};
    const double tran[] = {1, 2, 2, 11, 2, 3, 3, 13, 3, 5, 5, 17, 4, 8, 7, 19};
    SquareMatrix<4> a(data);
    // determinant
    assert(computeDeterminant(a) == 7);
    // trace
    assert(computeTrace(a) == 28);
    // transpose
    assert((computeTranspose(a) == SquareMatrix<4>(tran)));
    // Frobenius norm
    assert(std::abs(computeFrobeniusNorm(a) - std::sqrt(1159.)) <
           10 * std::numeric_limits<double>::epsilon());
  }
  {
    // inverse
    const double data[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    SquareMatrix<4> a(data);
    assert(computeInverse(a) == a);
  }
  {
    // inverse
    const double data[] = {0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0};
    SquareMatrix<4> a(data);
    assert(computeInverse(a) == a);
  }
  {
    // inverse
    const double data[] = {1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 8};
    const double inv[] = {1, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.25, 0, 0, 0, 0, 0.125};
    SquareMatrix<4> a(data);
    SquareMatrix<4> i(inv);
    assert(computeInverse(a) == i);
  }
  //
  // I/O
  //
  {
    const int data[] = {1, 2, 3, 4, 2, 3, 5, 8, 2, 3, 5, 7, 11, 13, 17, 19};
    std::stringstream file;
    SquareMatrix<4, int> a(data);
    file << a << '\n';
    SquareMatrix<4, int> b;
    file >> b;
    assert(a == b);
  }
  return 0;
}
