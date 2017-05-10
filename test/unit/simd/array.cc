// -*- C++ -*-

#include "stlib/simd/array.h"

#include <iostream>

#include <cassert>


template<typename T>
void
testAlignment()
{
  using stlib::simd::isAligned;
  // stlib::simd::array
  {
    stlib::simd::array<float, 0> a;
    assert(isAligned(a.data()));
  }
  {
    stlib::simd::array<float, 1> a;
    assert(isAligned(a.data()));
  }
  {
    stlib::simd::array<float, 2> a;
    assert(isAligned(a.data()));
  }
  {
    stlib::simd::array<float, 3> a;
    assert(isAligned(a.data()));
  }
  {
    stlib::simd::array<float, 4> a;
    assert(isAligned(a.data()));
  }

  // std::array
  {
    ALIGN_SIMD std::array<float, 0> a;
    assert(isAligned(a.data()));
  }
  {
    ALIGN_SIMD std::array<float, 1> a;
    assert(isAligned(a.data()));
  }
  {
    ALIGN_SIMD std::array<float, 2> a;
    assert(isAligned(a.data()));
  }
  {
    ALIGN_SIMD std::array<float, 3> a;
    assert(isAligned(a.data()));
  }
  {
    ALIGN_SIMD std::array<float, 4> a;
    assert(isAligned(a.data()));
  }
}


int
main()
{
  using stlib::simd::array;
  //---------------------------------------------------------------------------
  // Size.
  {
    std::cout << "Size of <float,0> = " << sizeof(array<float, 0>)
              << '\n'
              << "Size of <float,1> = " << sizeof(array<float, 1>)
              << '\n'
              << "Size of <float,2> = " << sizeof(array<float, 2>)
              << '\n';
    static_assert(sizeof(array<float, 0>) % stlib::simd::Alignment == 0,
                  "wrong size");
    static_assert(sizeof(array<float, 1>) % stlib::simd::Alignment == 0,
                  "wrong size");
    static_assert(sizeof(array<float, 2>) % stlib::simd::Alignment == 0,
                  "wrong size");
    static_assert(sizeof(array<float, 3>) % stlib::simd::Alignment == 0,
                  "wrong size");
    static_assert(sizeof(array<double, 0>) % stlib::simd::Alignment == 0,
                  "wrong size");
    static_assert(sizeof(array<double, 1>) % stlib::simd::Alignment == 0,
                  "wrong size");
    static_assert(sizeof(array<double, 2>) % stlib::simd::Alignment == 0,
                  "wrong size");
    static_assert(sizeof(array<double, 3>) % stlib::simd::Alignment == 0,
                  "wrong size");
  }
  //---------------------------------------------------------------------------
  // Alignment.
  testAlignment<float>();
  testAlignment<double>();
  testAlignment<char>();
  //---------------------------------------------------------------------------
  // Aggregate initializer.
  {
    array<float, 3> a = {{}};
    assert(a[0] == 0 && a[1] == 0 && a[2] == 0);
  }
  {
    array<float, 3> a = {{0}};
    assert(a[0] == 0 && a[1] == 0 && a[2] == 0);
  }
  {
    array<float, 3> a = {{2}};
    assert(a[0] == 2 && a[1] == 0 && a[2] == 0);
  }
  {
    float a[3] = {2};
    assert(a[0] == 2 && a[1] == 0 && a[2] == 0);
  }
  //---------------------------------------------------------------------------
  // fill()
  {
    array<float, 3> a;
    a.fill(23);
    assert(a == (array<float, 3>{{23, 23, 23}}));
  }
  //---------------------------------------------------------------------------
  // swap()
  {
    array<float, 3> a = {{0, 1, 2}};
    array<float, 3> b = {{2, 3, 5}};
    a.swap(b);
    assert(a == (array<float, 3>{{2, 3, 5}}));
    assert(b == (array<float, 3>{{0, 1, 2}}));
    swap(a, b);
    assert(a == (array<float, 3>{{0, 1, 2}}));
    assert(b == (array<float, 3>{{2, 3, 5}}));
  }
  //---------------------------------------------------------------------------
  // begin and end
  {
    array<float, 3> a;
    assert(std::distance(a.begin(), a.end()) == 3);
    assert(std::distance(a.rbegin(), a.rend()) == 3);
    assert(std::distance(a.cbegin(), a.cend()) == 3);
    assert(std::distance(a.crbegin(), a.crend()) == 3);
  }
  {
    array<float, 3> const a = {{}};
    assert(std::distance(a.begin(), a.end()) == 3);
    assert(std::distance(a.rbegin(), a.rend()) == 3);
  }
  //---------------------------------------------------------------------------
  // size
  {
    array<float, 0> const a = {{}};
    assert(a.size() == 0);
    assert(a.max_size() == 0);
    assert(a.empty());
  }
  {
    array<float, 3> const a = {{}};
    assert(a.size() == 3);
    assert(a.max_size() == 3);
    assert(! a.empty());
  }
  //---------------------------------------------------------------------------
  // element access
  {
    array<float, 3> const a = {{2, 3, 5}};
    assert(a[0] == 2);
    assert(a[1] == 3);
    assert(a[2] == 5);
    for (std::size_t i = 0; i != a.size(); ++i) {
      assert(a.at(i) == a[i]);
    }
    assert(a.front() == a[0]);
    assert(a.back() == a[a.size() - 1]);
    assert(a.data() == &a.front());
  }
  {
    array<float, 3> a = {{2, 3, 5}};
    assert(a[0] == 2);
    assert(a[1] == 3);
    assert(a[2] == 5);
    for (std::size_t i = 0; i != a.size(); ++i) {
      assert(a.at(i) == a[i]);
    }
    assert(a.front() == a[0]);
    assert(a.back() == a[a.size() - 1]);
    assert(a.data() == &a.front());
  }
  // Commented out due to compilation errors. See the source file.
#if 0
  //---------------------------------------------------------------------------
  // tuple interface
  {
    static_assert(std::tuple_size<array<float, 3> >::value == 3, "wrong size");
#if 0
    // CONTINUE: This causes an incomplete type error. I don't know why.
    static_assert(typeid(std::tuple_element<0, array<float, 3> >::type) ==
                  typeid(float), "wrong type");
#endif
  }
#endif
  //---------------------------------------------------------------------------
  // convert()
  {
    array<double, 0> a;
    array<float, 0> b;
    convert(a, &b);
    assert(b == (array<float, 0>{{}}));
  }
  {
    array<double, 1> a = {{2}};
    array<float, 1> b;
    convert(a, &b);
    assert(b == (array<float, 1>{{2}}));
  }
  {
    array<double, 2> a = {{2, 3}};
    array<float, 2> b;
    convert(a, &b);
    assert(b == (array<float, 2>{{2, 3}}));
  }
  {
    array<double, 3> a = {{2, 3, 5}};
    array<float, 3> b;
    convert(a, &b);
    assert(b == (array<float, 3>{{2, 3, 5}}));
  }
  {
    array<double, 4> a = {{2, 3, 5, 7}};
    array<float, 4> b;
    convert(a, &b);
    assert(b == (array<float, 4>{{2, 3, 5, 7}}));
  }
  {
    array<double, 5> a = {{2, 3, 5, 7, 11}};
    array<float, 5> b;
    convert(a, &b);
    assert(b == (array<float, 5>{{2, 3, 5, 7, 11}}));
  }

  return 0;
}
