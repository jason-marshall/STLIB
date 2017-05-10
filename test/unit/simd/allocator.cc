// -*- C++ -*-

#include "stlib/simd/simd.h"

#include <array>
#include <type_traits>

#include <list>
#include <vector>

#include <cassert>

using namespace stlib;

// This class may only be used for stack allocation. If it is allocated on the
// heap, center may not have the correct allignment.
struct Stack {
  __m128 center;
};

// This class may be allocated on the stack or on the heap. We allocate data
// for the SIMD vectors using std::vector and simd::allocator. Then the
// variable center just points to the aligned memory. Note that one needs
// to define the copy constructor and assignment operator. The synthesized
// versions would not work correctly.
struct Heap {
  typedef std::array<float, 4> m128;
  typedef std::vector<m128, simd::allocator<m128> > Vector128;

  Vector128 data;
  float* center;

  Heap() :
    data(1),
    center(&data[0][0])
  {
  }

  Heap(const Heap& other) :
    data(other.data),
    center(&data[0][0])
  {
  }

  Heap&
  operator=(const Heap& other)
  {
    if (this != &other) {
      data = other.data;
      center = &data[0][0];
    }
    return *this;
  }
};

int
main()
{
  {
    // std::array with std::vector
    typedef std::array<float, 4> m128;
    typedef std::vector<m128, simd::allocator<m128> > Vector128;

    for (std::size_t i = 1; i != 10; ++i) {
      Vector128 x(i);
      assert(std::size_t(&x[0]) % 16 == 0);
    }

    {
      Vector128 x;
      for (std::size_t i = 0; i != 10; ++i) {
        x.push_back(m128());
        assert(std::size_t(&x[0]) % 16 == 0);
      }
    }

    {
      Vector128 x(10);
      for (std::size_t i = 0; i != x.size(); ++i) {
        _mm_store_ps(&x[i][0], _mm_set1_ps(i));
      }
      for (std::size_t i = 0; i != x.size(); ++i) {
        for (std::size_t j = 0; j != 4; ++j) {
          assert(x[i][j] == i);
        }
        __m128 a = _mm_load_ps(&x[i][0]);
        const float* p = reinterpret_cast<const float*>(&a);
        for (std::size_t j = 0; j != 4; ++j) {
          assert(p[j] == i);
        }
      }
    }
  }
  {
    // std::array with std::list
    typedef std::array<float, 4> m128;
    typedef std::list<m128, simd::allocator<m128> > List128;

    {
      List128 x;
      for (std::size_t i = 0; i != 10; ++i) {
        m128 v = {{float(i), float(i), float(i), float(i)}};
        x.push_back(v);
      }
      for (List128::const_iterator i = x.begin(); i != x.end(); ++i) {
        // CONTINUE: This does not work with MSVC, at least not on 32-bit 
        // architectures.
#ifndef _MSC_VER
        assert(std::size_t(&*i) % 16 == 0);
#endif
      }
    }
  }

  // A class that may only be used if it is allocated on the stack.
  {
    Stack x;
    x.center = _mm_set_ps(0, 3, 2, 1);
    const float* p = reinterpret_cast<const float*>(&x.center);
    assert(p[0] == 1);
    assert(p[1] == 2);
    assert(p[2] == 3);
    assert(p[3] == 0);
  }

  // A class that stores a 3-D point. Allocation on the stack.
  {
    Heap x;
    _mm_store_ps(x.center, _mm_set_ps(0, 3, 2, 1));
    __m128 a = _mm_load_ps(x.center);
    const float* p = reinterpret_cast<const float*>(&a);
    assert(p[0] == 1);
    assert(p[1] == 2);
    assert(p[2] == 3);
    assert(p[3] == 0);
  }

  // A class that stores a 3-D point. Allocation on the heap.
  {
    Heap* x = new Heap;
    _mm_store_ps(x->center, _mm_set_ps(0, 3, 2, 1));
    __m128 a = _mm_load_ps(x->center);
    const float* p = reinterpret_cast<const float*>(&a);
    assert(p[0] == 1);
    assert(p[1] == 2);
    assert(p[2] == 3);
    assert(p[3] == 0);
    delete x;
  }

  {
    // aligned_storage
    typedef std::aligned_storage
    <sizeof(__m128), std::alignment_of<__m128>::value>::type Pod128;

    typedef std::vector<Pod128, simd::allocator<Pod128> > Vector128;

    for (std::size_t i = 1; i != 10; ++i) {
      Vector128 x(i);
      assert(std::size_t(&x[0]) % 16 == 0);
    }

    {
      Vector128 x;
      for (std::size_t i = 0; i != 10; ++i) {
        x.push_back(Pod128());
        assert(std::size_t(&x[0]) % 16 == 0);
      }
    }

    {
      Vector128 x(10);
      for (std::size_t i = 0; i != x.size(); ++i) {
        _mm_store_ps(reinterpret_cast<float*>(&x[i]), _mm_set1_ps(i));
      }
      for (std::size_t i = 0; i != x.size(); ++i) {
        __m128 a = _mm_load_ps(reinterpret_cast<const float*>(&x[i]));
        const float* p = reinterpret_cast<const float*>(&a);
        for (std::size_t j = 0; j != 4; ++j) {
          assert(p[j] == i);
        }
      }
    }
  }


  {
    typedef std::vector<float, simd::allocator<float, 16> >
    AlignedFloatVector;
    AlignedFloatVector x;
    assert(x.empty());
    x.push_back(0);
    assert(std::size_t(&x[0]) % 16 == 0);
  }
  {
    typedef std::vector<float, simd::allocator<float, 32> >
    AlignedFloatVector;
    AlignedFloatVector x;
    assert(x.empty());
    x.push_back(0);
    assert(std::size_t(&x[0]) % 32 == 0);
  }

  // resize();
  {
    std::vector<float, simd::allocator<float> > x;
    x.resize(1024);
    assert(x.size() == 1024);
  }

  return 0;
}
