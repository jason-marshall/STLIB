// -*- C++ -*-

#ifndef stlib_simd_allocator_h
#define stlib_simd_allocator_h

#include "stlib/simd/align.h"

#include <xmmintrin.h>
#include <memory>

#include <cassert>

namespace stlib
{
namespace simd
{

/// Allocator that uses _mm_malloc and _mm_free for aligned storage.
template<typename _T, std::size_t _Alignment = Alignment>
class allocator : public std::allocator<_T>
{
  //
  // Private types.
  //
private:
  typedef std::allocator<_T> Base;

public:

  template<typename _T2>
  struct rebind {
    typedef allocator<_T2, _Alignment> other;
  };

  //
  // Use the sythesized assignment operator, and destructor.
  //

  /// Default construtor.
  allocator() :
    Base()
  {
  }

  /// Copy constructor.
  allocator(const allocator& other) :
    Base(other)
  {
  }

  /// Copy constructor from another type.
  template<typename _T2>
  allocator(const allocator<_T2, _Alignment>& other) :
    Base(other)
  {
  }

  /// Allocate aligned memory.
  typename Base::pointer
  allocate(typename Base::size_type n, const void* = 0)
  {
    void* p = _mm_malloc(n * sizeof(typename Base::value_type), _Alignment);
#ifdef STLIB_DEBUG
    assert(std::size_t(p) % _Alignment == 0);
#endif
    return reinterpret_cast<typename Base::pointer>(p);
  }

  /// Deallocate memory.
  void
  deallocate(typename Base::pointer p, typename Base::size_type)
  {
    _mm_free(p);
  }

  /// Alligned allocation with initialization.
  void
  construct(typename Base::pointer p, typename Base::const_reference x)
  {
    new(p) typename Base::value_type(x);
  }

  /// Destroy the pointee by calling the destructor.
  void
  destroy(typename Base::pointer p)
  {
    p->~_T();
  }
};

}
}


#endif // stlib_simd_allocator_h
