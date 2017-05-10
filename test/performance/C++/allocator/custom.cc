// -*- C++ -*-

#include <iostream>
#include <new>
#include <vector>

#include <cstdlib>

template<typename _T>
class new_allocator
{
public:
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef _T* pointer;
  typedef const _T* const_pointer;
  typedef _T& reference;
  typedef const _T& const_reference;
  typedef _T value_type;

  template<typename _T1>
  struct rebind {
    typedef new_allocator<_T1> other;
  };

  new_allocator() throw() {}

  new_allocator(const new_allocator&) throw() {}

  template<typename _T1>
  new_allocator(const new_allocator<_T1>&) throw() {}

  ~new_allocator() throw() {}

  pointer
  address(reference x) const
  {
    return &x;
  }

  const_pointer
  address(const_reference x) const
  {
    return &x;
  }

  // NB: n is permitted to be 0.  The C++ standard says nothing
  // about what the return value is when n == 0.
  pointer
  allocate(size_type n, const void* = 0)
  {
    if (n > this->max_size()) {
      throw std::bad_alloc();
    }
    //return static_cast<_T*>(::operator new(n * sizeof(_T)));
    //return new _T[n];
    return static_cast<_T*>(malloc(n * sizeof(_T)));
  }

  // p is not permitted to be a null pointer.
  void
  deallocate(pointer p, size_type)
  {
    //::operator delete(p);
    //delete[] p;
    free(p);
  }

  size_type
  max_size() const throw()
  {
    return std::size_t(-1) / sizeof(_T);
  }

  // _GLIBCXX_RESOLVE_LIB_DEFECTS
  // 402. wrong new expression in [some_] allocator::construct
  void
  construct(pointer p, const _T& val)
  {
    ::new(p) _T(val);
  }

  void
  destroy(pointer p)
  {
    p->~_T();
  }
};

template<typename _T>
inline
bool
operator==(const new_allocator<_T>&, const new_allocator<_T>&)
{
  return true;
}

template<typename _T>
inline
bool
operator!=(const new_allocator<_T>&, const new_allocator<_T>&)
{
  return false;
}


int
main()
{
  {
    std::vector<char, new_allocator<char> > a(10000000);
  }
  getchar();

  return 0;
}
