// -*- C++ -*-

/*!
  \file stlib/container/DummyVector.h
  \brief %DummyVector is always an empty vector.
*/

#if !defined(__container_DummyVector_h__)
#define __container_DummyVector_h__

#include <boost/config.hpp>

#include <iterator>
#include <memory>

namespace stlib
{
namespace container
{

/// A replacement for std::vector for when you don't want to store the elements.
/**
   This class has the same interface as std::vector.  However, the
   member functions do nothing, so the %vector is always empty. This
   is useful when you have a class that optionally stores a %vector of
   values, and you want to determine the storage options at
   compile-time.
*/
template<class _T, class _Allocator = std::allocator<_T> >
class DummyVector
{
public:

  typedef _T value_type;
  typedef _Allocator allocator_type;
  typedef typename std::conditional<std::is_void<value_type>::value, void,
                                    _T&>::type reference;
  typedef typename std::conditional<std::is_void<value_type>::value, void,
                                    _T const&>::type const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef _T* pointer;
  typedef _T const* const_pointer;
  typedef pointer iterator;
  typedef const_pointer const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

private:

  /// We need a dummy variable so that we can return a reference to an instance of the value type.
  value_type _dummyValue;
  
public:

  DummyVector()
  {
  }

  explicit
  DummyVector(const allocator_type& /*a*/)
  {
  }

  explicit
  DummyVector(size_type /*n*/)
  {
  }
  
  explicit
  DummyVector(size_type /*n*/, const allocator_type& /*a*/)
  {
  }
  
  DummyVector(size_type /*n*/, const_reference /*x*/)
  {
  }
    
  DummyVector(size_type /*n*/, const_reference /*x*/,
              const allocator_type& /*a*/)
  {
  }
    
  template<class _InputIterator>
  DummyVector(_InputIterator /*first*/, _InputIterator /*last*/)
  {
  }
  
  template<class _InputIterator>
  DummyVector(_InputIterator /*first*/, _InputIterator /*last*/,
              const allocator_type& /*a*/)
  {
  }
    
  DummyVector(std::initializer_list<value_type> /*il*/)
  {
  }

  DummyVector(std::initializer_list<value_type> /*il*/,
              const allocator_type& /*a*/)
  {
  }

  DummyVector(const DummyVector& /*x*/)
  {
  }
  
  DummyVector(const DummyVector& /*x*/, const allocator_type& /*a*/)
  {
  }
    
  DummyVector&
  operator=(const DummyVector& /*x*/)
  {
    return *this;
  }
    
  DummyVector(DummyVector&& /*x*/)
  {
  }
    
  DummyVector(DummyVector&& /*x*/, const allocator_type& /*a*/)
  {
  }
    
  DummyVector&
  operator=(DummyVector&& /*x*/)
  {
    return *this;
  }

  DummyVector&
  operator=(std::initializer_list<value_type> /*il*/)
  {
    return *this;
  }

  template<class _InputIterator>
  void
  assign(_InputIterator /*first*/, _InputIterator /*last*/)
  {
  }

  void
  assign(size_type /*n*/, const_reference /*u*/)
  {
  }
  
  void
  assign(std::initializer_list<value_type> /*il*/)
  {
  }
    
  iterator
  begin() BOOST_NOEXCEPT
  {
    return nullptr;
  }
    
  const_iterator
  begin() const BOOST_NOEXCEPT
  {
    return nullptr;
  }
    
  iterator
  end() BOOST_NOEXCEPT
  {
    return nullptr;
  }
    
  const_iterator
  end() const BOOST_NOEXCEPT
  {
    return nullptr;
  }
    
  size_type
  size() const BOOST_NOEXCEPT
  {
    return 0;
  }
    
  size_type
  capacity() const BOOST_NOEXCEPT
  {
    return 0;
  }
    
  bool
  empty() const BOOST_NOEXCEPT
  {
    return true;
  }

  void
  reserve(size_type /*n*/)
  {
  }
  
  void
  shrink_to_fit() BOOST_NOEXCEPT
  {
  }

  reference
  operator[](size_type /*n*/)
  {
    return _dummyValue;
  }
  
  const_reference
  operator[](size_type /*n*/) const
  {
    return _dummyValue;
  }
    
  reference
  at(size_type /*n*/)
  {
    return _dummyValue;
  }
    
  const_reference
  at(size_type /*n*/) const
  {
    return _dummyValue;
  }

  reference
  front()
  {
    return _dummyValue;
  }

  const_reference
  front() const
  {
    return _dummyValue;
  }

  reference
  back()
  {
    return _dummyValue;
  }

  const_reference
  back()  const
  {
    return _dummyValue;
  }

  void
  push_back(const_reference /*x*/)
  {
  }
    
  void
  push_back(value_type&& /*x*/)
  {
  }

  void
  pop_back()
  {
  }

  iterator
  insert(const_iterator /*position*/, const_reference /*x*/)
  {
    return nullptr;
  }
  
  iterator
  insert(const_iterator /*position*/, value_type&& /*x*/)
  {
    return nullptr;
  }

  iterator
  insert(const_iterator /*position*/, size_type /*n*/, const_reference /*x*/)
  {
    return nullptr;
  }

  template<class _InputIterator>
  iterator
  insert(const_iterator /*position*/, _InputIterator /*first*/,
         _InputIterator /*last*/)
  {
    return nullptr;
  }
  
  iterator
  insert(const_iterator /*position*/, std::initializer_list<value_type> /*il*/)
  {
    return nullptr;
  }

  iterator
  erase(const_iterator /*position*/)
  {
    return nullptr;
  }
  
  iterator
  erase(const_iterator /*first*/, const_iterator /*last*/)
  {
    return nullptr;
  }

    
  void
  clear() BOOST_NOEXCEPT
  {
  }

  void
  resize(size_type /*sz*/)
  {
  }
    
  void
  resize(size_type /*sz*/, const_reference /*x*/)
  {
  }

  void
  swap(DummyVector& /*other*/)
  {
  }
};


template<class _T, class _Allocator>
inline
bool
operator==(DummyVector<_T, _Allocator> const& /*a*/,
           DummyVector<_T, _Allocator> const& /*b*/)
{
  return true;
}


} // namespace container
} // namespace stlib

#endif
