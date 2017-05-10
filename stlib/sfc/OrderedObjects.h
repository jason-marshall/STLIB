// -*- C++ -*-

#if !defined(__sfc_OrderedObjects_h__)
#define __sfc_OrderedObjects_h__

/**
  \file
  \brief Data structure for ordering objects with associated SFC codes.
*/

#include "stlib/ext/vector.h"

namespace stlib
{
namespace sfc
{

USING_STLIB_EXT_VECTOR_IO_OPERATORS;

/// Data structure for ordering objects with associated SFC codes.
class OrderedObjects
{
  //
  // Friends.
  //

  friend
  std::ostream&
  operator<<(std::ostream& out, OrderedObjects const& x);

private:
  /// The indices of the ordered objects.
  std::vector<std::size_t> _orderedIndices;

public:

  /// Copy the ordering.
  OrderedObjects const&
  operator=(OrderedObjects const& other)
  {
    _orderedIndices = other._orderedIndices;
    return *this;
  }
  
  /// Set the ordering.
  template<typename _Code>
  void
  set(std::vector<std::pair<_Code, std::size_t> > const& codeIndexPairs)
  {
    _orderedIndices.resize(codeIndexPairs.size());
    for (std::size_t i = 0; i != codeIndexPairs.size(); ++i) {
      _orderedIndices[i] = codeIndexPairs[i].second;
    }
  }

  /// Clear the ordering information.
  void
  clear()
  {
    _orderedIndices.clear();
  }

  /// Shrink the capacity to match the size.
  void
  shrink_to_fit()
  {
    _orderedIndices.shrink_to_fit();
  }

  /// Check the validity of the ordering.
  void
  checkValidity() const
  {
    // Check that the indices are a permutation of the natural numbers.
    std::vector<std::size_t> indices(_orderedIndices);
    std::sort(indices.begin(), indices.end());
    for (std::size_t i = 0; i != indices.size(); ++i) {
      assert(indices[i] == i);
    }
  }

  /// Return the required storage (in bytes) for the ordered indices.
  std::size_t
  storage() const
  {
    return _orderedIndices.size() * sizeof(std::size_t);
  }

  /// Return true if the other is equal.
  bool
  operator==(OrderedObjects const& other) const
  {
    return _orderedIndices == other._orderedIndices;
  }

  /// Order the sequence to match that of the ordered objects.
  template<typename _RandomAccessIterator>
  void
  order(_RandomAccessIterator begin, _RandomAccessIterator end) const
  {
    typedef typename std::iterator_traits<_RandomAccessIterator>::value_type
      value_type;
    assert(std::size_t(std::distance(begin, end)) == _orderedIndices.size());
    std::vector<std::size_t> mapping(_orderedIndices.size());
    for (std::size_t i = 0; i != mapping.size(); ++i) {
      mapping[_orderedIndices[i]] = i;
    }
    std::vector<value_type> const objects(begin, end);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      begin[mapping[i]] = objects[i];
    }
  }
  
  /// Map indices from the ordered objects to the original objects.
  template<typename _ForwardIterator>
  void
  mapToOriginalIndices(_ForwardIterator begin, _ForwardIterator end) const
  {
    for ( ; begin != end; ++begin) {
      *begin = _orderedIndices[*begin];
    }
  }

  /// Restore the original order of the objects.
  template<typename _RandomAccessIterator>
  void
  restore(_RandomAccessIterator begin, _RandomAccessIterator end) const
  {
    typedef typename std::iterator_traits<_RandomAccessIterator>::value_type
      value_type;
    assert(std::size_t(std::distance(begin, end)) == _orderedIndices.size());
    std::vector<value_type> const objects(begin, end);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      begin[_orderedIndices[i]] = objects[i];
    }
  }
};


/// Write information about the object ordering.
inline
std::ostream&
operator<<(std::ostream& out, OrderedObjects const& x)
{
  return out << "ordered object indices:\n" << x._orderedIndices;
}


} // namespace sfc
} // namespace stlib

#endif
