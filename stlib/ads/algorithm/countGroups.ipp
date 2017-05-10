// -*- C++ -*-

#if !defined(__ads_algorithm_countGroups_ipp__)
#error This file is an implementation detail of countGroups.
#endif

namespace stlib
{
namespace ads
{

template<typename _ForwardIterator>
inline
std::size_t
countGroups(_ForwardIterator first, _ForwardIterator last)
{
  typedef typename std::iterator_traits<_ForwardIterator>::value_type ValueType;
  return countGroups(first, last, std::equal_to<ValueType>());
}

template<typename _ForwardIterator, typename _BinaryPredicate>
inline
std::size_t
countGroups(_ForwardIterator first, _ForwardIterator last,
            _BinaryPredicate equal)
{
  // Dispense with the trivial case.
  if (first == last) {
    return 0;
  }
  std::size_t count = 1;
  _ForwardIterator current = first;
  for (++first; first != last; ++first) {
    if (! equal(*current, *first)) {
      ++count;
      current = first;
    }
  }
  return count;
}

} // namespace ads
} // namespace stlib
