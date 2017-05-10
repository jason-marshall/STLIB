// -*- C++ -*-

#if !defined(__lorg_sort_tcc__)
#error This file is an implementation detail of sort.
#endif

namespace stlib
{
namespace lorg
{

//--------------------------------------------------------------------------
// Constructors etc.

template<typename _Integer, typename _T>
inline
RciSort<_Integer, _T>::
RciSort(std::vector<Value>* pairs, const int digits) :
  _pairs(pairs),
  _buffer(pairs->size()),
  _insertIterators(),
  // Round up to a multiple of the RadixBits.
  _digits((digits + RadixBits - 1) / RadixBits* RadixBits)
{
  assert(0 <= _digits && _digits <= std::numeric_limits<_Integer>::digits);
}

// CONTINUE: Use threading to improve performance. After sorting by one radix,
// the 256 intervals could be partitioned amongst the available processors for
// the remaining sorting.
template<typename _Integer, typename _T>
inline
void
RciSort<_Integer, _T>::
_sort(const std::size_t begin, const std::size_t end, int shift)
{
  // Count the number of elements with each byte-valued key.
  std::array<std::size_t, Radix> counts;
  std::fill(counts.begin(), counts.end(), 0);
  for (std::size_t i = begin; i != end; ++i) {
    // Extract the radix, which is a byte, by left shifting and masking.
    ++counts[((*_pairs)[i].first >> shift) & Mask];
  }
  // Set iterators for inserting into the buffer.
  _insertIterators[0] = &_buffer[0];

  for (std::size_t i = 1; i != _insertIterators.size(); ++i) {
    _insertIterators[i] = _insertIterators[i - 1] + counts[i - 1];
  }
  // Sort according to the key by using the insert iterators.
  for (std::size_t i = begin; i != end; ++i) {
    *_insertIterators[((*_pairs)[i].first >> shift) & Mask]++ = (*_pairs)[i];
  }
  // memcpy is faster than std::copy.
  memcpy(&(*_pairs)[begin], &_buffer[0], (end - begin) * sizeof(Value));
  // Stop recursion if we sorted according to each radix.
  if (shift == 0) {
    return;
  }
  // Otherwise continue with depth-first recursion.
  shift -= RadixBits;
  std::size_t offset = 0;
  for (std::size_t i = 0; i != Radix; ++i) {
    if (counts[i] > 64) {
      _sort(begin + offset, begin + offset + counts[i], shift);
    }
    else if (counts[i] > 1) {
      // Note: Using a comparison functor is slightly slower.
      ads::insertion_sort(&(*_pairs)[begin + offset],
                          &(*_pairs)[begin + offset + counts[i]]);
    }
    offset += counts[i];
  }
}

} // namespace lorg
}
