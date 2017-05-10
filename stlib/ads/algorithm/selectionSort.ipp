// -*- C++ -*-

#if !defined(__ads_selectionSort_ipp__)
#error This file is an implementation detail of selectionSort.
#endif

namespace stlib
{
namespace ads
{


template<typename _RandomAccessIterator>
inline
void
selectionSort(_RandomAccessIterator first, _RandomAccessIterator const last)
{
  if (first + 1 >= last) {
    return;
  }

  _RandomAccessIterator min;
  while (first + 1 != last) {
    // Find the minimum element.
    min = first;
    for (_RandomAccessIterator i = min + 1; i != last; ++i) {
      if (*i < *min) {
        min = i;
      }
    }
    // Swap the first element in the remaining sequence with the minimum.
    std::swap(*first, *min);
    ++first;
  }
}


template<typename _RandomAccessIterator, typename _Compare>
inline
void
selectionSort(_RandomAccessIterator first, _RandomAccessIterator const last,
              _Compare compare)
{
  if (first + 1 >= last) {
    return;
  }

  _RandomAccessIterator min;
  while (first + 1 != last) {
    // Find the minimum element.
    min = first;
    for (_RandomAccessIterator i = min + 1; i != last; ++i) {
      if (compare(*i, *min)) {
        min = i;
      }
    }
    // Swap the first element in the remaining sequence with the minimum.
    std::swap(*first, *min);
    ++first;
  }
}


template<typename _RandomAccessIterator, typename _OutputIterator>
inline
void
selectionSortSeparateOutput(_RandomAccessIterator const first,
                            _RandomAccessIterator last,
                            _OutputIterator output)
{
  _RandomAccessIterator min;
  while (first != last) {
    // Find the minimum element.
    min = first;
    for (_RandomAccessIterator i = min + 1; i != last; ++i) {
      if (*i < *min) {
        min = i;
      }
    }
    // Write out the minimum element.
    *output++ = *min;
    // Remove the minimum element by overwriting with the last element.
    --last;
    *min = *last;
  }
}


template<typename _RandomAccessIterator, typename _OutputIterator,
         typename _Compare>
inline
void
selectionSortSeparateOutput(_RandomAccessIterator const first,
                            _RandomAccessIterator last,
                            _OutputIterator output,
                            _Compare compare)
{
  _RandomAccessIterator min;
  while (first != last) {
    // Find the minimum element.
    min = first;
    for (_RandomAccessIterator i = min + 1; i != last; ++i) {
      if (compare(*i, *min)) {
        min = i;
      }
    }
    // Write out the minimum element.
    *output++ = *min;
    // Remove the minimum element by overwriting with the last element.
    --last;
    *min = *last;
  }
}


} // namespace ads
} // namespace stlib
