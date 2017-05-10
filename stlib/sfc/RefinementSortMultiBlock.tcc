// -*- C++ -*-

#if !defined(__sfc_RefinementSortMultiBlock_tcc__)
#error This file is an implementation detail of RefinementSortMultiBlock.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Traits>
inline
RefinementSortMultiBlock<_Traits>::
RefinementSortMultiBlock(BlockCode<_Traits> const& blockCode, std::vector<Pair>* pairs,
                   std::size_t const maxElementsPerCell) :
  _blockCode(blockCode),
  _pairs(*pairs),
  _maxElementsPerCell(maxElementsPerCell),
  _buffer(pairs->size()),
  _insertIterators()
{
}


template<typename _Traits>
inline
void
RefinementSortMultiBlock<_Traits>::
operator()()
{
  // Because the termination criteria are applied at the end of _sort(),
  // we need to check them before calling at the top level.
  if (_blockCode.numLevels() == 0) {
    return;
  }
  if (_pairs.size() <= _maxElementsPerCell) {
    if (_pairs.size()) {
      _setLevel(0, _pairs.size(), 0);
    }
    return;
  }

  // Erase the level bits so they don't interfere with the multi-block radix.
  Code const mask = Code(-1) << _blockCode.levelBits();
  for (std::size_t i = 0; i != _pairs.size(); ++i) {
    _pairs[i].first &= mask;
  }

  // Sort.
  _sort(0, _pairs.size(), 0);
}


template<typename _Traits>
inline
void
RefinementSortMultiBlock<_Traits>::
_sort(std::size_t const begin, std::size_t const end, std::size_t const level)
{
  // The shift for extracting the radix.
  int const shift = Dimension * (_blockCode.numLevels() - level) +
    _blockCode.levelBits() - RadixBits;
  // Count the number of elements with each key.
  std::array<std::size_t, Radix> counts;
  std::fill(counts.begin(), counts.end(), 0);
  for (std::size_t i = begin; i != end; ++i) {
    // Extract the radix by left shifting and masking.
    ++counts[(_pairs[i].first >> shift) & Mask];
  }

  // Accumulate the counts.
  std::array<std::size_t, Radix + 1> delimiters;
  delimiters[0] = 0;
  std::partial_sum(counts.begin(), counts.end(), delimiters.begin() + 1);

  // Set iterators for inserting into the buffer.
  _insertIterators[0] = &_buffer[0];
  for (std::size_t i = 1; i != _insertIterators.size(); ++i) {
    _insertIterators[i] = _insertIterators[i - 1] + counts[i - 1];
  }

  // Sort according to the key by using the insert iterators.
  for (std::size_t i = begin; i != end; ++i) {
    *_insertIterators[(_pairs[i].first >> shift) & Mask]++ = _pairs[i];
  }
  // memcpy is faster than std::copy.
  memcpy(&_pairs[begin], &_buffer[0], (end - begin) * sizeof(Pair));

  _setLevelOrRecurse(begin, level + 1,
                     std::size_t(1) << (Dimension * (NumBlocks - 1)),
                     &delimiters[0]);
}


template<typename _Traits>
inline
void
RefinementSortMultiBlock<_Traits>::
_setLevelOrRecurse(std::size_t begin, std::size_t level,
                   std::size_t const stride, std::size_t const* delimiters)
{
  // When we reach the highest level of refinement, we are done.
  if (level == _blockCode.numLevels()) {
    // Set the level.
    std::size_t const count = delimiters[NumOrthants * stride] - delimiters[0];
    if (count) {
      _setLevel(begin, begin + count, level);
    }
    return;
  }

  if (stride == 1) {
    for (std::size_t i = 0; i != NumOrthants; ++i) {
      std::size_t const count =
        delimiters[(i + 1) * stride] - delimiters[i * stride];
      // If the cell has been sufficiently refined.
      if (count <= _maxElementsPerCell) {
        // Set the level.
        if (count) {
          _setLevel(begin, begin + count, level);
        }
      }
      else {
        // Otherwise, recurse.
        _sort(begin, begin + count, level);
      }
      begin += count;
    }
  }
  else {
    for (std::size_t i = 0; i != NumOrthants; ++i) {
      std::size_t count = delimiters[(i + 1) * stride] - delimiters[i * stride];
      // If the cell has been sufficiently refined.
      if (count <= _maxElementsPerCell) {
        // Set the level.
        if (count) {
          _setLevel(begin, begin + count, level);
        }
      }
      else {
        // Otherwise, recurse.
        _setLevelOrRecurse(begin, level + 1, stride >> Dimension,
                           &delimiters[i * stride]);
      }
      begin += count;
    }
  }
}


// Convert to block codes at the specified level.
template<typename _Traits>
inline
void
RefinementSortMultiBlock<_Traits>::
_setLevel(std::size_t const begin, std::size_t const end,
          std::size_t const level)
{
  // The mask erases the location code for higher levels and the level bits.
  Code const mask = Code(-1) <<
  (Dimension * (_blockCode.numLevels() - level) + _blockCode.levelBits());
  // Set the level.
  for (std::size_t i = begin; i != end; ++i) {
    _pairs[i].first = (_pairs[i].first & mask) | level;
  }
}


template<typename _Traits, typename _Object>
inline
void
refinementSortMultiBlock
(BlockCode<_Traits> const& blockCode,
 std::vector<_Object>* objects,
 std::vector<std::pair<typename _Traits::Code, std::size_t> >*
 codeIndexPairs,
 std::size_t const maxElementsPerCell)
{
  typedef typename _Traits::BBox BBox;

  // Allocate memory for the code/index pairs.
  codeIndexPairs->resize(objects->size());
  // Calculate the codes.
  for (std::size_t i = 0; i != codeIndexPairs->size(); ++i) {
    (*codeIndexPairs)[i].first =
      blockCode.code(centroid(geom::specificBBox<BBox>((*objects)[i])));
    (*codeIndexPairs)[i].second = i;
  }
  // Refine and sort by the codes.
  refinementSortMultiBlock(blockCode, codeIndexPairs, maxElementsPerCell);
  // Set the sorted objects.
  {
    std::vector<_Object> obj(codeIndexPairs->size());
    for (std::size_t i = 0; i != codeIndexPairs->size(); ++i) {
      obj[i] = (*objects)[(*codeIndexPairs)[i].second];
    }
    objects->swap(obj);
  }
}


} // namespace sfc
} // namespace stlib
