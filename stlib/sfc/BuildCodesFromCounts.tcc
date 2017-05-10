// -*- C++ -*-

#if !defined(__sfc_BuildCodesFromCounts_tcc__)
#error This file is an implementation detail of BuildCodesFromCounts.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Traits>
inline
BuildCodesFromCounts<_Traits>::
BuildCodesFromCounts(BlockCode<_Traits> const& blockCode,
                    std::vector<Pair> const& pairs,
                    std::size_t const maxElementsPerCell) :
  _blockCode(blockCode),
  _codes(pairs.size()),
  _partialSum(pairs.size() + 1),
  _maxElementsPerCell(maxElementsPerCell),
  _maxLevel(pairs.empty() ? 0 : blockCode.level(pairs.front().first))
{
  // Copy the codes into a vector.
  for (std::size_t i = 0; i != _codes.size(); ++i) {
    _codes[i] = pairs[i].first;
  }

  // Make a vector of the partial sums of the counts.
  // _partialSum[n + 1] - _partialSum[m] = \sum_{i=m}^n pairs[i].second
  _partialSum.front() = 0;
  for (std::size_t i = 0; i != pairs.size(); ++i) {
#ifdef STLIB_DEBUG
    assert(pairs[i].second > 0);
#endif
    _partialSum[i + 1] = pairs[i].second;
  }
  std::partial_sum(_partialSum.begin(), _partialSum.end(), _partialSum.begin());

#ifdef STLIB_DEBUG
  // The input codes must be at the same level of refinement.
  if (! _codes.empty()) {
    auto const level = _blockCode.level(_codes.front());
    for (auto code : _codes) {
      assert(_blockCode.level(code) == level);
    }
  }
#endif
}


template<typename _Traits>
inline
void
BuildCodesFromCounts<_Traits>::
operator()(std::vector<typename _Traits::Code>* outputCodes) const
{
  outputCodes->clear();
  if (! _codes.empty()) {
    _build(0, _codes.size(), 0, outputCodes);
  }
}


template<typename _Traits>
inline
void
BuildCodesFromCounts<_Traits>::
_build(std::size_t begin, std::size_t const end, Code code,
       std::vector<typename _Traits::Code>* outputCodes) const
{
  // If we have reached the maximum level, or if we are down to a single input
  // cell, or if the number of objects is no more than the allowed maximum,
  // record the cell.
  if (_blockCode.level(code) == _maxLevel ||
      begin + 1 == end ||
      _partialSum[end] - _partialSum[begin] <= _maxElementsPerCell) {
    outputCodes->push_back(code);
    return;
  }

  // We just increment the level to get the first child code.
  ++code;
  Code next = _blockCode.next(code);
  for (std::size_t i = 0; i != NumChildren; ++i) {
    // Find the range of objects covered by the child.
    std::size_t const length =
      std::upper_bound(&_codes[begin], &_codes[end], _blockCode.location(next))
      - &_codes[begin];
    // Recurse.
    if (length != 0) {
      _build(begin, begin + length, code, outputCodes);
    }
    // Move to the next child.
    begin += length;
    code = next;
    next = _blockCode.next(next);
  }
}


} // namespace sfc
}
