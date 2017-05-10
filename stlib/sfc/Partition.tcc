// -*- C++ -*-

#if !defined(__sfc_Partition_tcc__)
#error This file is an implementation detail of Partition.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Traits>
template<typename _Cell, template<typename> class _Order>
inline
void
Partition<_Traits>::
operator()(NonOverlappingCells<_Traits, _Cell, true, _Order> const& cells)
{
  // The weights are the numbers of objects.
  std::vector<std::size_t> weights(cells.size());
  for (std::size_t i = 0; i != weights.size(); ++i) {
    weights[i] = cells.delimiter(i + 1) - cells.delimiter(i);
  }
  // Determine a fair partitioning of the cells.
  const std::size_t numParts = delimiters.size() - 1;
  std::vector<std::size_t> indices;
  numerical::computePartitions(weights, numParts, std::back_inserter(indices));
  assert(indices.size() == delimiters.size());
  // Convert to codes.
  delimiters.front() = 0;
  for (std::size_t i = 1; i != delimiters.size() - 1; ++i) {
    // Note that we convert to locations. For LocationCode's this has no effect,
    // but for BlockCode's it masks out the level bits. Using locations 
    // allows one to search for cells at any level in the sequence of 
    // delimiters.
    delimiters[i] = cells.grid().location(cells.code(indices[i]));
  }
  // The code delimiters are always terminated with the guard code.
  delimiters.back() = _Traits::GuardCode;
}


} // namespace sfc
} // namespace stlib
