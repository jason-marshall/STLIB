// -*- C++ -*-

#if !defined(__VertexMCCSimple_ipp__)
#error This file is an implementation detail of the class VertexMCCSimple.
#endif

namespace stlib
{
namespace shortest_paths
{

//
// Constructors, Destructor.
//

template <typename WeightType>
inline
VertexMCCSimple<WeightType>&
VertexMCCSimple<WeightType>::
operator=(const VertexMCCSimple& rhs)
{
  if (this != &rhs) {
    base_type::operator=(rhs);
    _adjacent_edges = rhs._adjacent_edges;
    _min_incident_edge_weight = rhs._min_incident_edge_weight;
  }
  return *this;
}

//
// Mathematical operations
//

template <typename WeightType>
template <typename OutputIterator>
inline
OutputIterator
VertexMCCSimple<WeightType>::
label_adjacent(OutputIterator unlabeled_neighbors)
{
  const edge_type* iter = _adjacent_edges;
  const edge_type* iter_end = (this + 1)->_adjacent_edges;
  VertexMCCSimple* adjacent;
  for (; iter != iter_end; ++iter) {
    adjacent = iter->vertex();
    if (adjacent->status() != KNOWN) {
      if (adjacent->status() == UNLABELED) {
        *unlabeled_neighbors = adjacent;
        ++unlabeled_neighbors;
      }
      adjacent->label(*this, iter->weight());
    }
  }
  return unlabeled_neighbors;
}

} // namespace shortest_paths
}

