// -*- C++ -*-

#if !defined(__VertexDijkstra_ipp__)
#error This file is an implementation detail of the class VertexDijkstra.
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
VertexDijkstra<WeightType>&
VertexDijkstra<WeightType>::
operator=(const VertexDijkstra& rhs)
{
  if (&rhs != this) {
    base_type::operator=(rhs);
    _adjacent_edges = rhs._adjacent_edges;
    _heap_ptr = rhs._heap_ptr;
  }
  return *this;
}

} // namespace shortest_paths
}

