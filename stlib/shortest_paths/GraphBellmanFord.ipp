// -*- C++ -*-

#if !defined(__GraphBellmanFord_ipp__)
#error This file is an implementation detail of the class GraphBellmanFord.
#endif

namespace stlib
{
namespace shortest_paths
{

//
// Mathematical operations
//

template <typename WeightType>
inline
void
GraphBellmanFord<WeightType>::
bellman_ford(const int root_vertex_index)
{
  // Initialize the graph.
  initialize(root_vertex_index);

  edge_const_iterator
  edge_iter,
  edge_end = edges().end();
  bool changed = true;
  while (changed) {
    changed = false;
    // Loop over the edges.
    for (edge_iter = edges().begin(); edge_iter != edge_end; ++edge_iter) {
      // Relax along the edge.
      changed = edge_iter->relax() || changed;
    }
  }
}

} // namespace shortest_paths
}

