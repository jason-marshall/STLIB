// -*- C++ -*-

#if !defined(__GraphDijkstra_ipp__)
#error This file is an implementation detail of the class GraphDijkstra.
#endif

namespace stlib
{
namespace shortest_paths
{

//
// Mathematical operations
//

template <typename WeightType, typename HeapType>
inline
void
GraphDijkstra<WeightType, HeapType>::
build()
{
  // Allocate memory for the half edges.
  {
    half_edge_container temp(edges().size());
    _half_edges.swap(temp);
    _half_edges.clear();
  }

  // Sort the edges by source vertex.
  EdgeSourceCompare<edge_type> comp;
  std::sort(edges().begin(), edges().end(), comp);

  // Add the half edges.
  edge_const_iterator edge_iter = edges().begin();
  edge_const_iterator edge_end = edges().end();
  vertex_iterator vert_iter = vertices().begin();
  const vertex_iterator vert_end = vertices().end();
  for (; vert_iter != vert_end; ++vert_iter) {
    vert_iter->set_adjacent_edges(&*_half_edges.end());
    while (edge_iter != edge_end &&
           edge_iter->source() == &*vert_iter) {
      _half_edges.push_back(half_edge_type(edge_iter->target(),
                                           edge_iter->weight()));
      ++edge_iter;
    }
  }

  // Clear the edges.
  {
    edge_container temp;
    edges().swap(temp);
  }
}

template <typename WeightType, typename HeapType>
inline
void
GraphDijkstra<WeightType, HeapType>::
initialize(const int source_index)
{
  // Initialize the data in each vertex.
  vertex_iterator
  iter = vertices().begin(),
  iter_end = vertices().end();
  for (; iter != iter_end; ++iter) {
    iter->initialize();
  }

  // Set the source vertex to known.
  vertex_type& source = vertices()[source_index];
  source.set_status(KNOWN);
  source.set_distance(0);
  source.set_predecessor(0);
}

template <typename WeightType, typename HeapType>
inline
void
GraphDijkstra<WeightType, HeapType>::
dijkstra(const int root_vertex_index)
{
  // Initialize the graph.
  initialize(root_vertex_index);

  // The heap of labeled unknown vertices.
  heap_type labeled;
  // Label the adjacent neighbors of the root vertex.
  label_adjacent(labeled, &vertices()[root_vertex_index]);

  // All vertices are known when there are no labeled vertices left.
  // Loop while there are labeled vertices left.
  vertex_type* min_vertex;
  while (labeled.size()) {
    // The labeled vertex with minimum distance becomes known.
    min_vertex = labeled.top();
    labeled.pop();
    min_vertex->set_status(KNOWN);
    // Label the adjacent neighbors of the known vertex.
    label_adjacent(labeled, min_vertex);
  }
}

template <typename WeightType, typename HeapType>
inline
void
GraphDijkstra<WeightType, HeapType>::
label(heap_type& heap, vertex_type& vertex, const vertex_type& known_vertex,
      weight_type edge_weight)
{
  if (vertex.status() == UNLABELED) {
    vertex.set_status(LABELED);
    vertex.set_distance(known_vertex.distance() + edge_weight);
    vertex.set_predecessor(&known_vertex);
    heap.push(&vertex);
  }
  else { // _status == LABELED
    weight_type new_distance = known_vertex.distance() + edge_weight;
    if (new_distance < vertex.distance()) {
      vertex.set_distance(new_distance);
      vertex.set_predecessor(&known_vertex);
      heap.decrease(vertex.heap_ptr());
    }
  }
}

template <typename WeightType, typename HeapType>
inline
void
GraphDijkstra<WeightType, HeapType>::
label_adjacent(heap_type& heap, const vertex_type* known_vertex)
{
  vertex_type* adjacent;
  const half_edge_type* iter(known_vertex->adjacent_edges());
  const half_edge_type* iter_end((known_vertex + 1)->adjacent_edges());
  // Loop over the adjacent edges.
  for (; iter != iter_end; ++iter) {
    adjacent = static_cast<vertex_type*>(iter->vertex());
    if (adjacent->status() != KNOWN) {
      label(heap, *adjacent, *known_vertex, iter->weight());
    }
  }
}

} // namespace shortest_paths
}

