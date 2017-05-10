// -*- C++ -*-

#if !defined(__Graph_ipp__)
#error This file is an implementation detail of the class Graph.
#endif

namespace stlib
{
namespace shortest_paths
{

//
// Mathematical operations
//

/*
template <typename VertexType>
inline
void
Graph<VertexType>::
add_vertex()
{
  // Resizing the vertex array would invalidate pointers.
  assert( _vertices.size() != _vertices.capacity() );
  VertexType v;
  _vertices.push_back( v );
}
*/

template <typename VertexType>
inline
void
Graph<VertexType>::
add_edge(int source_index, int target_index, weight_type weight)
{
  assert(0 <= source_index &&
         source_index < static_cast<int>(_vertices.size()) - 1 &&
         0 <= target_index &&
         target_index < static_cast<int>(_vertices.size()) - 1 &&
         weight >= 0);
  _edges.push_back(edge_type(&*_vertices.begin() + source_index,
                             &*_vertices.begin() + target_index,
                             weight));
}

template <typename VertexType>
inline
void
Graph<VertexType>::
initialize(const int source_index)
{
  // Initialize the data in each vertex.
  const vertex_iterator iter_end = _vertices.end();
  for (vertex_iterator iter = _vertices.begin();
       iter != iter_end;
       ++iter) {
    iter->initialize();
  }
  // Set the source vertex.
  _vertices[source_index].set_root();
}

template <typename VertexType>
template <typename Generator>
inline
void
Graph<VertexType>::
rectangular_grid(const size_type x_size, const size_type y_size,
                 Generator& edge_weight)
{
  // The number of vertices.
  const size_type num_vertices = x_size * y_size;
  // The number of edges.
  const size_type num_edges = 4 * num_vertices;

  // Allocate memory for the vertices and edges.
  reserve(num_vertices, num_edges);

  /*
  // Add the vertices.
  for ( int i = 0; i < num_vertices; ++i ) {
    add_vertex();
  }
  // Add the dummy vertex.
  add_vertex();
  */

  // Add the edges.
  _edges.clear();
  int source;
  for (size_type i = 0; i < x_size; ++i) {
    for (size_type j = 0; j < y_size; ++j) {
      source = i * y_size + j;
      // Horizontal edges.
      add_edge(source, i * y_size + (j + 1) % y_size, edge_weight());
      add_edge(source, i * y_size + (y_size + j - 1) % y_size,
               edge_weight());
      // Vertical edges.
      add_edge(source, ((i + 1) % x_size) * y_size + j, edge_weight());
      add_edge(source, ((x_size + i - 1) % x_size) * y_size + j,
               edge_weight());
    }
  }

  // Do any necessary building.
  build();
}

template <typename VertexType>
template <typename Generator>
inline
void
Graph<VertexType>::
dense(const size_type num_vertices, Generator& edge_weight)
{
  // The number of edges.
  const size_type num_edges = 2 * num_vertices * (num_vertices - 1);

  // Allocate memory for the vertices and edges.
  reserve(num_vertices, num_edges);

  /*
  // Add the vertices.
  for ( int i = 0; i < num_vertices; ++i ) {
    add_vertex();
  }
  // Add the dummy vertex.
  add_vertex();
  */

  // Add the edges.
  _edges.clear();
  for (size_type i = 0; i < num_vertices; ++i) {
    for (size_type j = 0; j < num_vertices; ++j) {
      if (i != j) {
        add_edge(i, j, edge_weight());
        add_edge(j, i, edge_weight());
      }
    }
  }

  // Do any necessary building.
  build();
}

template <typename VertexType>
template <typename Generator>
inline
void
Graph<VertexType>::
random(const size_type num_vertices,
       const size_type num_adjacent_edges_per_vertex,
       Generator& edge_weight)
{
  // The number of edges.
  const size_type num_edges = num_vertices * num_adjacent_edges_per_vertex;

  // Allocate memory for the vertices and edges.
  reserve(num_vertices, num_edges);

  /*
  // Add the vertices.
  for ( int i = 0; i < num_vertices; ++i ) {
    add_vertex();
  }
  // Add the dummy vertex.
  add_vertex();
  */

  //
  // Add the edges.
  //
  _edges.clear();
  // Make an array of the vertex indices.
  std::vector<int> vertex_indices(num_vertices, 0);
  for (int i = 0; i < num_vertices; ++i) {
    vertex_indices[i] = i;
  }
  for (int i = 0; i < num_adjacent_edges_per_vertex; ++i) {
    std::random_shuffle(vertex_indices.begin(), vertex_indices.end());
    for (int j = 0; j < num_vertices; ++j) {
      add_edge(vertex_indices[j], vertex_indices[(j + 1) % num_vertices],
               edge_weight());
    }
  }

  // Do any necessary building.
  build();
}


//
// Private member functions
//

template <typename VertexType>
inline
void
Graph<VertexType>::
reserve(const size_type num_vertices, const size_type num_edges)
{
  {
    // We need an extra vertex to indicate the end of the edges for the last
    // vertex.
    vertex_container temp(num_vertices + 1);
    _vertices.swap(temp);
  }
  {
    edge_container temp(num_edges);
    _edges.swap(temp);
  }
}

//
// Stream Output
//

template <typename VertexType>
inline
void
Graph<VertexType>::
put(std::ostream& out) const
{
  {
    // Write the vertices.
    assert(_vertices.size());
    vertex_const_iterator
    i,
    i_begin = _vertices.begin(),
    i_end = _vertices.end() - 1;
    for (i = i_begin; i != i_end; ++i) {
      out << i - i_begin << " " << *i << '\n';
    }
  }
  {
    // Write the edges.
    edge_const_iterator
    i,
    i_begin = _edges.begin(),
    i_end = _edges.end();
    for (i = i_begin; i != i_end; ++i) {
      out << i - i_begin << " " << *i << '\n';
    }
  }
}

//
// Equality
//

template <typename VertexType1, typename VertexType2>
bool operator==(const Graph<VertexType1>& x,
                const Graph<VertexType2>& y)
{
  if (x.vertices().size() != y.vertices().size()) {
    return false;
  }

  typename Graph<VertexType1>::vertex_const_iterator
  i = x.vertices().begin(),
  i_end = x.vertices().end() - 1;
  typename Graph<VertexType2>::vertex_const_iterator
  j = y.vertices().begin();
  for (; i < i_end; ++i, ++j) {
    if (i->distance() != j->distance()) {
      return false;
    }
  }

  return true;
}

//
// Stream Output
//

template <typename VertexType>
std::ostream&
operator<<(std::ostream& out, const Graph<VertexType>& graph)
{
  graph.put(out);
  return out;
}

} // namespace shortest_paths
}

