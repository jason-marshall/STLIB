// -*- C++ -*-

/*!
  \file Graph.h
  \brief Implements a class for a weighted, directed graph.
*/

#if !defined(__Graph_h__)
#define __Graph_h__

#include "stlib/shortest_paths/VertexCompare.h"
#include "stlib/shortest_paths/Edge.h"

#include <iosfwd>
#include <vector>
#include <algorithm>

#include <cassert>

namespace stlib
{
namespace shortest_paths
{

//! A weighted, directed graph.
template <typename VertexType>
class Graph
{
public:

  //
  // Public types.
  //

  //! The vertex type.
  typedef VertexType vertex_type;
  //! The weight type.
  typedef typename vertex_type::weight_type weight_type;
  //! The size type is a signed integer.
  typedef int size_type;

  //! Vertex container.
  typedef std::vector< vertex_type > vertex_container;
  //! Vertex iterator.
  typedef typename vertex_container::iterator vertex_iterator;
  //! Vertex const iterator.
  typedef typename vertex_container::const_iterator vertex_const_iterator;

  //! Edge.
  typedef Edge< vertex_type > edge_type;
  //! Edge container.
  typedef std::vector< edge_type > edge_container;
  //! Edge iterator.
  typedef typename edge_container::iterator edge_iterator;
  //! Edge const iterator.
  typedef typename edge_container::const_iterator edge_const_iterator;

private:

  //
  // Not implemented.
  //

  //! Copy constructor not implemented.
  Graph(const Graph&);

  //! Assignment operator not implemented.
  Graph&
  operator=(const Graph&);

private:

  //
  // Member data.
  //

  vertex_container _vertices;

  edge_container _edges;

  VertexCompare<vertex_type*> _vertex_compare;

public:

  //------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{

  //! Default constructor.
  Graph() :
    _vertices(),
    _edges() {}

  //! Construct with room for specified number of vertices and edges.
  Graph(int num_vertices, int num_edges)
  {
    reserve(num_vertices, num_edges);
  }

  //! Destructor.
  virtual
  ~Graph() {}

  //@}
  //------------------------------------------------------------------------
  //! \name Accessors.
  //@{

  //! Return the vertices.
  const vertex_container&
  vertices() const
  {
    return _vertices;
  }

  //! Return the edges.
  const edge_container&
  edges() const
  {
    return _edges;
  }

  //! The vertex comparison functor.
  const VertexCompare<vertex_type*>&
  vertex_compare() const
  {
    return _vertex_compare;
  }

  //@}
  //------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  //! Return the vertices.
  vertex_container&
  vertices()
  {
    return _vertices;
  }

  //! Return the edges.
  edge_container&
  edges()
  {
    return _edges;
  }

  //@}
  //------------------------------------------------------------------------
  //! \name Mathematical operations.
  //@{

  //! Add a vertex to the graph.
  //  void add_vertex();

  //! Add an edge to the graph.
  void
  add_edge(const int source_index, const int target_index,
           const weight_type weight);

  //! Clear the graph.  Remove all vertices and edges.
  /*
    void clear()
  {
    _vertices.clear();
    _edges.clear();
  }
  */

  //! Make a rectangular grid.
  /*!
    The vertices are lattice points.  Each vertex is doubly
    connected to its adjacent vertices.

    \param x_size:  The number of vertices in the x direction.
    \param y_size: The number of vertices in the y direction.
    \param edge_weight A functional with no arguments giving the edge weight.
  */
  template <typename Generator>
  void
  rectangular_grid(const size_type x_size, const size_type y_size,
                   Generator& edge_weight);

  //! Make a dense graph.
  /*!
    Each vertex is doubly connected to every other vertex.
    \param num_vertices The number of vertices in the graph.
    \param edge_weight A functional with no arguments giving the edge weight.
  */
  template <typename Generator>
  void
  dense(const size_type num_vertices, Generator& edge_weight);

  //! Make a graph with random edges.
  /*!
    \param num_vertices The number of vertices.
    \param num_adjacent_edges_per_vertex Each vertex has this many
    adjacent edges.
    \param edge_weight A functional with no arguments giving the edge weight.
  */
  template <typename Generator>
  void
  random(const size_type num_vertices,
         const size_type num_adjacent_edges_per_vertex,
         Generator& edge_weight);

  //@}
  //------------------------------------------------------------------------
  //! \name File I/O.
  //@{

  //! Write the graph to the stream.
  virtual
  void
  put(std::ostream& out) const;

  //@}

protected:

  //! Do any necessary building after the vertices and edges have been added.
  virtual
  void
  build() {}

  //! Initialize for a shortest path computation.
  /*!
    \param source_index is the index of the root vertex of the tree.
  */
  void
  initialize(const int source_index);

  //! Reserve memory for the specified number of vertices and edges.
  void
  reserve(const size_type num_vertices, const size_type num_edges);
};

//
// Equality
//

//! Return true if the vertices have the same distance.
/*! \relates Graph */
template <typename VertexType1, typename VertexType2>
bool
operator==(const Graph<VertexType1>& x, const Graph<VertexType2>& y);

//
// Stream Output
//

//! Write a graph.
/*! \relates Graph */
template <typename VertexType>
std::ostream&
operator<<(std::ostream& out, const Graph<VertexType>& graph);

} // namespace shortest_paths
}

#define __Graph_ipp__
#include "stlib/shortest_paths/Graph.ipp"
#undef __Graph_ipp__

#endif
