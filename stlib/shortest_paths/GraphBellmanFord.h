// -*- C++ -*-

/*!
  \file GraphBellmanFord.h
  \brief Implements a class for a weighted, directed graph.
*/

#if !defined(__GraphBellmanFord_h__)
#define __GraphBellmanFord_h__

#include "stlib/shortest_paths/Graph.h"
#include "stlib/shortest_paths/Vertex.h"

namespace stlib
{
namespace shortest_paths
{

//! A weighted, directed graph.
template <typename WeightType>
class GraphBellmanFord :
  public Graph< Vertex<WeightType> >
{
private:

  typedef Graph< Vertex<WeightType> > base_type;

protected:

  //
  // Protected types.
  //

  //! Vertex container.
  typedef typename base_type::vertex_container vertex_container;
  //! Vertex iterator.
  typedef typename base_type::vertex_iterator vertex_iterator;
  //! Vertex const iterator.
  typedef typename base_type::vertex_const_iterator vertex_const_iterator;

  //! Edge.
  typedef typename base_type::edge_type edge_type;
  //! Edge container.
  typedef typename base_type::edge_container edge_container;
  //! Edge iterator.
  typedef typename base_type::edge_iterator edge_iterator;
  //! Edge const iterator.
  typedef typename base_type::edge_const_iterator edge_const_iterator;

public:

  //
  // Public types.
  //

  //! The vertex type.
  typedef typename base_type::vertex_type vertex_type;
  //! The weight type.
  typedef typename vertex_type::weight_type weight_type;
  //! The size type is a signed integer.
  typedef typename base_type::size_type size_type;

private:

  //
  // Not implemented.
  //

  //! Copy constructor not implemented.
  GraphBellmanFord(const GraphBellmanFord&);

  //! Assignment operator not implemented.
  GraphBellmanFord&
  operator=(const GraphBellmanFord&);

public:

  //------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{

  //! Default constructor.
  GraphBellmanFord() :
    base_type() {}

  //! Destructor.
  virtual
  ~GraphBellmanFord() {}

  //@}
  //------------------------------------------------------------------------
  //! \name Accessors.
  //@{

  //! Return the vertices.
  const vertex_container&
  vertices() const
  {
    return base_type::vertices();
  }

  //! Return the edges.
  const edge_container&
  edges() const
  {
    return base_type::edges();
  }

  //@}
  //------------------------------------------------------------------------
  //! \name Mathematical operations.
  //@{

  //! Add an edge to the graph.
  void
  add_edge(const int source_index, const int target_index,
           const weight_type weight)
  {
    base_type::add_edge(source_index, target_index, weight);
  }

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
                   Generator& edge_weight)
  {
    base_type::rectangular_grid(x_size, y_size, edge_weight);
  }

  //! Make a dense graph.
  /*!
    Each vertex is doubly connected to every other vertex.
    \param num_vertices The number of vertices in the graph.
    \param edge_weight A functional with no arguments giving the edge weight.
  */
  template <typename Generator>
  void
  dense(const size_type num_vertices, Generator& edge_weight)
  {
    base_type::dense(num_vertices, edge_weight);
  }

  //! Make a graph with random edges.
  /*!
    \param num_vertices The number of vertices in the graph.
    \param num_adjacent_edges_per_vertex Each vertex has this many
    adjacent edges.
    \param edge_weight A functional with no arguments giving the edge weight.
  */
  template <typename Generator>
  void
  random(const size_type num_vertices,
         const size_type num_adjacent_edges_per_vertex,
         Generator& edge_weight)
  {
    base_type::random(num_vertices, num_adjacent_edges_per_vertex,
                      edge_weight);
  }

  //! Compute the shortest path tree from the given vertex.
  /*!
    \param root_vertex_index The index of the root vertex.
  */
  void
  bellman_ford(const int root_vertex_index);

  //@}

private:

  //! Initialize for a shortest path computation.
  /*!
    \param source_index is the index of the root vertex of the tree.
  */
  void
  initialize(const int source_index)
  {
    base_type::initialize(source_index);
  }
};

} // namespace shortest_paths
}

#define __GraphBellmanFord_ipp__
#include "stlib/shortest_paths/GraphBellmanFord.ipp"
#undef __GraphBellmanFord_ipp__

#endif
