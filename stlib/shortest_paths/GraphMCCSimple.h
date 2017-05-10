// -*- C++ -*-

/*!
  \file GraphMCCSimple.h
  \brief Implements a class for a weighted, directed graph.
  Designed for shortest paths calculations with the marching with a correctness
  criterion using a simple correctness criterion.
*/

#if !defined(__GraphMCCSimple_h__)
#define __GraphMCCSimple_h__

#include "stlib/shortest_paths/Graph.h"
#include "stlib/shortest_paths/VertexMCCSimple.h"
#include "stlib/shortest_paths/EdgeCompare.h"

#include "stlib/performance/SimpleTimer.h"

#include <iosfwd>

#include <functional>
#include <iterator>
#include <vector>

namespace stlib
{
namespace shortest_paths
{

//! A weighted, directed graph.
template <typename WeightType>
class GraphMCCSimple :
  public Graph< VertexMCCSimple<WeightType> >
{
private:

  typedef Graph< VertexMCCSimple<WeightType> > base_type;

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

  //! Half edge.
  typedef HalfEdge<vertex_type> half_edge_type;
  //! Half edge container.
  typedef std::vector< half_edge_type > half_edge_container;
  //! Half edge iterator.
  typedef typename half_edge_container::iterator half_edge_iterator;
  //! Half edge const iterator.
  typedef typename half_edge_container::const_iterator
  half_edge_const_iterator;

private:

  //
  // Not implemented.
  //

  //! Copy constructor not implemented.
  GraphMCCSimple(const GraphMCCSimple&);

  //! Assignment operator not implemented.
  GraphMCCSimple&
  operator=(const GraphMCCSimple&);

private:

  //
  // Member data
  //

  half_edge_container _adjacent_edges;

public:

  //------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{

  //! Default constructor.
  GraphMCCSimple() :
    base_type(),
    _adjacent_edges() {}

  //! Destructor.
  virtual
  ~GraphMCCSimple() {}

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

  //! The vertex comparison functor.
  const VertexCompare<vertex_type*>&
  vertex_compare() const
  {
    return base_type::vertex_compare();
  }

  //@}
  //------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  //! Return the vertices.
  vertex_container&
  vertices()
  {
    return base_type::vertices();
  }

  //! Return the edges.
  edge_container&
  edges()
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
    \param root_vertex_index is the index of the root vertex.
  */
  void
  marching_with_correctness_criterion(const int root_vertex_index);

  //! Compute the shortest path tree from the given vertex.  Print stats.
  /*!
    \param root_vertex_index is the index of the root vertex.
  */
  void
  marching_with_correctness_criterion_count(const int root_vertex_index);

  //@}

private:

  //! Build the adjacency information for the vertices.
  void
  build();

  //! Initialize for a shortest path computation.
  /*!
    \param source_index is the index of the root vertex of the tree.
  */
  void
  initialize(const int source_index);
};

} // namespace shortest_paths
}

#define __GraphMCCSimple_ipp__
#include "stlib/shortest_paths/GraphMCCSimple.ipp"
#undef __GraphMCCSimple_ipp__

#endif
