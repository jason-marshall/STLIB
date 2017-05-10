// -*- C++ -*-

/*!
  \file VertexDijkstra.h
  \brief Implements a class for a vertex in a weighted, directed graph.
  The vertex is designed for shortest paths calculations with Dijkstra's
  algorithm.
*/

#if !defined(__VertexDijkstra_h__)
#define __VertexDijkstra_h__

// Local
#include "stlib/shortest_paths/VertexLabel.h"
#include "stlib/shortest_paths/HalfEdge.h"

namespace stlib
{
namespace shortest_paths
{

//! A vertex in a weighted, directed graph.
template <typename WeightType>
class VertexDijkstra :
  public VertexLabel<WeightType>
{
private:

  //
  // Private types.
  //

  typedef VertexLabel<WeightType> base_type;

public:

  //
  // Public types.
  //

  //! The weight type.
  typedef typename base_type::weight_type weight_type;
  //! The edge type.
  typedef HalfEdge<VertexDijkstra> edge_type;

private:

  //
  // Member data.
  //

  //! Pointer to the begining of an array of the adjacent edges.
  const edge_type* _adjacent_edges;

  //! A pointer into a heap of pointers to vertices.
  VertexDijkstra** _heap_ptr;

public:

  //------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{

  //! Default constructor.
  VertexDijkstra() :
    base_type(),
    _adjacent_edges(0),
    _heap_ptr(0) {}

  //! Copy constructor.
  VertexDijkstra(const VertexDijkstra& vertex) :
    base_type(vertex),
    _adjacent_edges(vertex._adjacent_edges),
    _heap_ptr(vertex._heap_ptr) {}

  //! Assignment operator.
  VertexDijkstra&
  operator=(const VertexDijkstra& rhs);

  //! Trivial destructor.
  virtual
  ~VertexDijkstra() {}

  //@}
  //------------------------------------------------------------------------
  //! \name Accessors.
  //@{

  //! Return the distance from the source.
  weight_type distance() const
  {
    return base_type::distance();
  }

  //! Return the predecessor in the shortest paths tree.
  const Vertex<weight_type>*
  predecessor() const
  {
    return base_type::predecessor();
  }

  //! Return the status of the vertex.
  VertexStatus
  status() const
  {
    return base_type::status();
  }

  //! Return a pointer to the begining of the adjacent edges.
  const edge_type*
  adjacent_edges() const
  {
    return _adjacent_edges;
  }

  //! Return the heap pointer.
  VertexDijkstra**
  heap_ptr() const
  {
    return _heap_ptr;
  }

  //@}
  //------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  //! Set the distance from the source.
  void
  set_distance(const weight_type d)
  {
    base_type::set_distance(d);
  }

  //! Set the predecessor in the shortest paths tree.
  void
  set_predecessor(const Vertex<weight_type>* p)
  {
    base_type::set_predecessor(p);
  }

  //! Return a reference to the status of the vertex.
  void
  set_status(const VertexStatus s)
  {
    base_type::set_status(s);
  }

  //! Set the pointer to the begining of the adjacent edges.
  void
  set_adjacent_edges(const edge_type* e)
  {
    _adjacent_edges = e;
  }

  //! Return a reference to the heap pointer.
  VertexDijkstra**&
  heap_ptr()
  {
    return _heap_ptr;
  }

  //! Set the heap pointer.
  void
  set_heap_ptr(VertexDijkstra** p)
  {
    _heap_ptr = p;
  }

  //@}
  //------------------------------------------------------------------------
  //! \name Mathematical operations.
  //@{

  //! Initialize the vertex for a shortest path calculation.
  /*!
    Set the status of the vertex to UNLABELED.
  */
  void
  initialize()
  {
    base_type::initialize();
  }

  //! Set this to be the root of the shortest paths tree.
  void
  set_root()
  {
    base_type::set_root();
  }

  //! Label self from a vertex and a weight.
  /*!
    If the status is UNLABELED, change the status to LABELED and
    set the distance and predecessor.  Otherwise, see if the distance
    from the vertex is lower than the current distance and if so,
    update the distance and predecessor.
  */
  void
  label(const VertexLabel<weight_type>& known_vertex,
        weight_type edge_weight)
  {
    base_type::label(known_vertex, edge_weight);
  }

  //@}
};

} // namespace shortest_paths
}

#define __VertexDijkstra_ipp__
#include "stlib/shortest_paths/VertexDijkstra.ipp"
#undef __VertexDijkstra_ipp__

#endif
