// -*- C++ -*-

/*!
  \file VertexMCCSimple.h
  \brief Implements a class for a vertex in a weighted, directed graph.
*/

#if !defined(__VertexMCCSimple_h__)
#define __VertexMCCSimple_h__

// Local
#include "stlib/shortest_paths/VertexLabel.h"
#include "stlib/shortest_paths/HalfEdge.h"

namespace stlib
{
namespace shortest_paths
{

//! A vertex in a weighted, directed graph.
template <typename WeightType>
class VertexMCCSimple :
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
  typedef HalfEdge<VertexMCCSimple> edge_type;

private:

  //
  // Member data.
  //

  //! Pointer to the begining of an array of the adjacent edges.
  const edge_type* _adjacent_edges;

  //! The minimum incident edge weight for this vertex.
  weight_type _min_incident_edge_weight;

public:

  //------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{

  //! Default constructor.
  VertexMCCSimple() :
    base_type::VertexLabel(),
    _adjacent_edges(0),
    _min_incident_edge_weight(0) {}

  //! Copy constructor.
  VertexMCCSimple(const VertexMCCSimple& vertex) :
    base_type(vertex),
    _adjacent_edges(vertex._adjacent_edges),
    _min_incident_edge_weight(vertex._min_incident_edge_weight) {}

  //! Assignment operator.
  VertexMCCSimple&
  operator=(const VertexMCCSimple& rhs);

  //! Trivial destructor.
  virtual
  ~VertexMCCSimple() {}

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

  //! Return the minimum incident edge weight.
  weight_type
  min_incident_edge_weight() const
  {
    return _min_incident_edge_weight;
  }

  //! Return the adjacent edges.
  const edge_type*
  adjacent_edges() const
  {
    return _adjacent_edges;
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

  //! Return a reference to the minimum incident edge weight.
  void
  set_min_incident_edge_weight(weight_type w)
  {
    _min_incident_edge_weight = w;
  }

  //! Return the adjacent edges.
  void
  set_adjacent_edges(const edge_type* e)
  {
    _adjacent_edges = e;
  }

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

  //! Label the unkown neighbors of the specified vertex.
  /*!
    Add the neighbors that were unlabeled to unlabeled_neighbors.
  */
  template <typename OutputIterator>
  OutputIterator label_adjacent(OutputIterator unlabeled_neighbors);

  //@}
};

} // namespace shortest_paths
}

#define __VertexMCCSimple_ipp__
#include "stlib/shortest_paths/VertexMCCSimple.ipp"
#undef __VertexMCCSimple_ipp__

#endif
