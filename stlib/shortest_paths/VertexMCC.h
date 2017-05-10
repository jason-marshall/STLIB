// -*- C++ -*-

/*!
  \file VertexMCC.h
  \brief Implements a class for a vertex in a weighted, directed graph.
*/

#if !defined(__VertexMCC_h__)
#define __VertexMCC_h__

// Local
#include "stlib/shortest_paths/VertexLabel.h"
#include "stlib/shortest_paths/HalfEdge.h"

namespace stlib
{
namespace shortest_paths
{

//! A vertex in a weighted, directed graph.
template <typename WeightType>
class VertexMCC :
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
  typedef HalfEdge<VertexMCC> edge_type;

private:

  //
  // Member data.
  //

  //! Pointer to the begining of an array of the adjacent edges.
  const edge_type* _adjacent_edges;

  //! Pointer to the begining of an array of the incident edges.
  const edge_type* _incident_edges;

  //! Pointer to incident edge from an unknown vertex.
  const edge_type* _unknown_incident_edge;

  //! The weight of the minimum incident edge.
  weight_type _min_incident_edge_weight;

public:

  //------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{

  //! Default constructor.
  VertexMCC() :
    base_type(),
    _adjacent_edges(0),
    _incident_edges(0),
    _unknown_incident_edge(0),
    _min_incident_edge_weight(0) {}

  //! Copy constructor.
  VertexMCC(const VertexMCC& vertex);

  //! Assignment operator.
  VertexMCC&
  operator=(const VertexMCC& rhs);

  //! Trivial destructor.
  virtual
  ~VertexMCC() {}

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

  //! Return the adjacent edges.
  const edge_type*
  adjacent_edges() const
  {
    return _adjacent_edges;
  }

  //! Return the incident edges.
  const edge_type*
  incident_edges() const
  {
    return _incident_edges;
  }

  //! Return the minimum incident edge weight.
  weight_type
  min_incident_edge_weight() const
  {
    return _min_incident_edge_weight;
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

  //! Return the adjacent edges.
  void
  set_adjacent_edges(const edge_type* e)
  {
    _adjacent_edges = e;
  }

  //! Return the incident edges.
  void
  set_incident_edges(const edge_type* e)
  {
    _incident_edges = e;
  }

  //! Return a reference to the minimum incident edge weight.
  void
  set_min_incident_edge_weight(const weight_type w)
  {
    _min_incident_edge_weight = w;
  }

  //------------------------------------------------------------------------
  //! \name Mathematical operations.
  //@{

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

  //! Initialize the vertex for a shortest path calculation.
  /*!
    Set the status of the vertex to UNLABELED.
    Initialize the unknown incident edge.
  */
  void
  initialize();

  //! Return true if the value of the vertex is known.
  bool
  is_correct(const weight_type min_unknown_distance);

  //! Label the unkown neighbors of the specified vertex.
  /*!
    Add the neighbors that were unlabeled to unlabeled_neighbors.
  */
  template <typename OutputIterator>
  OutputIterator
  label_adjacent(OutputIterator unlabeled_neighbors);

  //@}

private:

  //! Return a lower bound on the distance of the vertex.
  weight_type
  lower_bound(const weight_type min_unknown_distance);

  //! Find the next unknown incident edge.
  /*!
    Set _unknown_incident_edge to this edge.  If there are no more
    unknown incident edges, set _unknown_incident_edge to 0.
  */
  void
  get_unknown_incident_edge();
};

} // namespace shortest_paths
}

#define __VertexMCC_ipp__
#include "stlib/shortest_paths/VertexMCC.ipp"
#undef __VertexMCC_ipp__

#endif
