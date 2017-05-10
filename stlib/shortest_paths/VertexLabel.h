// -*- C++ -*-

/*!
  \file VertexLabel.h
  \brief Implements a class for a vertex in a weighted, directed graph.
  The vertex can be labeled with a status.
*/

#if !defined(__VertexLabel_h__)
#define __VertexLabel_h__

// Local
#include "stlib/shortest_paths/Vertex.h"

namespace stlib
{
namespace shortest_paths
{

//! The possible states of a vertex.
enum VertexStatus { KNOWN, LABELED, UNLABELED };

//! A vertex in a weighted, directed graph.
template <typename WeightType>
class VertexLabel :
  public Vertex<WeightType>
{
private:

  //
  // Private types.
  //

  typedef Vertex<WeightType> base_type;

public:

  //
  // Public types.
  //

  //! The weight type.
  typedef typename base_type::weight_type weight_type;

private:

  //
  // Member data.
  //

  //! The status of the vertex.
  VertexStatus _status;

public:

  //------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{

  //! Default constructor.
  VertexLabel() :
    base_type(),
    _status(UNLABELED) {}

  //! Copy constructor.
  VertexLabel(const VertexLabel& vertex) :
    base_type(vertex),
    _status(vertex._status) {}

  //! Assignment operator.
  VertexLabel&
  operator=(const VertexLabel& rhs);

  //! Trivial destructor.
  virtual
  ~VertexLabel() {}

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
    return _status;
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
    _status = s;
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
  initialize();

  //! Set this to be the root of the shortest paths tree.
  void
  set_root();

  //! Label self from a vertex and a weight.
  /*!
    If the status is UNLABELED, change the status to LABELED and
    set the distance and predecessor.  Otherwise, see if the distance
    from the vertex is lower than the current distance and if so,
    update the distance and predecessor.
  */
  void
  label(const VertexLabel& known_vertex, weight_type edge_weight);

  //@}
  //------------------------------------------------------------------------
  //! \name File I/O.
  //@{

  //! Write the vertex to an output stream.
  virtual
  void
  put(std::ostream& out) const;

  //@}
};

} // namespace shortest_paths
}

#define __VertexLabel_ipp__
#include "stlib/shortest_paths/VertexLabel.ipp"
#undef __VertexLabel_ipp__

#endif
