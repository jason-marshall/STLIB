// -*- C++ -*-

/*!
  \file shortest_paths/HalfEdge.h
  \brief Implements a class for an edge in a weighted, directed graph.
  The edge is meant to be stored by a vertex.  It has only target vertex
  and weight information.
*/

#if !defined(__HalfEdge_h__)
#define __HalfEdge_h__

#include <iostream>

namespace stlib
{
namespace shortest_paths
{

//! An edge in a weighted, directed graph.
template <typename VertexType>
class HalfEdge
{
public:

  //
  // Public types.
  //

  //! A vertex.
  typedef VertexType vertex_type;
  //! Number type for the weight.
  typedef typename vertex_type::weight_type weight_type;

private:

  //
  // Member data.
  //

  // The vertex.
  vertex_type* _vertex;

  // The weight of the edge.
  weight_type _weight;

public:

  //------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{

  //! Default constructor.
  HalfEdge(VertexType* vertex = 0, weight_type weight = 0) :
    _vertex(vertex),
    _weight(weight) {}

  //! Copy constructor.
  HalfEdge(const HalfEdge& edge);

  //! Assignment operator.
  HalfEdge&
  operator=(const HalfEdge& rhs);

  //! Trivial destructor.
  virtual
  ~HalfEdge() {}

  //@}
  //------------------------------------------------------------------------
  //! \name Accessors.
  //@{

  //! Return a pointer to the vertex.
  vertex_type*
  vertex() const
  {
    return _vertex;
  }

  //! Return the weight of the edge.
  weight_type
  weight() const
  {
    return _weight;
  }

  //@}
  //------------------------------------------------------------------------
  //! \name File I/O.
  //@{

  //! Write the edge to the ostream.
  virtual
  void
  put(std::ostream& out) const;

  //@}
};

//
// Equality
//

//! Return true if the edges are equal.
/*! \relates HalfEdge */
template <typename VertexType>
bool
operator==(const HalfEdge<VertexType>& x, const HalfEdge<VertexType>& y);

//
// Stream Output
//

//! Write the half-edge.
/*! \relates HalfEdge */
template <typename VertexType>
std::ostream&
operator<<(std::ostream& out, const HalfEdge<VertexType>& edge);

} // namespace shortest_paths
}

#define __HalfEdge_ipp__
#include "stlib/shortest_paths/HalfEdge.ipp"
#undef __HalfEdge_ipp__

#endif
