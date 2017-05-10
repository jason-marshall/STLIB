// -*- C++ -*-

/*!
  \file Edge.h
  \brief Implements a class for an edge in a weighted, directed graph.
*/

#if !defined(__Edge_h__)
#define __Edge_h__

#include <iostream>

namespace stlib
{
namespace shortest_paths
{

//! An edge in a weighted, directed graph.
template <typename VertexType>
class Edge
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

  // The source vertex.
  vertex_type* _source;

  // The target vertex.
  vertex_type* _target;

  // The weight of the edge.
  weight_type _weight;

public:

  //------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{

  //! Default constructor.
  Edge(vertex_type* source = 0, vertex_type* target = 0,
       weight_type weight = 0) :
    _source(source),
    _target(target),
    _weight(weight) {}

  //! Copy constructor.
  Edge(const Edge& edge);

  //! Assignment operator.
  Edge&
  operator=(const Edge& rhs);

  //! Trivial destructor.
  virtual
  ~Edge() {}

  //@}
  //------------------------------------------------------------------------
  //! \name Accessors.
  //@{

  //! Return a pointer to the source vertex.
  vertex_type*
  source() const
  {
    return _source;
  }

  //! Return a pointer to the target vertex.
  vertex_type*
  target() const
  {
    return _target;
  }

  //! Return the weight of the edge.
  weight_type
  weight() const
  {
    return _weight;
  }

  //@}
  //------------------------------------------------------------------------
  //! \name Operations.
  //@{

  //! Relax along this edge.
  bool
  relax() const;

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
/*! \relates Edge */
template <typename VertexType>
bool
operator==(const Edge<VertexType>& x, const Edge<VertexType>& y);

//
// Stream Output
//

//! Write the edge.
/*! \relates Edge */
template <typename VertexType>
std::ostream&
operator<<(std::ostream& out, const Edge<VertexType>& edge);

} // namespace shortest_paths
}

#define __Edge_ipp__
#include "stlib/shortest_paths/Edge.ipp"
#undef __Edge_ipp__

#endif
