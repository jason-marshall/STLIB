// -*- C++ -*-

/*!
  \file Vertex.h
  \brief Implements a class for a vertex in a weighted, directed graph.
*/

#if !defined(__Vertex_h__)
#define __Vertex_h__

#include <iostream>
#include <limits>

namespace stlib
{
namespace shortest_paths
{

//! A vertex in a weighted, directed graph.
/*!
  This vertex implements the common functionality for vertices.
*/
template <typename WeightType>
class Vertex
{
public:

  //
  // Typedefs
  //

  //! The number type for the weight.
  typedef WeightType weight_type;

private:

  //
  // Member data.
  //

  // The distance from the source.
  weight_type _distance;

  // A pointer to the predecessor in the shortest paths tree.
  const Vertex* _predecessor;

public:

  //-------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{

  //! Default constructor.
  Vertex() :
    _distance(-1),
    _predecessor(0) {}

  //! Copy constructor.
  Vertex(const Vertex& vertex) :
    _distance(vertex._distance),
    _predecessor(vertex._predecessor) {}

  //! Assignment operator.
  Vertex&
  operator=(const Vertex& rhs);

  //! Trivial destructor.
  virtual
  ~Vertex() {}

  //@}
  //-------------------------------------------------------------------------
  //! \name Accessors.
  //@{

  //! Return the distance from the source.
  weight_type distance() const
  {
    return _distance;
  }

  //! Return the predecessor in the shortest paths tree.
  const Vertex*
  predecessor() const
  {
    return _predecessor;
  }

  //@}
  //-------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  //! Set the distance from the source.
  void
  set_distance(const weight_type d)
  {
    _distance = d;
  }

  //! Set the predecessor in the shortest paths tree.
  void
  set_predecessor(const Vertex* p)
  {
    _predecessor = p;
  }

  //@}
  //-------------------------------------------------------------------------
  //! \name Mathematical operations.
  //@{

  //! Initialize the distance to infinity and set the predecessor to 0.
  void
  initialize();

  //! Set the distance to 0 and the predecessor to 0.
  void
  set_root();

  //! Update self from a source vertex and a weight.
  /*!
    See if the distance from the vertex is lower than the current
    distance and if so, update the distance and predecessor.
  */
  void
  relax(const Vertex& known_vertex, weight_type edge_weight);

  //@}
  //-------------------------------------------------------------------------
  //! \name File I/O.
  //@{

  //! Write the vertex to an output stream.
  virtual
  void
  put(std::ostream& out) const;

  //@}
};

//
// Stream output
//

//! Write the vertex to an output stream.
/*! \relates Vertex */
template <typename WeightType>
std::ostream&
operator<<(std::ostream& out, const Vertex<WeightType>& vertex);

//
// Equality
//

//! Return true if the vertices are equal.
/*! \relates Vertex */
template <typename WeightType>
bool
operator==(const Vertex<WeightType>& x, const Vertex<WeightType>& y);

//! Return true if the vertices are not equal.
/*! \relates Vertex */
template <typename WeightType>
inline
bool
operator!=(const Vertex<WeightType>& x, const Vertex<WeightType>& y)
{
  return !(x == y);
}

} // namespace shortest_paths
}

#define __Vertex_ipp__
#include "stlib/shortest_paths/Vertex.ipp"
#undef __Vertex_ipp__

#endif
