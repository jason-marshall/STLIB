// -*- C++ -*-

#if !defined(__HalfEdge_ipp__)
#error This file is an implementation detail of the class HalfEdge.
#endif

namespace stlib
{
namespace shortest_paths
{

template <typename VertexType>
inline
HalfEdge<VertexType>::
HalfEdge(const HalfEdge& edge) :
  _vertex(edge._vertex),
  _weight(edge._weight)
{
}

template <typename VertexType>
inline
HalfEdge<VertexType>&
HalfEdge<VertexType>::
operator=(const HalfEdge& rhs)
{
  if (&rhs != this) {
    _vertex = rhs._vertex;
    _weight = rhs._weight;
  }
  return *this;
}

//
// Equality
//

template <typename VertexType>
inline
bool
operator==(const HalfEdge<VertexType>& x, const HalfEdge<VertexType>& y)
{
  return (x.vertex() == y.vertex() && x.weight() == y.weight());
}


//
// Stream Output
//

template <typename VertexType>
inline
void
HalfEdge<VertexType>::
put(std::ostream& out) const
{
  out << "vertex = " << _vertex
      << ",  weight = " << _weight;
}


template <typename VertexType>
std::ostream&
operator<<(std::ostream& out, const HalfEdge<VertexType>& edge)
{
  edge.put(out);
  return out;
}

} // namespace shortest_paths
}

