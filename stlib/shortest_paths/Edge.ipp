// -*- C++ -*-

#if !defined(__Edge_ipp__)
#error This file is an implementation detail of the class Edge.
#endif

namespace stlib
{
namespace shortest_paths
{

template <typename VertexType>
inline
Edge<VertexType>::
Edge(const Edge& edge) :
  _source(edge._source),
  _target(edge._target),
  _weight(edge._weight) {}


template <typename VertexType>
inline
Edge<VertexType>&
Edge<VertexType>::
operator=(const Edge& rhs)
{
  if (&rhs != this) {
    _source = rhs._source;
    _target = rhs._target;
    _weight = rhs._weight;
  }
  return *this;
}

//
// Operations
//

template <typename VertexType>
inline
bool
Edge<VertexType>::
relax() const
{
  if (_source->distance() + _weight < _target->distance()) {
    _target->set_distance(_source->distance() + _weight);
    _target->set_predecessor(_source);
    return true;
  }
  return false;
}

//
// Equality
//

template <typename VertexType>
inline
bool
operator==(const Edge<VertexType>& x, const Edge<VertexType>& y)
{
  return (x.source() == y.source() &&
          x.target() == y.target() &&
          x.weight() == y.weight());
}

//
// File I/O.
//

template <typename VertexType>
inline
void
Edge<VertexType>::
put(std::ostream& out) const
{
  out << "source = " << _source
      << ",  target = " << _target
      << ",  weight = " << _weight;
}


template <typename VertexType>
inline
std::ostream&
operator<<(std::ostream& out, const Edge<VertexType>& edge)
{
  edge.put(out);
  return out;
}

} // namespace shortest_paths
}
