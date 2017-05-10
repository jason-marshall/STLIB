// -*- C++ -*-

#if !defined(__Vertex_ipp__)
#error This file is an implementation detail of the class Vertex.
#endif

namespace stlib
{
namespace shortest_paths
{

//
// Constructors, etc.
//

template <typename WeightType>
inline
Vertex<WeightType>&
Vertex<WeightType>::
operator=(const Vertex& rhs)
{
  if (&rhs != this) {
    _distance = rhs._distance;
    _predecessor = rhs._predecessor;
  }
  return *this;
}

//
// Mathematical operations
//

template <typename WeightType>
inline
void
Vertex<WeightType>::
initialize()
{
  _distance = std::numeric_limits<weight_type>::max();
  _predecessor = 0;
}

/* CONTINUE: REMOVE
template <>
inline
void
Vertex<double>::
initialize()
{
  _distance = DBL_MAX;
  _predecessor = 0;
}

template <>
inline
void
Vertex<float>::
initialize()
{
  _distance = FLT_MAX;
  _predecessor = 0;
}

template <>
inline
void
Vertex<int>::
initialize()
{
  _distance = INT_MAX;
  _predecessor = 0;
}
*/

template <typename WeightType>
inline
void
Vertex<WeightType>::
set_root()
{
  _distance = 0;
  _predecessor = 0;
}

template <typename WeightType>
inline
void
Vertex<WeightType>::
relax(const Vertex& source_vertex, weight_type edge_weight)
{
  weight_type new_distance = source_vertex.distance() + edge_weight;
  if (new_distance < _distance) {
    _distance = new_distance;
    _predecessor = &source_vertex;
  }
}

//
// Stream output
//

template <typename WeightType>
inline
void
Vertex<WeightType>::
put(std::ostream& out) const
{
  out << "dist = " << _distance
      << ",  pred = " << _predecessor;
}

template <typename WeightType>
inline
std::ostream&
operator<<(std::ostream& out, const Vertex<WeightType>& v)
{
  v.put(out);
  return out;
}

//
// Equality
//

template <typename WeightType>
inline
bool
operator==(const Vertex<WeightType>& x, const Vertex<WeightType>& y)
{
  return (x.distance() == y.distance() &&
          x.predecessor() == y.predecessor());
}

} // namespace shortest_paths
}

