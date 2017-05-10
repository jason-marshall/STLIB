// -*- C++ -*-

#if !defined(__VertexLabel_ipp__)
#error This file is an implementation detail of the class VertexLabel.
#endif

namespace stlib
{
namespace shortest_paths
{

//
// Constructors, Destructor.
//

template <typename WeightType>
inline
VertexLabel<WeightType>&
VertexLabel<WeightType>::
operator=(const VertexLabel& rhs)
{
  if (&rhs != this) {
    base_type::operator=(rhs);
    _status = rhs._status;
  }
  return *this;
}

//
// Mathematical operations
//

template <typename WeightType>
inline
void
VertexLabel<WeightType>::
initialize()
{
  base_type::initialize();
  _status = UNLABELED;
}

template <typename WeightType>
inline
void
VertexLabel<WeightType>::
set_root()
{
  base_type::set_root();
  _status = KNOWN;
}


template <typename WeightType>
inline
void
VertexLabel<WeightType>::
label(const VertexLabel& known_vertex, weight_type edge_weight)
{
  if (_status == UNLABELED) {
    _status = LABELED;
    set_distance(known_vertex.distance() + edge_weight);
    set_predecessor(&known_vertex);
  }
  else { // _status == LABELED
    weight_type new_distance = known_vertex.distance() + edge_weight;
    if (new_distance < distance()) {
      set_distance(new_distance);
      set_predecessor(&known_vertex);
    }
  }
}

//
// Stream output
//

template <typename WeightType>
inline
void
VertexLabel<WeightType>::
put(std::ostream& out) const
{
  base_type::put(out);
  out << ", status = " << _status;
}

} // namespace shortest_paths
}

