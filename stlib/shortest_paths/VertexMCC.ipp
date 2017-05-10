// -*- C++ -*-

#if !defined(__VertexMCC_ipp__)
#error This file is an implementation detail of the class VertexMCC.
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
VertexMCC<WeightType>::
VertexMCC(const VertexMCC& vertex) :
  base_type(vertex),
  _adjacent_edges(vertex._adjacent_edges),
  _incident_edges(vertex._incident_edges),
  _unknown_incident_edge(vertex._unknown_incident_edge),
  _min_incident_edge_weight(vertex._min_incident_edge_weight)
{
}

template <typename WeightType>
inline
VertexMCC<WeightType>&
VertexMCC<WeightType>::
operator=(const VertexMCC& vertex)
{
  if (this != &vertex) {
    base_type::operator=(vertex);
    _adjacent_edges = vertex._adjacent_edges;
    _incident_edges = vertex._incident_edges;
    _unknown_incident_edge = vertex._unknown_incident_edge;
    _min_incident_edge_weight = vertex._min_incident_edge_weight;
  }
  return *this;
}


//
// Mathematical operations
//

template <typename WeightType>
inline
void
VertexMCC<WeightType>::
initialize()
{
  base_type::initialize();
  _unknown_incident_edge = _incident_edges;
}

template <typename WeightType>
template <typename OutputIterator>
inline
OutputIterator
VertexMCC<WeightType>::
label_adjacent(OutputIterator unlabeled_neighbors)
{
  const edge_type* iter = _adjacent_edges;
  const edge_type* iter_end = (this + 1)->_adjacent_edges;
  VertexMCC* adjacent;
  for (; iter != iter_end; ++iter) {
    adjacent = iter->vertex();
    if (adjacent->status() != KNOWN) {
      if (adjacent->status() == UNLABELED) {
        *unlabeled_neighbors = adjacent;
        ++unlabeled_neighbors;
      }
      adjacent->label(*this, iter->weight());
    }
  }
  return unlabeled_neighbors;
}

// Precondition: _status != KNOWN
template <typename WeightType>
inline
typename VertexMCC<WeightType>::weight_type
VertexMCC<WeightType>::
lower_bound(const weight_type min_unknown_distance)
{
  return std::min(distance(),
                  min_unknown_distance + _min_incident_edge_weight);
}

template <typename WeightType>
inline
bool
VertexMCC<WeightType>::
is_correct(const weight_type min_unknown_distance)
{
  get_unknown_incident_edge();
  while (_unknown_incident_edge) {
    if (_unknown_incident_edge->weight()
        + _unknown_incident_edge->vertex()->
        lower_bound(min_unknown_distance) < distance()) {
      return false;
    }
    ++_unknown_incident_edge;
    get_unknown_incident_edge();
  }
  return true;
}

template <typename WeightType>
inline
void
VertexMCC<WeightType>::
get_unknown_incident_edge()
{
  // Go through the incident edges until an unknown one is found.
  const edge_type* edges_end = (this + 1)->_incident_edges;
  for (;
       _unknown_incident_edge != edges_end &&
       _unknown_incident_edge->vertex()->status() == KNOWN;
       ++_unknown_incident_edge)
    ;
  if (_unknown_incident_edge == edges_end) {
    _unknown_incident_edge = 0;
  }
}

} // namespace shortest_paths
}

