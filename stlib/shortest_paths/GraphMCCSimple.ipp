// -*- C++ -*-

#if !defined(__GraphMCCSimple_ipp__)
#error This file is an implementation detail of the class GraphMCCSimple.
#endif

namespace stlib
{
namespace shortest_paths
{

//
// Mathematical operations
//

template <typename WeightType>
inline
void
GraphMCCSimple<WeightType>::
build()
{
  // Allocate memory for the adjacent edges.
  {
    half_edge_container temp(edges().size());
    _adjacent_edges.swap(temp);
    _adjacent_edges.clear();
  }

  // Sort the edges by source vertex.
  {
    EdgeSourceCompare<edge_type> comp;
    std::sort(edges().begin(), edges().end(), comp);
  }

  // Add the adjacent edges.
  {
    edge_const_iterator edge_iter = edges().begin();
    edge_const_iterator edge_end = edges().end();
    vertex_iterator vert_iter = vertices().begin();
    const vertex_iterator vert_end = vertices().end();
    for (; vert_iter != vert_end; ++vert_iter) {
      vert_iter->set_adjacent_edges(&*_adjacent_edges.end());
      while (edge_iter != edge_end &&
             edge_iter->source() == &*vert_iter) {
        _adjacent_edges.push_back(half_edge_type(edge_iter->target(),
                                  edge_iter->weight()));
        ++edge_iter;
      }
    }
  }

  // Sort the edges by target vertex.
  {
    EdgeTargetCompare<edge_type> comp;
    std::sort(edges().begin(), edges().end(), comp);
  }

  // Compute the minimum incident edge weight for each vertex.
  {
    edge_const_iterator edge_iter = edges().begin();
    edge_const_iterator edge_end = edges().end();
    vertex_iterator vert_iter = vertices().begin();
    const vertex_iterator vert_end = vertices().end();
    for (; vert_iter != vert_end; ++vert_iter) {
      if (edge_iter != edge_end && edge_iter->target() == &*vert_iter) {
        vert_iter->set_min_incident_edge_weight(edge_iter->weight());
        ++edge_iter;
      }
      while (edge_iter != edge_end &&
             edge_iter->target() == &*vert_iter) {
        if (edge_iter->weight() < vert_iter->min_incident_edge_weight()) {
          vert_iter->set_min_incident_edge_weight(edge_iter->weight());
        }
        ++edge_iter;
      }
    }
  }

  // Clear the edges.
  {
    edge_container temp;
    edges().swap(temp);
  }
}

template <typename WeightType>
inline
void
GraphMCCSimple<WeightType>::
initialize(const int source_index)
{
  // Initialize the data in each vertex.
  vertex_iterator
  iter = vertices().begin(),
  iter_end = vertices().end();
  for (; iter != iter_end; ++iter) {
    iter->initialize();
  }

  // Set the source vertex to known.
  vertex_type& source = vertices()[source_index];
  source.set_status(KNOWN);
  source.set_distance(0);
  source.set_predecessor(0);
}

template <typename WeightType>
inline
void
GraphMCCSimple<WeightType>::
marching_with_correctness_criterion(const int root_vertex_index)
{
  // The list of labeled unknown vertices to check during a step.
  std::vector<vertex_type*> labeled;
  // The list of vertices which are added to label during a step.
  vertex_type** new_labeled = new vertex_type*[ vertices().size()];
  vertex_type** new_labeled_end;

  // Initialize the graph.
  initialize(root_vertex_index);

  // Label the neighbors of the root.
  vertices()[root_vertex_index].label_adjacent(std::back_inserter(labeled));

  // All vertices are known when there are no labeled vertices left.
  // Loop while there are labeled vertices left.
  typename std::vector<vertex_type*>::iterator vert_ptr_iter;
  typename std::vector<vertex_type*>::iterator labeled_end;
  WeightType min_unknown_distance;
  //  WeightType threshold;
  while (labeled.size() > 0) {

    // Find the minimum unknown distance.
    min_unknown_distance
      = (*(std::min_element(labeled.begin(), labeled.end(),
                            vertex_compare())))->distance();

    // Loop through the labeled vertices.
    labeled_end = labeled.end();
    new_labeled_end = new_labeled;
    for (vert_ptr_iter = labeled.begin();
         vert_ptr_iter != labeled_end;
         ++vert_ptr_iter) {

      if ((*vert_ptr_iter)->distance() <= min_unknown_distance +
          (*vert_ptr_iter)->min_incident_edge_weight()) {

        // Mark the vertex as known.
        (*vert_ptr_iter)->set_status(KNOWN);
        // Label the adjacent vertices.
        new_labeled_end = (*vert_ptr_iter)->label_adjacent(new_labeled_end);
        // Flag the vertex for deletion.
        *vert_ptr_iter = 0;
      }
    }

    // Remove the vertices that became known.
    labeled.erase(std::remove_copy(labeled.begin(), labeled.end(),
                                   labeled.begin(),
                                   static_cast<vertex_type*>(0)),
                  labeled.end());
    // Add the newly labeled vertices.
    labeled.insert(labeled.end(), new_labeled, new_labeled_end);
  }
  delete[] new_labeled;
}

template <typename WeightType>
inline
void
GraphMCCSimple<WeightType>::
marching_with_correctness_criterion_count(const int root_vertex_index)
{
  performance::SimpleTimer timer;
  double find_min_time = 0, label_time = 0, erase_time = 0;

  // Allocate time begin.
  timer.start();

  // The list of labeled unknown vertices to check during a step.
  std::vector<vertex_type*> labeled;
  // The list of vertices which are added to label during a step.
  vertex_type** new_labeled = new vertex_type*[ vertices().size()];
  vertex_type** new_labeled_end;

  // Allocate time end.
  timer.stop();
  double allocate_time = timer.elapsed();


  // Initialize time begin.
  timer.start();

  // Initialize the graph.
  initialize(root_vertex_index);

  // Initialize time end.
  timer.stop();
  double initialize_time = timer.elapsed();


  // Label the neighbors of the root.
  vertices()[root_vertex_index].label_adjacent(std::back_inserter(labeled));

  int is_determined_count = 0, label_adjacent_count = 0;

  // All vertices are known when there are no labeled vertices left.
  // Loop while there are labeled vertices left.
  typename std::vector<vertex_type*>::iterator vert_ptr_iter;
  typename std::vector<vertex_type*>::iterator labeled_end;
  weight_type min_unknown_distance;
  while (labeled.size() > 0) {

    // Find min time begin.
    timer.start();

    // Find the minimum unknown distance.
    min_unknown_distance
      = (*(std::min_element(labeled.begin(), labeled.end(),
                            vertex_compare())))->distance();

    // Find min time end.
    timer.stop();
    find_min_time += timer.elapsed();

    is_determined_count += labeled.size();

    // Label time begin.
    timer.start();

    // Loop through the labeled vertices.
    labeled_end = labeled.end();
    new_labeled_end = new_labeled;
    for (vert_ptr_iter = labeled.begin();
         vert_ptr_iter != labeled_end;
         ++vert_ptr_iter) {

      if ((*vert_ptr_iter)->distance() <= min_unknown_distance +
          (*vert_ptr_iter)->min_incident_edge_weight()) {

        ++label_adjacent_count;

        // Mark the vertex as known.
        (*vert_ptr_iter)->status() = KNOWN;
        // Label the adjacent vertices.
        new_labeled_end = (*vert_ptr_iter)->label_adjacent(new_labeled_end);
        // Flag the vertex for deletion.
        *vert_ptr_iter = 0;
      }
    }

    // Label time end.
    timer.stop();
    label_time += timer.elapsed();

    // Erase time begin.
    timer.start();

    // Remove the vertices that became known.
    labeled.erase(std::remove_copy(labeled.begin(), labeled.end(),
                                   labeled.begin(),
                                   static_cast<vertex_type*>(0)),
                  labeled.end());
    // Add the newly labeled vertices.
    labeled.insert(labeled.end(), new_labeled, new_labeled_end);

    // Erase time end.
    timer.stop();
    erase_time += timer.elapsed();
  }
  delete[] new_labeled;

  std::cout << "Allocate time =   " << allocate_time << '\n'
            << "Initialize time = " << initialize_time << '\n'
            << "Find min time =   " << find_min_time << '\n'
            << "Label time =      " << label_time << '\n'
            << "Erase time =      " << erase_time << '\n'
            << "Is determined = " << is_determined_count << '\n'
            << "Label adjacent = " << label_adjacent_count << '\n';
}

} // namespace shortest_paths
}

