// -*- C++ -*-

#if !defined(__geom_ISS_SimplexQuery_ipp__)
#error This file is an implementation detail of the class ISS_SimplexQuery.
#endif

namespace stlib
{
namespace geom {

template<class ISS>
inline
void
ISS_SimplexQuery<ISS>::
build() {
   std::vector<BBox> boxes(_iss.indexedSimplices.size());
   Simplex simplex;
   // For each simplex.
   for (std::size_t n = 0; n != _iss.indexedSimplices.size(); ++n) {
      _iss.getSimplex(n, &simplex);
      // Make a bounding box around the simplex.
      boxes[n] = specificBBox<BBox>(simplex.begin(), simplex.end());
   }
   // Build the tree from the bounding boxes.
   _bboxTree.build(boxes.begin(), boxes.end());
}


//
// Queries.
//


// Get the indices of the simplices that contain the point.
template<class ISS>
template <typename IntOutIter>
inline
void
ISS_SimplexQuery<ISS>::
computePointQuery(IntOutIter iter, const Vertex& x) const {
   std::size_t n;
   Simplex s;
   std::vector<std::size_t> candidates;
   _bboxTree.computePointQuery(std::back_inserter(candidates), x);
   for (std::vector<std::size_t>::const_iterator i = candidates.begin();
         i != candidates.end(); ++i) {
      n = *i;
      // Get the simplex.
      _iss.getSimplex(n, &s);
      if (isIn(s, x)) {
         *iter++ = n;
      }
   }
}


// Get the indices of the simplices that overlap the window.
template<class ISS>
template <typename IntOutIter>
inline
void
ISS_SimplexQuery<ISS>::
computeWindowQuery(IntOutIter iter, const BBox& window) const {
   _bboxTree.computeWindowQuery(iter, window);
}



// Return the index of the simplex of minimum distance.
template<class ISS>
inline
int
ISS_SimplexQuery<ISS>::
computeMinimumDistanceAndIndex(const Vertex& x, Number* minDistance) const {
   // If there are no simplices.
   if (_iss.indexedSimplices.size() == 0) {
      // Return an invalid index.
      return -1;
   }

   std::size_t n;
   Simplex s;

   //
   // Get the candidates simplices.
   //
   std::vector<std::size_t> candidates;
   _bboxTree.computeMinimumDistanceQuery(std::back_inserter(candidates), x);

   //
   // Calculate distance to the candidate simplices.
   //
   std::vector<Number> distances(candidates.size());
   const std::size_t i_end = candidates.size();
   for (std::size_t i = 0; i != i_end; ++i) {
      n = candidates[i];
      // Get the simplex.
      _iss.getSimplex(n, &s);
      // Calculate the signed or unsigned distance to the simplex.
      distances[i] = computeDistance(s, x);
      // CONTINUE REMOVE
      //std::cerr << "simplex = " << s << " d = " << distances[i] << "\n";
   }

   //
   // Choose the one of minimum distance.
   //
   typename std::vector<Number>::const_iterator minIter =
      std::min_element(distances.begin(), distances.end());

   assert(minIter != distances.end());

   // Record the minimum distance.
   *minDistance = *minIter;
   // Return the index of the closest simplex.
   return candidates[minIter - distances.begin()];
}



// Return the minimum distance to the mesh.
template<class ISS>
inline
typename ISS_SimplexQuery<ISS>::Number
ISS_SimplexQuery<ISS>::
computeMinimumDistance(const Vertex& x) const {
   // If there are no simplices.
   if (_iss.indexedSimplices.size() == 0) {
      // Return infinity.
      return std::numeric_limits<Number>::max();
   }
   // REMOVE
   //std::cerr << "x = " << x << "\n";

   std::size_t n;
   Simplex s;

   //
   // Get the candidates simplices.
   //
   std::vector<std::size_t> candidates;
   // REMOVE
   //std::cerr << "start\n";
   _bboxTree.computeMinimumDistanceQuery(std::back_inserter(candidates), x);
   // REMOVE
   //std::cerr << "finish\n";

   // REMOVE
   //std::cerr << "size = " << candidates.size() << "\n";

   //
   // Calculate distance to the candidate simplices.
   //
   Number d;
   Number minDist = std::numeric_limits<Number>::max();
   const std::size_t iEnd = candidates.size();
   for (std::size_t i = 0; i != iEnd; ++i) {
      n = candidates[i];
      // Get the simplex.
      _iss.getSimplex(n, &s);
      // Calculate the signed distance to the simplex.
      d = computeDistance(s, x);
      // REMOVE
      //std::cerr << "d = " << d << "\n";
      // Update the minimum distance.
      if (d < minDist) {
         minDist = d;
      }
   }

   assert(minDist != std::numeric_limits<Number>::max());

   return minDist;
}


} // namespace geom
}
