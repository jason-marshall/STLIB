// -*- C++ -*-

#if !defined(__geom_mesh_iss_distinct_points_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


template < std::size_t N, typename PtForIter, typename PtOutIter, typename IntOutIter,
         typename T >
inline
void
buildDistinctPoints(PtForIter pointsBeginning, PtForIter pointsEnd,
                    PtOutIter distinctPointsOutput,
                    IntOutIter indicesOutput,
                    const T minDistance) {
   // Get the point type from the input iterator.
   typedef typename std::iterator_traits<PtForIter>::value_type Point;
   // The container for the points.
   typedef std::vector<Point> PointContainer;
   // A record is a const iterator on points.
   typedef typename PointContainer::const_iterator Record;
   // The ORQ data structure.
   typedef geom::CellArray<N, ads::Dereference<Record> > ORQDS;
   // The bounding box type.
   typedef typename ORQDS::BBox BBox;
   // A Cartesian point.
   typedef std::array<T, N> CartesianPoint;

   // If there are no points, do nothing.
   if (pointsBeginning == pointsEnd) {
      return;
   }

   const T minSquaredDistance = minDistance * minDistance;
   // A container to hold the distinct points.
   PointContainer distinctPoints;
   // Reserve memory for these records.  This is necessary because the ORQ
   // data structure stores pointers to the records.  Resizing the array
   // would invalidate the pointers.
   distinctPoints.reserve(std::distance(pointsBeginning, pointsEnd));

   //
   // Make the ORQ data structure.
   //

   // Make a bounding box around the points.
   BBox domain = specificBBox<BBox>(pointsBeginning, pointsEnd);
   // Choose a box size so that there will be about 10 boxes in each dimension.
   CartesianPoint delta = domain.upper;
   delta -= domain.lower;
   delta /= T(9.0);
   // Make a semi-open interval that contains the points.
   domain.upper = domain.upper + delta;
   // Make the ORQ data structure.
   ORQDS orqds(delta, domain);

   // The window containing close records.
   BBox window;
   // The radius of the search window.
   CartesianPoint offset;
   std::fill(offset.begin(), offset.end(), minDistance);
   // The vector to hold the candidate records.
   std::vector<Record> candidates;

   Point p;
   std::size_t m;
   // Loop over the points.
   for (; pointsBeginning != pointsEnd; ++pointsBeginning) {
      // The point.
      p = *pointsBeginning;
      // Make the query window.
      window.lower = p - offset;
      window.upper = p + offset;
      // Do the window query.
      candidates.clear();
      orqds.computeWindowQuery(std::back_inserter(candidates), window);
      // Loop over the candidate points.
      const std::size_t size = candidates.size();
      for (m = 0; m != size; ++m) {
         if (ext::squaredDistance(p, *candidates[m]) <= minSquaredDistance) {
            break;
         }
      }
      // If we did not find a close point.
      if (m == size) {
         // The index of the distinct point.
         *indicesOutput++ = int(distinctPoints.size());
         // Add another distinct point.
         distinctPoints.push_back(p);
         orqds.insert(distinctPoints.end() - 1);
      }
      else {
         // The index of the distinct point.
         *indicesOutput++ = int(candidates[m] - distinctPoints.begin());
      }
   }
   // Write the distinct points to the output iterator.
   const std::size_t size = distinctPoints.size();
   for (m = 0; m != size; ++m) {
      *distinctPointsOutput++ = distinctPoints[m];
   }
}




// Internal function for computing the maximum distance from the origin.
// The return type is the number type.
template<typename PtForIter>
inline
typename std::iterator_traits<PtForIter>::value_type::value_type
_computeMaxDistanceFromOrigin(PtForIter pointsBeginning,
                              PtForIter pointsEnd) {
   // Get the point type from the forward iterator.
   typedef typename std::iterator_traits<PtForIter>::value_type Point;
   // Get the number type from the point type.
   typedef typename Point::value_type Number;

   //
   // Determine the max distance from the origin.
   //
   // Initialize to zero.
   Point p = {{}};
   Number d;
   Number maxDistance = 0;
   for (PtForIter i = pointsBeginning; i != pointsEnd; ++i) {
     d = ext::squaredDistance(p, *i);
      if (d > maxDistance) {
         maxDistance = d;
      }
   }
   return std::sqrt(maxDistance);
}




template<std::size_t N, typename PtForIter, typename PtOutIter, typename IntOutIter>
inline
void
buildDistinctPoints(PtForIter pointsBeginning, PtForIter pointsEnd,
PtOutIter distinctPoints, IntOutIter indices) {
   // Get the point type from the forward iterator.
   typedef typename std::iterator_traits<PtForIter>::value_type Point;
   // Get the number type from the point type.
   typedef typename Point::value_type Number;

   // Determine the max distance from the origin.
   const Number maxDistance =
   _computeMaxDistanceFromOrigin(pointsBeginning, pointsEnd);

   // Call distinct_points() with a minimum distance.
   buildDistinctPoints<N>(pointsBeginning, pointsEnd,
   distinctPoints, indices,
   maxDistance *
   std::sqrt(std::numeric_limits<Number>::epsilon()));
}



// Remove duplicate vertices.
template<std::size_t N, std::size_t M, typename T>
inline
void
removeDuplicateVertices(IndSimpSet<N, M, T>* x, const T minDistance) {
   typedef IndSimpSet<N, M, T> ISS;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::IndexedSimplexIterator IndexedSimplexIterator;

   //
   // Get the indexed set of distinct vertices.
   //
   std::vector<Vertex> vertices;
   std::vector<std::size_t> indices;
   buildDistinctPoints<N>(x->vertices.begin(), x->vertices.end(),
                          std::back_inserter(vertices),
                          std::back_inserter(indices), minDistance);

   // If all of the vertices are distinct.
   if (vertices.size() == x->vertices.size()) {
      // Do nothing.
      return;
   }

   assert(vertices.size() < x->vertices.size());
   assert(indices.size() == x->vertices.size());

   //
   // Update the vertices.
   //
   vertices.swap(x->vertices);

   //
   // Update the indexed simplices.
   //
   std::size_t m = 0;
   const IndexedSimplexIterator iEnd = x->indexedSimplices.end();
   // For each indexed simplex.
   for (IndexedSimplexIterator i = x->indexedSimplices.begin();
   i != iEnd; ++i) {
      // For each vertex index.
      for (m = 0; m != M + 1; ++m) {
         // Update the index.
         (*i)[m] = indices[(*i)[m]];
      }
   }

   //
   // Update the topology.
   //
   x->updateTopology();
}


// Remove duplicate vertices.
template<std::size_t N, std::size_t M, typename T>
inline
void
removeDuplicateVertices(IndSimpSet<N, M, T>* x) {
   // Determine the max distance from the origin.
   const T maxDistance =
   _computeMaxDistanceFromOrigin(x->vertices.begin(),
   x->vertices.end());

   // Call the above function with an approprate minimum distance.
   removeDuplicateVertices
   (x, maxDistance * std::sqrt(std::numeric_limits<T>::epsilon()));
}

} // namespace geom
}
