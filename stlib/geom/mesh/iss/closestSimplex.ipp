// -*- C++ -*-

#if !defined(__geom_mesh_iss_closestSimplex_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


template<template<std::size_t, typename> class _Orq, std::size_t SpaceD,
         typename _T>
inline
void
closestSimplex(const IndSimpSet<SpaceD, SpaceD, _T>& mesh,
               const std::vector<std::array<_T, SpaceD> >& points,
               std::vector<std::size_t>* indices) {
   typedef std::array<_T, SpaceD> Point;
   typedef typename std::vector<Point>::const_iterator Record;
   typedef IndSimpSet<SpaceD, SpaceD, _T> Mesh;
   typedef typename Mesh::Simplex Simplex;

   // Initialize the indices.
   indices->resize(points.size());
   std::fill(indices->begin(), indices->end(),
             std::numeric_limits<std::size_t>::max());
   // Trivial case.
   if (indices->empty()) {
      return;
   }
   // The distance to the closest simplex.
   std::vector<_T> distance(points.size(), std::numeric_limits<_T>::infinity());
   // Store the points in an ORQ data structure.
   _Orq<SpaceD, ads::Dereference<Record> > orq(points.begin(), points.end());

   // Construct a bounding ball around each simplex.
   std::vector<Ball<_T, SpaceD> > balls(mesh.indexedSimplices.size());
   Simplex simplex;
   _T minRadius = std::numeric_limits<_T>::max();
   for (std::size_t i = 0; i != mesh.indexedSimplices.size(); ++i) {
      mesh.getSimplex(i, &simplex);
      // First compute the center.
      Point& center = balls[i].center;
      center = simplex[0];
      for (std::size_t j = 1; j != simplex.size(); ++j) {
         center += simplex[j];
      }
      center /= _T(simplex.size());
      // Then the radius.
      _T squaredRadius = 0;
      for (std::size_t j = 0; j != simplex.size(); ++j) {
         const _T d2 = ext::squaredDistance(center, simplex[j]);
         if (d2 > squaredRadius) {
            squaredRadius = d2;
         }
      }
      balls[i].radius = std::sqrt(squaredRadius);
      minRadius = std::min(minRadius, balls[i].radius);
   }

   // Determine an appropriate epsilon. We simply pick 0.1 times the minimum
   //  radius of the balls. Note that the choice of epsilon is not critical.
   const _T epsilon = 0.1 * minRadius;

   // Compute the distance from each simplex to the points in its neighborhood.
   SimplexDistance<SpaceD, SpaceD, _T> simplexDistance;
   std::vector<Record> records;
   std::vector<std::size_t> candidates;
   std::back_insert_iterator<std::vector<Record> > backInserter(records);
   for (std::size_t i = 0; i != mesh.indexedSimplices.size(); ++i) {
      mesh.getSimplex(i, &simplex);
      // Put a bounding box around the simplex and enlarge it by epsilon.
      BBox<_T, SpaceD> box =
        specificBBox<BBox<_T, SpaceD> >(simplex.begin(), simplex.end());
      offset(&box, epsilon);
      // Get the points in the box.
      records.clear();
      orq.computeWindowQuery(backInserter, box);
      // Gather the points that might be within epsilon of the simplex. We
      // check this by computing the distance to the bounding ball.
      candidates.clear();
      for (std::size_t j = 0; j != records.size(); ++j) {
         if (ext::euclideanDistance(balls[i].center, *records[j]) -
             balls[i].radius <= epsilon) {
            // Convert from a record (point iterator) to a point index.
            candidates.push_back(std::distance(points.begin(), records[j]));
         }
      }
      // If there are no candidates, continue with the next simplex.
      if (candidates.empty()) {
         continue;
      }
      // Initialize the data structure for computing distance.
      simplexDistance.initialize(simplex);
      // Compute the distance to the simplex for the candidate points.
      for (std::size_t j = 0; j != candidates.size(); ++j) {
         const std::size_t n = candidates[j];
         const _T d = simplexDistance(points[n]);
         if (d < distance[n]) {
            distance[n] = d;
            (*indices)[n] = i;
         }
      }
   }

   // Finally, deal with the points that are farther than epsilon from all
   // simplices.
   for (std::size_t i = 0; i != distance.size(); ++i) {
      if (distance[i] <= epsilon) {
         continue;
      }
      // Compute the distance to each simplex.
      for (std::size_t j = 0; j != balls.size(); ++j) {
         // Get a lower bound on the distance by computing the signed distance
         // to the bounding ball. If the lower bound is greater than the
         // current distance, continue to the next simplex.
         if (ext::euclideanDistance(balls[j].center, points[i]) -
             balls[j].radius > distance[i]) {
            continue;
         }
         // Initialize the data structure for computing distance.
         mesh.getSimplex(j, &simplex);
         simplexDistance.initialize(simplex);
         const _T d = simplexDistance(points[i]);
         if (d < distance[i]) {
            distance[i] = d;
            (*indices)[i] = j;
         }
      }
   }
#ifdef STLIB_DEBUG
   for (std::size_t i = 0; i != indices->size(); ++i) {
      assert((*indices)[i] < mesh.indexedSimplices.size());
   }
#endif
}


} // namespace geom
}
