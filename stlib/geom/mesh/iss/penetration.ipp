// -*- C++ -*-

#if !defined(__geom_mesh_iss_penetration_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

template < std::size_t N, typename T,
         typename PointRandomAccessIterator,
         typename TupleOutputIterator >
inline
std::size_t
reportPenetrations(const IndSimpSetIncAdj<N, N, T>& mesh,
                   PointRandomAccessIterator pointsBeginning,
                   PointRandomAccessIterator pointsEnd,
                   TupleOutputIterator penetrations) {
   typedef IndSimpSetIncAdj<N, N, T> Mesh;
   typedef IndSimpSet < N, N - 1, T > Boundary;
   typedef ISS_SignedDistance<Boundary, N> SignedDistance;
   typedef typename Mesh::Number Number;
   typedef typename Mesh::Vertex Point;
   typedef typename Mesh::Simplex Simplex;

   typedef CellArrayStatic<N, ads::Dereference<PointRandomAccessIterator> > Orq;

   typedef std::map<std::size_t, std::pair<std::size_t, Number> > Map;
   typedef typename Map::value_type MapValue;

   // If there are no points, do nothing.
   if (pointsBeginning == pointsEnd) {
      return 0;
   }

   Orq orq(pointsBeginning, pointsEnd);

   //
   // Find the points that are inside simplices.
   //
   std::vector<PointRandomAccessIterator> pointsInside;
   Map indexAndDistance;
   Simplex simplex;
   BBox<Number, N> bbox;
   Number distance;
   // For each simplex in the mesh.
   for (std::size_t simplexIndex = 0; simplexIndex != mesh.indexedSimplices.size();
         ++simplexIndex) {
      mesh.getSimplex(simplexIndex, &simplex);
      // Make a bounding box around the simplex.
      computeBBox(simplex, &bbox);
      // Get the points in the bounding box.
      pointsInside.clear();
      orq.computeWindowQuery(std::back_inserter(pointsInside), bbox);
      // For each point inside the bounding box.
      for (typename std::vector<PointRandomAccessIterator>::const_iterator
            i = pointsInside.begin(); i != pointsInside.end(); ++i) {
         const Point& point = **i;
         // If the point is inside the simplex.
         if (isIn(simplex, point)) {
            const std::size_t pointIndex = *i - pointsBeginning;
            // Compute the distance to the simplex.
            distance = computeDistanceInterior(simplex, point);
            // See if this point is inside another simplex.
            typename Map::iterator i = indexAndDistance.find(pointIndex);
            // If this point has not yet been found in another simplex.
            if (i == indexAndDistance.end()) {
               indexAndDistance.insert(MapValue(pointIndex,
                                                std::make_pair(simplexIndex,
                                                      distance)));
            }
            // If this point is also in another simplex, but is further inside
            // this one.
            else if (distance < i->second.second) {
               i->second.first = simplexIndex;
               i->second.second = distance;
            }
         }
      }
   }

   // Early exit if there are no penetrating points.
   if (indexAndDistance.empty()) {
      return 0;
   }

   //
   // Determine the closest points for the penetrating points
   //
   // Make the boundary mesh.
   Boundary boundary;
   buildBoundary(mesh, &boundary);
   // Make the data structure for computing the signed distance and closest
   // point on the boundary.
   SignedDistance signedDistance(boundary);
   Point closestPoint;
   // For each point inside a simplex.
   for (typename Map::const_iterator i = indexAndDistance.begin();
         i != indexAndDistance.end(); ++i) {
      // Compute the closest point on the boundary. Ignore the signed distance.
      signedDistance(pointsBeginning[i->first], &closestPoint);
      // Record the penetration: point index, simplex index, and closest point.
      *penetrations++ = std::make_tuple(i->first, i->second.first,
                                             closestPoint);
   }

   // Return the number of penetrations.
   return indexAndDistance.size();
}

// Return the maximum incident edge length.
template<std::size_t N, typename T>
inline
T
maximumIncidentEdgeLength(const IndSimpSetIncAdj<N, N, T>& mesh,
                          const std::size_t n) {
   typedef IndSimpSetIncAdj<N, N, T> Mesh;

   T d, length = 0;
   // For each incident simplex.
   for (std::size_t s = 0; s != mesh.incident.size(n); ++s) {
      // For each edge.
      for (std::size_t i = 0; i != Mesh::M; ++i) {
         for (std::size_t j = i + 1; j != Mesh::M + 1; ++j) {
            d = ext::squaredDistance(mesh.getSimplexVertex(s, i),
                                     mesh.getSimplexVertex(s, j));
            if (d > length) {
               length = d;
            }
         }
      }
   }
   return std::sqrt(length);
}

// Report the maximum relative penetration for a boundary node.
template<std::size_t N, typename T>
inline
T
maximumRelativePenetration(const IndSimpSetIncAdj<N, N, T>& mesh) {
   typedef IndSimpSetIncAdj<N, N, T> Mesh;
   typedef typename Mesh::Vertex Vertex;
   typedef std::tuple<std::size_t, std::size_t, Vertex> Record;

   // Check for the trivial case of an empty mesh.
   if (mesh.indexedSimplices.size() == 0) {
      return 0;
   }
   // Label the components of the mesh.
   std::vector<std::size_t> labels;
   const std::size_t numComponents = labelComponents(mesh, &labels);
   // If there is only one component, there are no penetrations.
   if (numComponents == 1) {
      return 0.;
   }
   // The simplex indices for each component.
   std::vector<std::vector<std::size_t> > indices(numComponents);
   for (std::size_t i = 0; i != labels.size(); ++i) {
      indices[labels[i]].push_back(i);
   }
   // Separate the mesh into connected components.
   std::vector<Mesh> components(numComponents);
   std::vector<std::vector<std::size_t> > boundaryIndices(numComponents);
   for (std::size_t i = 0; i != indices.size(); ++i) {
      buildFromSubsetSimplices(mesh, indices[i].begin(), indices[i].end(),
                               &components[i]);
      determineBoundaryVertices(components[i],
                                std::back_inserter(boundaryIndices[i]));
   }

   T maxRelPen = 0;
   std::vector<std::size_t> componentIndices;
   std::vector<std::size_t> vertexIndices;
   std::vector<Vertex> vertices;
   std::vector<Record> penetrations;
   // For each component.
   for (std::size_t i = 0; i != components.size(); ++i) {
      componentIndices.clear();
      vertexIndices.clear();
      vertices.clear();
      penetrations.clear();
      // For each other component.
      for (std::size_t j = 0; j != components.size(); ++j) {
         if (j == i) {
            continue;
         }
         // Collect the boundary vertices.
         for (std::size_t k = 0; k != boundaryIndices[j].size(); ++k) {
            componentIndices.push_back(j);
            vertexIndices.push_back(boundaryIndices[j][k]);
            vertices.push_back(components[j].vertices[boundaryIndices[j][k]]);
         }
      }
      // Get the penetrations.
      reportPenetrations(components[i], vertices.begin(), vertices.end(),
                         std::back_inserter(penetrations));
      for (std::size_t j = 0; j != penetrations.size(); ++j) {
         // The penetration index.
         const std::size_t pi = std::get<0>(penetrations[j]);
         T d = ext::euclideanDistance(vertices[pi],
                                      std::get<2>(penetrations[j]));
         const std::size_t ci = componentIndices[pi];
         const std::size_t vi = vertexIndices[pi];
         // Divide by the maximum incident edge length.
         const T length = maximumIncidentEdgeLength(components[ci], vi);
         if (length != 0) {
            d /= length;
            if (d > maxRelPen) {
               maxRelPen = d;
            }
         }
      }
   }
   return maxRelPen;
}

} // namespace geom
}
