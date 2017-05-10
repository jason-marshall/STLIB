// -*- C++ -*-

#if !defined(__geom_mesh_iss_optimize_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


//-------------------------------------------------------------------------
// Interior
//-------------------------------------------------------------------------


// Optimize the position of all interior vertices.
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T >
inline
void
geometricOptimizeInterior(IndSimpSetIncAdj<N, N, T>* mesh,
                          const std::size_t numSweeps) {
   // Get the interior vertices.
   std::vector<std::size_t> interiorIndices;
   determineInteriorVertices(*mesh, std::back_inserter(interiorIndices));
   // Perform the sweeps over the interior vertices.
   geometricOptimizeInterior<QF>
   (mesh, interiorIndices.begin(), interiorIndices.end(), numSweeps);
}






// Make \c numSweeps optimization sweeps over the given interior vertices with
// the quality function given as a template parameter.
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T,
         typename IntForIter >
inline
void
geometricOptimizeInterior(IndSimpSetIncAdj<N, N, T>* mesh,
                          IntForIter begin, IntForIter end,
                          std::size_t numSweeps) {
   typedef IndSimpSetIncAdj<N, N, T> ISS;
   typedef typename ISS::Number Number;
   typedef typename ISS::SimplexFace SimplexFace;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::IncidenceConstIterator IncidenceConstIterator;
   typedef ComplexWithFreeVertex<QF, N, Number> Complex;
   typedef ComplexNorm2Mod<Complex> ComplexQuality;

   // A local mesh.
   Complex sc;
   // The neighboring faces of a vertex to be moved.
   std::vector<SimplexFace> neighboringFaces;
   // Quality function of the local mesh.
   ComplexQuality function(sc);
   // The optimization class.
   numerical::QuasiNewton<N, ComplexQuality> method(function);
   // CONTINUE Set the x tolerance.
   SimplexFace face;
   Vertex oldPosition;
   BBox<Number, N> bb;
   Number radius = 0;

   // Make numSweeps sweeps.
   std::size_t n = 0;
   while (numSweeps-- > 0) {
      // For each vertex.
      for (IntForIter vi = begin; vi != end; ++vi) {
         // The index of the vertex.
         n = *vi;

         // If the vertex has no incident simplices, do nothing.
         if (mesh->incident.empty(n)) {
            continue;
         }

         //
         // Build the local mesh.
         //
         neighboringFaces.clear();
         // For each incident simplex index to the n_th vertex.
         for (IncidenceConstIterator indexIter = mesh->incident.begin(n);
               indexIter != mesh->incident.end(n); ++indexIter) {
            getFace(*mesh, *indexIter, n, &face);
            neighboringFaces.push_back(face);
         }
         sc.set(neighboringFaces.begin(), neighboringFaces.end());

         // Calculate an approximate radius for the complex.
         sc.computeBBox(mesh->vertices[n], &bb);
         radius = 0.0;
         for (std::size_t j = 0; j != N; ++j) {
            radius += bb.upper[j] - bb.lower[j];
         }
         radius /= (2 * N);

         //
         // Optimize the position of the free vertex.
         //
         try {
            oldPosition = mesh->vertices[n];
            const Number oldValue = function(oldPosition);
            // Initialize to avoid a warning.
            Number value = 0;
            std::size_t numIterations;
            method.find_minimum(mesh->vertices[n], value, numIterations,
                                0.1 * radius);
            // If the new position is not better than the old one.
            if (value >= oldValue) {
               // Undo the change.
               mesh->vertices[n] = oldPosition;
            }
         }
         catch (...) {
            std::cerr << "Error in optimizing vertex " << n
                      << " in geometricOptimizeInterior().\n"
                      << "radius = " << radius << '\n';
         }
      }
   }
}







//-------------------------------------------------------------------------
// Boundary
//-------------------------------------------------------------------------


// Optimize the position of all boundary vertices.
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T, std::size_t SD >
inline
void
geometricOptimizeBoundary(IndSimpSetIncAdj<N, N, T>* mesh,
                          PointsOnManifold<N, N-1, SD, T>* boundaryManifold,
                          const std::size_t numSweeps) {
   // Get the boundary vertices.
   std::vector<std::size_t> boundaryIndices;
   determineBoundaryVertices(*mesh, std::back_inserter(boundaryIndices));
   // Perform the sweeps over the boundary vertices.
   geometricOptimizeBoundary<QF>
      (mesh, boundaryIndices.begin(), boundaryIndices.end(), boundaryManifold,
       numSweeps);
}






// Make \c numSweeps optimization sweeps over the given boundary vertices with
// the quality function given as a template parameter.
template < template<std::size_t, typename> class QF,
         typename T,
         typename IntForIter, std::size_t SD >
inline
void
geometricOptimizeBoundary(IndSimpSetIncAdj<2, 2, T>* mesh,
                          IntForIter begin, IntForIter end,
                          PointsOnManifold<2, 1, SD, T>* boundaryManifold,
                          std::size_t numSweeps) {
   // The space dimension.
   const std::size_t N = 2;

   typedef IndSimpSetIncAdj<N, N, T> ISS;
   typedef typename ISS::Number Number;
   typedef typename ISS::SimplexFace SimplexFace;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::IncidenceConstIterator IncidenceConstIterator;
   typedef ParametrizedLine<N, Number> Manifold;
   typedef ComplexWithFreeVertexOnManifold < QF, N, N - 1, Manifold, Number > Complex;
   typedef ComplexManifoldNorm2Mod<Complex> ComplexQuality;
   typedef typename Complex::ManifoldPoint ManifoldPoint;

   // A local mesh.
   Complex sc;
   // The neighboring faces of a vertex to be moved.
   std::vector<SimplexFace> neighboringFaces;
   // Quality function of the local mesh.
   ComplexQuality function(sc);
   // The optimization class.
   numerical::Simplex < N - 1, ComplexQuality > method(function);
   method.set_max_function_calls(100);
   method.set_are_checking_function_calls(false);

   SimplexFace face;
   Vertex source, target;
   ManifoldPoint manifoldPoint;
   const std::array<ManifoldPoint, 2>
   startingSimplex = {{
         ext::filled_array<ManifoldPoint>(0.0),
         ext::filled_array<ManifoldPoint>(1.0)
      }
   };

   // Set the manifold for the simplicial complex here (we will modify the
   // line below).
   ParametrizedLine<N, Number> line;
   sc.setManifold(&line);

   // Make numSweeps sweeps.
   std::size_t n = 0;
   while (numSweeps-- > 0) {
      // For each vertex.
      for (IntForIter vi = begin; vi != end; ++vi) {
         // The index of the vertex.
         n = *vi;

         // If the vertex has no incident simplices, do nothing.
         if (mesh->incident.empty(n)) {
            continue;
         }

         // If the vertex is a corner feature, we cannot move it.
         if (boundaryManifold->isOnCorner(n)) {
            continue;
         }
         // Otherwise, it is a surface feature.

         //
         // Build the local mesh.
         //
         neighboringFaces.clear();
         // For each incident simplex index to the n_th vertex.
         for (IncidenceConstIterator indexIter = mesh->incident.begin(n);
               indexIter != mesh->incident.end(n); ++indexIter) {
            getFace(*mesh, *indexIter, n, &face);
            neighboringFaces.push_back(face);
         }
         sc.set(neighboringFaces.begin(), neighboringFaces.end());

         // Get the simplices in the surface manifold in a neighborhood
         // of the vertex.
         std::vector<std::size_t> neighborhood;
         boundaryManifold->getNeighborhood(n, std::back_inserter(neighborhood));

         //
         // Optimize the position of the free vertex.
         //
         Number oldValue = sc.computeNorm2Modified(mesh->vertices[n]);
         Number value;
         // For each simplex in the neighborhood in the boundary manifold.
         for (std::vector<std::size_t>::const_iterator i = neighborhood.begin();
               i != neighborhood.end(); ++i) {
            const std::size_t simplexIndex = *i;
            // The source and target for the line segment that comprises the
            // surface simplex in the boundary manifold.
            source = boundaryManifold->getSurfaceSimplexVertex(simplexIndex, 0);
            target = boundaryManifold->getSurfaceSimplexVertex(simplexIndex, 1);

            //
            // First check the endpoints.
            //

            // The source.
            value = sc.computeNorm2Modified(source);
            if (value < oldValue) {
               mesh->vertices[n] = source;
               oldValue = value;
               // Update the point on the manifold.
               boundaryManifold->changeSurfaceSimplex(n, simplexIndex);
            }

            // The target.
            value = sc.computeNorm2Modified(target);
            if (value < oldValue) {
               mesh->vertices[n] = target;
               oldValue = value;
               // Update the point on the manifold.
               boundaryManifold->changeSurfaceSimplex(n, simplexIndex);
            }

            //
            // Then try optimizing on the line segment.
            //

            // Build the parametrized line.
            line.build(source, target);

            // Optimize in parameter space.  Ignore the return value.  It is not
            // necessary that the minimization converge.
            method.find_minimum(startingSimplex);
            manifoldPoint = method.minimum_point();
            value = method.minimum_value();
            // CONTINUE: REMOVE
            //std::cerr << oldValue << " " << value << " " << manifoldPoint[0] << " "
            //  << method.num_function_calls() << "\n";
            // If the minimum value is an improvement and if the point is in
            // the interior of the line segment.
            if (value < oldValue && 0 < manifoldPoint[0] &&
                  manifoldPoint[0] < 1) {
               // CONTINUE: REMOVE
               //std::cerr << "Success\n";
               // Calculate the new point from the parameter value.
               // Set the location in the mesh.
               mesh->vertices[n] = line.computePosition(manifoldPoint);
               // Update the point on the manifold.
               boundaryManifold->changeSurfaceSimplex(n, simplexIndex);
            }
         } // End loop over the boundary manifold simplices.
      }
   }
}





// Make \c numSweeps optimization sweeps over the given boundary vertices with
// the quality function given as a template parameter.
template < template<std::size_t, typename> class QF,
         typename T,
         typename IntForIter, std::size_t SD >
inline
void
geometricOptimizeBoundary(IndSimpSetIncAdj<3, 3, T>* mesh,
                          IntForIter begin, IntForIter end,
                          PointsOnManifold<3, 2, SD, T>* boundaryManifold,
                          std::size_t numSweeps)  {
   // The space dimension.
   const std::size_t N = 3;

   typedef IndSimpSetIncAdj<N, N, T> ISS;
   typedef typename ISS::Number Number;
   typedef typename ISS::SimplexFace SimplexFace;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::IncidenceConstIterator IncidenceConstIterator;

   typedef ParametrizedPlane<N, Number> SurfaceManifold;
   typedef ComplexWithFreeVertexOnManifold < QF, N, N - 1, SurfaceManifold, Number >
   SurfaceComplex;
   typedef ComplexManifoldNorm2Mod<SurfaceComplex> SurfaceComplexQuality;
   typedef typename SurfaceComplex::ManifoldPoint SurfacePoint;

   typedef ParametrizedLine<N, Number> EdgeManifold;
   typedef ComplexWithFreeVertexOnManifold < QF, N, N - 2, EdgeManifold, Number >
   EdgeComplex;
   typedef ComplexManifoldNorm2Mod<EdgeComplex> EdgeComplexQuality;
   typedef typename EdgeComplex::ManifoldPoint EdgePoint;

   // A local mesh for optimizing on surface features.
   SurfaceComplex surfaceComplex;
   // A local mesh for optimizing on edge features.
   EdgeComplex edgeComplex;
   // The neighboring faces of a vertex to be moved.
   std::vector<SimplexFace> neighboringFaces;
   // Quality function of the local mesh for optimizing on surface features.
   SurfaceComplexQuality surfaceFunction(surfaceComplex);
   // Quality function of the local mesh for optimizing on edge features.
   EdgeComplexQuality edgeFunction(edgeComplex);

   // The optimization class for surface features.
   numerical::Simplex < N - 1, SurfaceComplexQuality > surfaceMethod(surfaceFunction);
   surfaceMethod.set_max_function_calls(1000);
   surfaceMethod.set_are_checking_function_calls(false);
   // The optimization class for edge features.
   numerical::Simplex < N - 2, EdgeComplexQuality > edgeMethod(edgeFunction);
   edgeMethod.set_max_function_calls(100);
   edgeMethod.set_are_checking_function_calls(false);

   SimplexFace face;
   // Source and target for a line segment.
   Vertex source, target;
   // Vertices for a triangle.
   std::array<Vertex, 3> vertices;
   // Parameter for the plane.
   SurfacePoint surfacePoint;
   // Parameter for the line.
   EdgePoint edgePoint;
   // The starting simplex for surface optimization.
   const std::array<SurfacePoint, N>
   startingSurfaceSimplex = {{
       {{0., 0.}},
       {{1., 0.}},
       {{0., 1.}}
      }
   };
   // The starting simplex for edge optimization.
   const std::array < EdgePoint, N - 1 >
   startingEdgeSimplex = {{
       {{0.0}},
       {{1.0}}
      }
   };
   // The indices of simplices in a neighborhood of the vertex.
   std::vector<std::size_t> neighborhood;

   // Set the manifold for the simplicial complex here (we will modify the
   // plane and line below).
   ParametrizedPlane<N, Number> plane;
   surfaceComplex.setManifold(&plane);
   ParametrizedLine<N, Number> line;
   edgeComplex.setManifold(&line);

   // Make numSweeps sweeps.
   std::size_t n = 0;
   while (numSweeps-- > 0) {
      // For each vertex.
      for (IntForIter vi = begin; vi != end; ++vi) {
         // The index of the vertex.
         n = *vi;

         // If the vertex has no incident simplices, do nothing.
         if (mesh->incident.empty(n)) {
            continue;
         }

         // If the vertex is a corner feature, we cannot move it.
         if (boundaryManifold->isOnCorner(n)) {
            continue;
         }
         // Otherwise, it is a surface feature or an edge feature.

         //
         // Get the faces for building the local mesh.
         //
         neighboringFaces.clear();
         // For each incident simplex index to the n_th vertex.
         for (IncidenceConstIterator indexIter = mesh->incident.begin(n);
               indexIter != mesh->incident.end(n); ++indexIter) {
            getFace(*mesh, *indexIter, n, &face);
            neighboringFaces.push_back(face);
         }

         // Build the complex for optimization on edges.
         // This is needed whether the vertex is on an edge feature or a
         // surface feature.
         edgeComplex.set(neighboringFaces.begin(), neighboringFaces.end());

         // If the vertex is on an edge feature.
         if (boundaryManifold->isOnEdge(n)) {
            // Get the simplices in the surface manifold in a neighborhood
            // of the vertex.
            neighborhood.clear();
            boundaryManifold->
            getNeighborhood(n, std::back_inserter(neighborhood));

            //
            // Optimize the position of the free vertex.
            //
            Number oldValue =
               edgeComplex.computeNorm2Modified(mesh->vertices[n]);
            Number value;
            // For each edge simplex in the neighborhood in the boundary manifold.
            for (std::vector<std::size_t>::const_iterator i = neighborhood.begin();
                  i != neighborhood.end(); ++i) {
               const std::size_t simplexIndex = *i;
               // The source and target for the line segment that comprises the
               // edge simplex in the boundary manifold.
               source = boundaryManifold->getEdgeSimplexVertex(simplexIndex, 0);
               target = boundaryManifold->getEdgeSimplexVertex(simplexIndex, 1);

               //
               // First check the endpoints.
               //

               // The source.
               value = edgeComplex.computeNorm2Modified(source);
               if (value < oldValue) {
                  mesh->vertices[n] = source;
                  oldValue = value;
                  // Update the point on the manifold.
                  boundaryManifold->changeEdgeSimplex(n, simplexIndex);
               }

               // The target.
               value = edgeComplex.computeNorm2Modified(target);
               if (value < oldValue) {
                  mesh->vertices[n] = target;
                  oldValue = value;
                  // Update the point on the manifold.
                  boundaryManifold->changeEdgeSimplex(n, simplexIndex);
               }

               //
               // Then try optimizing on the line segment.
               //

               // Build the parametrized plane.
               line.build(source, target);

               // Optimize in parameter space.  Ignore the return value.  It is not
               // necessary that the minimization converge.
               edgeMethod.find_minimum(startingEdgeSimplex);
               edgePoint = edgeMethod.minimum_point();
               value = edgeMethod.minimum_value();
               // If the minimum value is an improvement and if the point is in
               // the interior of the line segment.
               if (value < oldValue && 0 < edgePoint[0] && edgePoint[0] < 1) {
                  // Calculate the new point from the parameter value.
                  // Set the location in the mesh.
                  mesh->vertices[n] = line.computePosition(edgePoint);
                  // Update the point on the manifold.
                  boundaryManifold->changeEdgeSimplex(n, simplexIndex);
               }
            } // End loop over the boundary manifold edge simplices.
         } // End: If the vertex is on an edge feature.
         else {
            assert(boundaryManifold->isOnSurface(n));

            // Build the complex for optimization on surface features.
            surfaceComplex.set(neighboringFaces.begin(), neighboringFaces.end());

            // Get the simplices in the surface manifold in a neighborhood
            // of the vertex.
            neighborhood.clear();
            boundaryManifold->
            getNeighborhood(n, std::back_inserter(neighborhood));

            //
            // Optimize the position of the free vertex.
            //
            Number oldValue =
               surfaceComplex.computeNorm2Modified(mesh->vertices[n]);
            Number value;
            // For each edge simplex in the neighborhood in the boundary manifold.
            for (std::vector<std::size_t>::const_iterator i = neighborhood.begin();
                  i != neighborhood.end(); ++i) {
               const std::size_t simplexIndex = *i;
               // The three vertices of the triangle face.
               for (std::size_t j = 0; j != 3; ++j) {
                  vertices[j] =
                     boundaryManifold->getSurfaceSimplexVertex(simplexIndex, j);
               }

               //
               // First check the corners.
               //
               for (std::size_t j = 0; j != 3; ++j) {
                  value = surfaceComplex.computeNorm2Modified(vertices[j]);
                  if (value < oldValue) {
                     mesh->vertices[n] = vertices[j];
                     oldValue = value;
                     // Update the point on the manifold.
                     boundaryManifold->changeSurfaceSimplex(n, simplexIndex);
                  }
               }

               //
               // Then try optimizing on the line segments.
               //
               for (std::size_t j = 0; j != 3; ++j) {
                  // Build the parametrized line.
                  line.build(vertices[j], vertices[(j + 1) % 3]);

                  // Optimize in parameter space.  Ignore the return value.  It
                  // is not necessary that the minimization converge.
                  edgeMethod.find_minimum(startingEdgeSimplex);
                  edgePoint = edgeMethod.minimum_point();
                  value = edgeMethod.minimum_value();
                  // If the minimum value is an improvement and if the point
                  // is in the interior of the line segment.
                  if (value < oldValue && 0 < edgePoint[0] && edgePoint[0] < 1) {
                     // Calculate the new point from the parameter value.
                     // Set the location in the mesh.
                     mesh->vertices[n] = line.computePosition(edgePoint);
                     // Update the point on the manifold.
                     boundaryManifold->changeSurfaceSimplex(n, simplexIndex);
                  }
               }

               //
               // Finally, try optimizing on the plane of the triangle face.
               //

               // Build the parametrized plane.
               plane.build(vertices[0], vertices[1], vertices[2]);

               // Optimize in parameter space.  Ignore the return value.  It is not
               // necessary that the minimization converge.
               surfaceMethod.find_minimum(startingSurfaceSimplex);
               surfacePoint = surfaceMethod.minimum_point();
               value = surfaceMethod.minimum_value();
               // If the minimum value is an improvement and if the point is in
               // the interior of the triangle segment.
               if (value < oldValue &&
                     0 < surfacePoint[0] && 0 < surfacePoint[1] &&
                     surfacePoint[0] + surfacePoint[1] < 1) {
                  // Calculate the new point from the parameter value.
                  // Set the location in the mesh.
                  mesh->vertices[n] = plane.computePosition(surfacePoint);
                  // Update the point on the manifold.
                  boundaryManifold->changeSurfaceSimplex(n, simplexIndex);
               }
            } // End loop over the boundary manifold surface simplices.
         } // End: If the vertex is on a surface feature.
      } // For each vertex.
   } // Sweeps.
}






// Make \c numSweeps optimization sweeps over the given boundary vertices with
// the quality function given as a template parameter.
template < template<std::size_t, typename> class QF,
         typename T,
         typename IntForIter, std::size_t SD >
inline
void
geometricOptimizeBoundary(IndSimpSetIncAdj<3, 2, T>* /*mesh*/,
                          IntForIter /*begin*/, IntForIter /*end*/,
                          PointsOnManifold<3, 2, SD, T>* /*boundaryManifold*/,
                          std::size_t /*numSweeps*/) {
   // CONTINUE
   std::cerr << "Warning: Not yet implemented.\n";
}


//-------------------------------------------------------------------------
// Mixed
//-------------------------------------------------------------------------


//! Optimize the position of all vertices.
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T, std::size_t SD >
inline
void
geometricOptimize(IndSimpSetIncAdj<N, N, T>* mesh,
                  PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
                  std::size_t numSweeps) {
   // Get the interior and boundary vertices.
   std::vector<std::size_t> interior, boundary;
   determineInteriorVertices(*mesh, std::back_inserter(interior));
   determineComplementSetOfIndices(mesh->vertices.size(),
                                   interior.begin(), interior.end(),
                                   std::back_inserter(boundary));

   while (numSweeps-- > 0) {
      // If a boundary manifold has been specified.
      if (boundaryManifold != 0) {
         // Optimize the boundary vertices.
         geometricOptimizeBoundary<QF>(mesh, boundary.begin(), boundary.end(),
                                       boundaryManifold, 1);
      }
      // Optimize the interior vertices.
      geometricOptimizeInterior<QF>(mesh, interior.begin(), interior.end(), 1);
   }
}




//-------------------------------------------------------------------------
// Other methods.
//-------------------------------------------------------------------------




// Optimize the position of all vertices.
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T, std::size_t SD >
inline
void
geometricOptimizeWithBoundaryCondition
(IndSimpSetIncAdj<N, N, T>* mesh,
 PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
 const std::size_t numSweeps) {
   // Perform the sweeps over all vertices.
   geometricOptimizeWithBoundaryCondition<QF>
   (mesh, ads::IntIterator<>(0), ads::IntIterator<>(mesh->vertices.size()),
    boundaryManifold, numSweeps);
}






// Optimize the position of all vertices.
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T,
         class BoundaryCondition >
inline
void
geometricOptimizeWithCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                               const BoundaryCondition& condition,
                               const std::size_t numSweeps) {
   // Perform the sweeps over all vertices.
   geometricOptimizeWithCondition<QF>
   (mesh, ads::IntIterator<>(0), ads::IntIterator<>(mesh->vertices.size()),
    condition, numSweeps);
}






// Make \c numSweeps optimization sweeps over the given vertices with
// the quality function given as a template parameter.
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T,
         typename IntForIter, std::size_t SD >
inline
void
geometricOptimizeWithBoundaryCondition
(IndSimpSetIncAdj<N, N, T>* mesh,
 IntForIter begin, IntForIter end,
 PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
 std::size_t numSweeps) {
   typedef IndSimpSetIncAdj<N, N, T> ISS;
   typedef typename ISS::Number Number;
   typedef typename ISS::SimplexFace SimplexFace;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::IncidenceConstIterator IncidenceConstIterator;
   typedef ComplexWithFreeVertex<QF, N, Number> Complex;
   typedef ComplexNorm2Mod<Complex> ComplexQuality;

   // A local mesh.
   Complex sc;
   // The neighboring faces of a vertex to be moved.
   std::vector<SimplexFace> neighboringFaces;
   // Quality function of the local mesh.
   ComplexQuality function(sc);
   // The optimization class.
   numerical::QuasiNewton<N, ComplexQuality> method(function);
   // CONTINUE Set the x tolerance.
   SimplexFace face;
   Vertex oldPosition;
   BBox<Number, N> bb;
   Number radius = 0;

   // Make numSweeps sweeps.
   std::size_t n = 0;
   while (numSweeps-- > 0) {
      // For each vertex.
      for (IntForIter vi = begin; vi != end; ++vi) {
         // The index of the vertex.
         n = *vi;
         // If the vertex has incident simplices.
         if (! mesh->incident.empty(n)) {
            //
            // Build the local mesh.
            //
            neighboringFaces.clear();
            // For each adjacent simplex index to the n_th vertex.
            for (IncidenceConstIterator indexIter = mesh->incident.begin(n);
                  indexIter != mesh->incident.end(n); ++indexIter) {
               getFace(*mesh, *indexIter, n, &face);
               neighboringFaces.push_back(face);
            }
            sc.set(neighboringFaces.begin(), neighboringFaces.end());

            // Calculate an approximate radius for the complex.
            sc.computeBBox(mesh->vertices[n], &bb);
            radius = 0.0;
            for (std::size_t j = 0; j != N; ++j) {
               radius += bb.upper[j] - bb.lower[j];
            }
            radius /= (2 * N);

            //
            // Optimize the position of the free vertex.
            //
            try {
               oldPosition = mesh->vertices[n];
               const Number oldValue = function(oldPosition);
               Number value;
               std::size_t numIterations;
               method.find_minimum(mesh->vertices[n], value, numIterations,
                                   0.1 * radius);
               // If the vertex is on the boundary.
               if (mesh->isVertexOnBoundary(n)) {
                  // Return the point to the boundary manifold.
                  mesh->vertices[n] =
                     boundaryManifold->computeClosestPoint(n, mesh->vertices[n]);
                  value = function(mesh->vertices[n]);
                  // If the new position is not better than the old one.
                  if (value >= oldValue) {
                     // Undo the change.
                     mesh->vertices[n] = oldPosition;
                  }
                  else {
                     // Keep the change.  Update the point on the manifold.
                     boundaryManifold->updatePoint();
                  }
               }
               // Else, the vertex is in the interior.
               else {
                  // If the new position is not better than the old one.
                  if (value >= oldValue) {
                     // Undo the change.
                     mesh->vertices[n] = oldPosition;
                  }
               }
            }
            catch (...) {
               std::cerr << "Error in optimizing vertex " << n
                         << " in sweep().\n"
                         << "radius = " << radius << '\n';
            }
         }
      }
   }
}









// Make \c numSweeps optimization sweeps over the given vertices with the quality function given as a template parameter.
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T,
         typename IntForIter, class BoundaryCondition >
inline
void
geometricOptimizeWithCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                               IntForIter begin, IntForIter end,
                               const BoundaryCondition& condition,
                               std::size_t numSweeps) {
   typedef IndSimpSetIncAdj<N, N, T> ISS;
   typedef typename ISS::Number Number;
   typedef typename ISS::SimplexFace SimplexFace;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::IncidenceConstIterator IncidenceConstIterator;
   typedef ComplexWithFreeVertex<QF, N, Number> Complex;
   typedef ComplexNorm2Mod<Complex> ComplexQuality;

   // A local mesh.
   Complex sc;
   // The neighboring faces of a vertex to be moved.
   std::vector<SimplexFace> neighboringFaces;
   // Quality function of the local mesh.
   ComplexQuality function(sc);
   // The optimization class.
   numerical::QuasiNewton<N, ComplexQuality> method(function);
   // CONTINUE Set the x tolerance.
   SimplexFace face;
   Vertex oldPosition;
   BBox<Number, N> bb;
   Number radius = 0;

   // Make numSweeps sweeps.
   std::size_t n = 0;
   while (numSweeps-- > 0) {
      // For each vertex.
      for (IntForIter vi = begin; vi != end; ++vi) {
         // The index of the vertex.
         n = *vi;
         // If the vertex has incident simplices.
         if (! mesh->incident.empty(n)) {
            //
            // Build the local mesh.
            //
            neighboringFaces.clear();
            // For each adjacent simplex index to the n_th vertex.
            for (IncidenceConstIterator
                  indexIter = mesh->incident.begin(n);
                  indexIter != mesh->incident.end(n); ++indexIter) {
               getFace(*mesh, *indexIter, n, &face);
               neighboringFaces.push_back(face);
            }
            sc.set(neighboringFaces.begin(), neighboringFaces.end());

            // Calculate an approximate radius for the complex.
            sc.computeBBox(mesh->vertices[n], &bb);
            radius = 0.0;
            for (std::size_t j = 0; j != N; ++j) {
               radius += bb.upper[j] - bb.lower[j];
            }
            radius /= (2 * N);

            //
            // Optimize the position of the free vertex.
            //
            try {
               oldPosition = mesh->vertices[n];
               const Number oldValue = function(oldPosition);
               // Initialize to avoid a warning.
               Number value = 0;
               std::size_t numIterations;
               method.find_minimum(mesh->vertices[n], value, numIterations,
                                   0.1 * radius);
               // If the vertex is on the boundary.
               if (mesh->isVertexOnBoundary(n)) {
                  // Apply the condition to the vertex.
                  applyBoundaryCondition(mesh, condition, n);
                  // Update the value.
                  value = function(mesh->vertices[n]);
               }
               // If the new position is not better than the old one.
               if (value >= oldValue) {
                  // Undo the change.
                  mesh->vertices[n] = oldPosition;
               }
            }
            catch (...) {
               std::cerr << "Error in optimizing vertex " << n
                         << " in sweep().\n"
                         << "radius = " << radius << '\n';
            }
         }
      }
   }
}



// Make \c numSweeps constrained optimization sweeps over the given vertices with the quality function given as a template parameter.
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T,
         typename IntForIter, class BoundaryCondition >
inline
void
geometricOptimizeWithConditionConstrained
(IndSimpSetIncAdj<N, N, T>* mesh,
 IntForIter begin, IntForIter end,
 const BoundaryCondition& condition,
 const T maxConstraintError,
 const std::size_t numSweeps) {
   typedef IndSimpSetIncAdj<N, N, T> ISS;
   typedef typename ISS::Number Number;
   typedef typename ISS::SimplexFace SimplexFace;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::IncidenceConstIterator IncidenceConstIterator;
   typedef ComplexWithFreeVertex<QF, N, Number> Complex;
   typedef ComplexNorm2Mod<Complex> ComplexQuality;
   typedef ComplexContentConstraint<Complex> Constraint;

   // A local mesh.
   Complex sc;
   // The neighboring faces of a vertex to be moved.
   std::vector<SimplexFace> neighboringFaces;
   // Quality function of the local mesh.
   ComplexQuality function(sc);
   // Content constraint.
   Constraint constraint(sc);
   // The constrained optimization class.
   numerical::PenaltyQuasiNewton<N, ComplexQuality, Constraint>
   method(function, constraint, maxConstraintError);
   SimplexFace face;
   Vertex oldPosition;
   Vertex fg, cg;
   Number mfg, mcg;
   BBox<Number, N> bb;
   Number radius = 0;

   // Make numSweeps sweeps.
   std::size_t n = 0;
   for (std::size_t i = 0; i != numSweeps; ++i) {
      // For each vertex.
      for (IntForIter vi = begin; vi != end; ++vi) {
         // The index of the vertex.
         n = *vi;
         //
         // Build the local mesh.
         //
         neighboringFaces.clear();
         // For each adjacent simplex index.
         for (IncidenceConstIterator
               indexIter = mesh->incident.begin(n);
               indexIter != mesh->incident.end(n); ++indexIter) {
            getFace(*mesh, *indexIter, n, &face);
            neighboringFaces.push_back(face);
         }
         sc.set(neighboringFaces.begin(), neighboringFaces.end());

         // Calculate an approximate radius for the complex.
         sc.computeBBox(mesh->vertices[n], &bb);
         radius = 0.0;
         for (std::size_t j = 0; j != N; ++j) {
            radius += bb.upper[j] - bb.lower[j];
         }
         radius /= (2 * N);

         // Initialize the constraint.
         constraint.initialize(mesh->vertices[n]);

         //
         // Set the initial value of the penalty parameter.
         //
         constraint.gradient(mesh->vertices[n], cg);
         function.gradient(mesh->vertices[n], fg);
         mfg = ext::magnitude(fg);
         mcg = ext::magnitude(cg);
         if (mfg > std::numeric_limits<Number>::epsilon()) {
            method.set_initial_penalty_parameter(0.5 * maxConstraintError *
                                                 mcg / mfg);
         }
         else {
            method.set_initial_penalty_parameter(maxConstraintError);
         }

         //
         // Optimize the position of the free vertex.
         //
         try {
            oldPosition = mesh->vertices[n];
            const Number oldValue = function(oldPosition);
            Number value;
            std::size_t numIterations;
            method.find_minimum(mesh->vertices[n], value, numIterations,
                                0.1 * radius);
            // Apply the condition to the vertex.
            applyBoundaryCondition(mesh, condition, n);
            // CONTINUE REMOVE
            //mesh->vertices[n] = condition(mesh->vertices[n]);
            // If the new position is not better than the old one.
            if (value > oldValue) {
               // Undo the change.
               mesh->vertices[n] = oldPosition;
            }
         }
         catch (...) {
            std::cerr << "Error in optimizing vertex " << n
                      << " in geometricOptimizeWithConditionConstrained().\n"
                      << "radius = " << radius << '\n';
         }
      }
   }
}

} // namespace geom
}
