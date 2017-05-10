// -*- C++ -*-

#if !defined(__geom_mesh_iss_transfer_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

template < std::size_t N, std::size_t M, typename T,
         class PointArray, class IndexArray >
inline
void
transferIndices(const IndSimpSet<N, M, T>& mesh,
                const PointArray& points, IndexArray* indices) {
   typedef IndSimpSet<N, M, T> ISS;

   //
   // Sanity checks.
   //

   // The source mesh must not be trivial.
   assert(mesh.vertices.size() > 0);
   assert(mesh.indexedSimplices.size() > 0);
   // Points and indices should be the same size.
   assert(points.size() == indices->size());

   //
   // Build the data structure for doing simplex queries.
   //

   ISS_SimplexQuery<ISS> meshQuery(mesh);

   // For each target vertex.
   for (std::size_t i = 0; i != points.size(); ++i) {
      // Find the relevant simplex in the source mesh.
      (*indices)[i] = meshQuery.computeMinimumDistanceIndex(points[i]);
   }
}



template < std::size_t N, std::size_t M, typename T,
         class SourceFieldArray, class PointArray, class TargetFieldArray >
inline
void
transfer(const IndSimpSet<N, M, T>& mesh,
         const SourceFieldArray& sourceFields,
         const PointArray& points,
         TargetFieldArray* targetFields) {
   typedef IndSimpSet<N, M, T> ISS;
   typedef typename SourceFieldArray::value_type Field;

   //
   // Sanity checks.
   //

   // The source mesh must not be trivial.
   assert(mesh.vertices.size() > 0);
   assert(mesh.indexedSimplices.size() > 0);
   // The fields are specified at the vertices.
   assert(mesh.vertices.size() == sourceFields.size());
   assert(points.size() == targetFields->size());

   // Build the data structure for doing simplex queries and interpolation.
   ISS_Interpolate<ISS, Field> meshInterp(mesh, &sourceFields[0]);

   // For each target vertex.
   for (std::size_t i = 0; i != points.size(); ++i) {
      (*targetFields)[i] = meshInterp(points[i]);
   }
}

} // namespace geom
}
