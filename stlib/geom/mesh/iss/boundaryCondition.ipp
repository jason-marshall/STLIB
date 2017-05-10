// -*- C++ -*-

#if !defined(__geom_mesh_iss_boundaryCondition_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

// Apply the closest point boundary condition at a vertex.
template < typename T,
         class ISS >
inline
void
applyBoundaryCondition(IndSimpSetIncAdj<2, 1, T>* mesh,
                       const ISS_SD_ClosestPoint<ISS>& condition,
                       const std::size_t n) {
   mesh->vertices[n] = condition(mesh->vertices[n]);
}

// Apply the closest point in the normal direction boundary condition
// at a vertex.
template<typename T, class ISS>
inline
void
applyBoundaryCondition(IndSimpSetIncAdj<2, 1, T>* mesh,
                       const ISS_SD_ClosestPointDirection<ISS>& condition,
                       const std::size_t n) {
   typedef IndSimpSetIncAdj<2, 1, T> Mesh;
   typedef typename Mesh::Vertex Vertex;

   // Get the normal direction.
   Vertex vertexNormal;
   computeVertexNormal(*mesh, n, &vertexNormal);

   // Apply the boundary condition.
   // Closest point in the normal direction.
   mesh->vertices[n] = condition(mesh->vertices[n], vertexNormal);
}

// Apply the closest point boundary condition at a vertex.
template<typename T, class ISS>
inline
void
applyBoundaryCondition(IndSimpSetIncAdj<3, 2, T>* mesh,
                       const ISS_SD_ClosestPoint<ISS>& condition,
                       const std::size_t n) {
   mesh->vertices[n] = condition(mesh->vertices[n]);
}

// Apply the closest point in the normal direction boundary condition
// at a vertex.
template<typename T, class ISS>
inline
void
applyBoundaryCondition(IndSimpSetIncAdj<3, 2, T>* mesh,
                       const ISS_SD_ClosestPointDirection<ISS>& condition,
                       const std::size_t n) {
   typedef IndSimpSetIncAdj<3, 2, T> Mesh;
   typedef typename Mesh::Vertex Vertex;

   // Get the normal direction.
   Vertex vertexNormal;
   computeVertexNormal(*mesh, n, &vertexNormal);

   // Apply the boundary condition.
   // Closest point in the normal direction.
   mesh->vertices[n] = condition(mesh->vertices[n], vertexNormal);
}

// Apply the closest point boundary condition at a vertex.
template < std::size_t N, typename T,
         class UnaryFunction >
inline
void
applyBoundaryCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                       const UnaryFunction& condition,
                       const std::size_t n) {
   // The vertex may or may not be on the boundary.
   mesh->vertices[n] = condition(mesh->vertices[n]);
}

// Apply the closest point boundary condition at a vertex.
template<std::size_t N, typename T, class ISS>
inline
void
applyBoundaryCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                       const ISS_SD_ClosestPoint<ISS>& condition,
                       const std::size_t n) {
#ifdef STLIB_DEBUG
   assert(mesh->isVertexOnBoundary(n));
#endif
   mesh->vertices[n] = condition(mesh->vertices[n]);
}

// Apply the closer point boundary condition at a vertex.
template<std::size_t N, typename T, class ISS>
inline
void
applyBoundaryCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                       const ISS_SD_CloserPoint<ISS>& condition,
                       const std::size_t n) {
#ifdef STLIB_DEBUG
   assert(mesh->isVertexOnBoundary(n));
#endif
   mesh->vertices[n] = condition(mesh->vertices[n]);
}

// Apply the closest point in the normal direction boundary condition
// at a vertex.
template<std::size_t N, typename T, class ISS>
inline
void
applyBoundaryCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                       const ISS_SD_ClosestPointDirection<ISS>& condition,
                       const std::size_t n) {
   typedef IndSimpSetIncAdj<N, N, T> Mesh;
   typedef typename Mesh::Vertex Vertex;

   // Get the normal direction.
   Vertex vertexNormal;
   computeVertexNormal(*mesh, n, &vertexNormal);

   // Apply the boundary condition.
   // Closest point in the normal direction.
   mesh->vertices[n] = condition(mesh->vertices[n], vertexNormal);
}

// Apply the closer point in the normal direction boundary condition
// at a vertex.
template<std::size_t N, typename T, class ISS>
inline
void
applyBoundaryCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                       const ISS_SD_CloserPointDirection<ISS>& condition,
                       const std::size_t n) {
   typedef IndSimpSetIncAdj<N, N, T> Mesh;
   typedef typename Mesh::Vertex Vertex;

   // Get the normal direction.
   Vertex vertexNormal;
   computeVertexNormal(*mesh, n, &vertexNormal);

   // Apply the boundary condition.
   // Closest point in the normal direction.
   mesh->vertices[n] = condition(mesh->vertices[n], vertexNormal);
}

} // namespace geom
}
