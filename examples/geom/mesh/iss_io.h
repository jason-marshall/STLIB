// -*- C++ -*-

#if !defined(__examples_geom_mesh_iss_io_h__)
#define __examples_geom_mesh_iss_io_h__

#include "stlib/geom/mesh/iss/file_io.h"

#include "mesh_io.h"

using namespace stlib;

//! Write an indexed simplex set in ascii format.
template<std::size_t N, std::size_t M, typename T>
inline
void
writeAscii(const char* name, const geom::IndSimpSet<N, M, T>& mesh) {
   writeMeshAscii(name, mesh);
}


//! Write an indexed simplex set in binary format.
template<std::size_t N, std::size_t M, typename T>
inline
void
writeBinary(const char* name, const geom::IndSimpSet<N, M, T>& mesh) {
   writeMeshBinary(name, mesh);
}


//! Write in VTK XML unstructured grid format.
template<std::size_t N, std::size_t M, typename T>
inline
void
writeVtkXml(const char* name, const geom::IndSimpSet<N, M, T>& mesh) {
   writeMeshVtkXml(name, mesh);
}


//! Write the mesh and the cell data in VTK XML unstructured grid format.
template<std::size_t N, std::size_t M, typename T, typename F>
inline
void
writeVtkXml(const char* name, const geom::IndSimpSet<N, M, T>& mesh,
            const std::vector<F>& cellData,
            std::string dataName = "Attribute") {
   writeMeshVtkXml(name, mesh, cellData, dataName);
}


//! Write the mesh and the cell data in VTK XML unstructured grid format.
template < std::size_t N, std::size_t M, typename T,
         typename ContainerIter, typename StringIter >
inline
void
writeVtkXml(const char* name, const geom::IndSimpSet<N, M, T>& mesh,
            ContainerIter cellDataContainersBeginning,
            ContainerIter cellDataContainersEnd,
            StringIter dataNamesBeginning,
            StringIter dataNamesEnd) {
   writeMeshVtkXml(name, mesh, cellDataContainersBeginning,
                   cellDataContainersEnd,
                   dataNamesBeginning, dataNamesEnd);
}


//! Write in legacy VTK unstructured grid format.
template<std::size_t N, std::size_t M, typename T>
inline
void
writeVtkLegacy(const char* name,
               const geom::IndSimpSet<N, M, T>& mesh,
               std::string title = "") {
   writeMeshVtkLegacy(name, mesh, title);
}


//! Read an indexed simplex set in ascii format.
template<std::size_t N, std::size_t M, typename T>
inline
void
readAscii(const char* name, geom::IndSimpSet<N, M, T>* mesh) {
   readMeshAscii(name, mesh);
}


//! Read an indexed simplex set in binary format.
template<std::size_t N, std::size_t M, typename T>
inline
void
readBinary(const char* name, geom::IndSimpSet<N, M, T>* mesh) {
   readMeshBinary(name, mesh);
}

#endif
