// -*- C++ -*-

#if !defined(__examples_geom_mesh_smr_io_h__)
#define __examples_geom_mesh_smr_io_h__

#include "stlib/geom/mesh/simplicial/file_io.h"

#include "mesh_io.h"

using namespace stlib;

//! Write a mesh as an indexed simplex set in ascii format.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
writeAscii(const char* name,
           const geom::SimpMeshRed<N, M, T, Node, Cell, Cont>& mesh) {
   writeMeshAscii(name, mesh);
}


//! Write a mesh as an indexed simplex set in binary format.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
writeBinary(const char* name,
            const geom::SimpMeshRed<N, M, T, Node, Cell, Cont>& mesh) {
   writeMeshBinary(name, mesh);
}


//! Write in VTK XML unstructured grid format.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
writeVtkXml(const char* name,
            const geom::SimpMeshRed<N, M, T, Node, Cell, Cont>& mesh) {
   writeMeshVtkXml(name, mesh);
}


//! Write in legacy VTK unstructured grid format.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
writeVtkLegacy(const char* name,
               const geom::SimpMeshRed<N, M, T, Node, Cell, Cont>& mesh,
               std::string title = "") {
   writeMeshVtkLegacy(name, mesh, title);
}


//! Read a mesh as an indexed simplex set in ascii format.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
readAscii(const char* name, geom::SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh) {
   readMeshAscii(name, mesh);
}


//! Read an indexed simplex set in binary format.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
readBinary(const char* name, geom::SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh) {
   readMeshBinary(name, mesh);
}

#endif
