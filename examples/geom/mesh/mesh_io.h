// -*- C++ -*-

#if !defined(__examples_geom_mesh_mesh_io_h__)
#define __examples_geom_mesh_mesh_io_h__

// Include the ISS and/or the SMR I/O files before including this file.

#include <iostream>
#include <fstream>

using namespace stlib;

template<class Mesh>
inline
void
writeMeshAscii(const char* name, const Mesh& mesh) {
   std::ofstream file(name);
   if (! file) {
      std::cerr << "Bad output file.  Exiting...\n";
      exit(1);
   }
   geom::writeAscii(file, mesh);
}


template<class Mesh>
inline
void
writeMeshBinary(const char* name, const Mesh& mesh) {
   std::ofstream file(name);
   if (! file) {
      std::cerr << "Bad output file.  Exiting...\n";
      exit(1);
   }
   geom::writeBinary(file, mesh);
}


template<class Mesh>
inline
void
writeMeshVtkXml(const char* name, const Mesh& mesh) {
   std::ofstream file(name);
   if (! file) {
      std::cerr << "Bad output file.  Exiting...\n";
      exit(1);
   }
   geom::writeVtkXml(file, mesh);
}


template<class Mesh, typename F>
inline
void
writeMeshVtkXml(const char* name, const Mesh& mesh,
                const std::vector<F>& cellData,
                std::string dataName) {
   std::ofstream file(name);
   if (! file) {
      std::cerr << "Bad output file.  Exiting...\n";
      exit(1);
   }
   geom::writeVtkXml(file, mesh, cellData, dataName);
}


template<class Mesh, typename ContainerIter, typename StringIter>
inline
void
writeMeshVtkXml(const char* name, const Mesh& mesh,
                ContainerIter cellDataContainersBeginning,
                ContainerIter cellDataContainersEnd,
                StringIter dataNamesBeginning,
                StringIter dataNamesEnd) {
   std::ofstream file(name);
   if (! file) {
      std::cerr << "Bad output file.  Exiting...\n";
      exit(1);
   }
   geom::writeVtkXml(file, mesh, cellDataContainersBeginning,
                     cellDataContainersEnd,
                     dataNamesBeginning, dataNamesEnd);
}


//! Write in legacy VTK unstructured grid format.
template<class Mesh>
inline
void
writeMeshVtkLegacy(const char* name, const Mesh& mesh,
                   std::string title = "") {
   std::ofstream file(name);
   if (! file) {
      std::cerr << "Bad output file.  Exiting...\n";
      exit(1);
   }
   geom::writeVtkLegacy(file, mesh, title);
}


//! Read an indexed simplex set in ascii format.
template<class Mesh>
inline
void
readMeshAscii(const char* name, Mesh* mesh) {
   std::ifstream file(name);
   if (! file) {
      std::cerr << "Bad input file.  Exiting...\n";
      exit(1);
   }
   geom::readAscii(file, mesh);
}


//! Read an indexed simplex set in binary format.
template<class Mesh>
inline
void
readMeshBinary(const char* name, Mesh* mesh) {
   std::ifstream file(name);
   if (! file) {
      std::cerr << "Bad input file.  Exiting...\n";
      exit(1);
   }
   geom::readBinary(file, mesh);
}

#endif
