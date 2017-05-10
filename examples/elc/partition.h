// -*- C++ -*-

/*!
  \file partition.h
  \brief Partition the Lagrangian mesh.
*/

#if !defined(__elc_test_partition_h__)
#define __elc_test_partition_h__

#include "concurrent/partition/rcb.h"
#include "array/ArrayRef.h"


//! Partition a mesh.
/*!
  \param numberOfProcessors The number of processors.
  \param rank The rank of the processors.
  \param vertices The array of global vertices.
  \param elements The array of global elements.
  \param localIdentifiers The vertex identifiers for the rank_th partition.
  \param localVertices The vertices for the rank_th partition.
  \param localElements The elements for the rank_th partition.  These
  are tuples of indices into the local vertex array.
*/
template < std::size_t N, typename T, typename Vertex, typename IndexedElement1,
         typename IndexedElement2 >
void
partitionMesh(std::size_t numberOfProcessors, std::size_t rank,
              const std::vector<Vertex>& vertices,
              const std::vector<IndexedElement1>& elements,
              std::vector<int>* localIdentifiers,
              std::vector<Vertex>* localVertices,
              std::vector<IndexedElement2>* localElements);


#define __elc_test_partition_ipp__
#include "partition.ipp"
#undef __elc_test_partition_ipp__

#endif
