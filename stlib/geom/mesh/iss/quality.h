// -*- C++ -*-

/*!
  \file geom/mesh/iss/quality.h
  \brief Implements quality measures for IndSimpSet.
*/

#if !defined(__geom_mesh_iss_quality_h__)
#define __geom_mesh_iss_quality_h__

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"
#include "stlib/geom/mesh/iss/topology.h"

#include "stlib/geom/mesh/simplex/SimplexModMeanRatio.h"
#include "stlib/geom/mesh/simplex/SimplexModCondNum.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_quality Quality
  These function measure quality statistics for simplicial meshes.
*/
//@{

//! Calculate the adjacency counts for the simplices in the mesh.
/*!
  \relates IndSimpSet
  Each simplex has between 0 and M+1 (inclusive) adjacent simplices.
*/
template<std::size_t N, std::size_t M, typename T>
void
countAdjacencies(const IndSimpSetIncAdj<N, M, T>& iss,
                 std::array < std::size_t, M + 2 > * counts);

// CONTINUE: edge length statistics.
// Add the edge length to the functions that print quality.


//! Identify the elements that have a low quality using the condition number metric.
/*! \relates IndSimpSet
  \pre 0 < minimumAllowedQuality <= 1.
*/
template<std::size_t N, std::size_t M, typename T, typename OutputIterator>
void
identifyLowQualityWithCondNum(const IndSimpSet<N, M, T>& mesh,
                              T minimumAllowedQuality,
                              OutputIterator indices);

//! Compute edge length statistics.
/*!
  With a bucket of simplices, one can compute the minimum and maximum, but
  not the mean.
*/
template<std::size_t M, typename SimpInIter, typename T>
void
computeEdgeLengthStatistics(SimpInIter simplicesBeginning,
                            SimpInIter simplicesEnd,
                            T* minimumLength,
                            T* maximumLength);



//! Compute edge length statistics.
template<std::size_t M, typename VertRAIter, typename ISInIter, typename T>
void
computeEdgeLengthStatistics(VertRAIter vertices,
                            ISInIter indexedSimplicesBeginning,
                            ISInIter indexedSimplicesEnd,
                            T* minimumLength,
                            T* maximumLength);



//! Compute edge length statistics.
template<std::size_t N, typename T>
void
computeEdgeLengthStatistics(const IndSimpSetIncAdj<N, 2, T>& mesh,
                            T* minimumLength,
                            T* maximumLength,
                            T* meanLength);


//! Return the minimum edge length.
template<std::size_t M, typename T, typename SimpInIter>
T
computeMinimumEdgeLength(SimpInIter simplicesBeginning,
                         SimpInIter simplicesEnd);



//! Return the minimum edge length.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
inline
T
computeMinimumEdgeLength(const IndSimpSet<N, M, T>& mesh) {
   return computeMinimumEdgeLength<M, T>(mesh.getSimplicesBegin(),
                                         mesh.getSimplicesEnd());
}


//! Return the maximum edge length.
template<std::size_t M, typename T, typename SimpInIter>
T
computeMaximumEdgeLength(SimpInIter simplicesBeginning,
                         SimpInIter simplicesEnd);


//! Return the maximum edge length.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
inline
T
computeMaximumEdgeLength(const IndSimpSet<N, M, T>& mesh) {
   return computeMaximumEdgeLength<M, T>(mesh.getSimplicesBegin(),
                                         mesh.getSimplicesEnd());
}

//! Compute edge length statistics.
/*! \relates IndSimpSetIncAdj */
template<std::size_t N, std::size_t M, typename T>
inline
T
computeMeanEdgeLength(const IndSimpSetIncAdj<N, M, T>& mesh) {
   T minimumLength, maximumLength, meanLength;
   computeEdgeLengthStatistics(mesh, &minimumLength, &maximumLength,
                               &meanLength);
   return meanLength;
}


//! Return the total content of the simplices in the mesh.
template<std::size_t M, typename T, typename VertRAIter, typename ISInIter>
T
computeContent(VertRAIter vertices,
               ISInIter indexedSimplicesBeginning, ISInIter indexedSimplicesEnd);

//! Return the total content of the simplices in the mesh.
template<std::size_t M, typename T, typename SimpInIter>
T
computeContent(SimpInIter simplicesBeginning, SimpInIter simplicesEnd);

//! Return the total content of the simplices in the mesh.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
T
computeContent(const IndSimpSet<N, M, T>& iss);


//! Calculate content (hypervolume) statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename VertRAIter, typename ISInIter>
void
computeContentStatistics(VertRAIter vertices,
                         ISInIter indexedSimplicesBeginning,
                         ISInIter indexedSimplicesEnd,
                         T* minimumContent,
                         T* maximumContent,
                         T* meanContent);


//! Calculate content (hypervolume) statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename SimpInIter>
void
computeContentStatistics(SimpInIter simplicesBeginning,
                         SimpInIter simplicesEnd,
                         T* minimumContent,
                         T* maximumContent,
                         T* meanContent);


//! Calculate content (hypervolume) statistics for the simplices in the mesh.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
void
computeContentStatistics(const IndSimpSet<N, M, T>& iss,
                         T* minimumContent,
                         T* maximumContent,
                         T* meanContent);



//! Calculate determinant statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename VertRAIter, typename ISInIter>
void
computeDeterminantStatistics(VertRAIter vertices,
                             ISInIter indexedSimplicesBeginning,
                             ISInIter indexedSimplicesEnd,
                             T* minimumDeterminant,
                             T* maximumDeterminant,
                             T* meanDeterminant);

//! Calculate determinant statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename SimpInIter>
void
computeDeterminantStatistics(SimpInIter simplicesBeginning,
                             SimpInIter simplicesEnd,
                             T* minimumDeterminant,
                             T* maximumDeterminant,
                             T* meanDeterminant);

//! Calculate determinant statistics for the simplices in the mesh.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
void
computeDeterminantStatistics(const IndSimpSet<N, M, T>& iss,
                             T* minimumDeterminant,
                             T* maximumDeterminant,
                             T* meanDeterminant);




//! Calculate modified mean ratio function statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename VertRAIter, typename ISInIter>
void
computeModifiedMeanRatioStatistics(VertRAIter vertices,
                                   ISInIter indexedSimplicesBeginning,
                                   ISInIter indexedSimplicesEnd,
                                   T* minimumModMeanRatio,
                                   T* maximumModMeanRatio,
                                   T* meanModMeanRatio);

//! Calculate modified mean ratio function statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename SimpInIter>
void
computeModifiedMeanRatioStatistics(SimpInIter simplicesBeginning,
                                   SimpInIter simplicesEnd,
                                   T* minimumModMeanRatio,
                                   T* maximumModMeanRatio,
                                   T* meanModMeanRatio);

//! Calculate modified mean ratio function statistics for the simplices in the mesh.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
void
computeModifiedMeanRatioStatistics(const IndSimpSet<N, M, T>& iss,
                                   T* minimumModMeanRatio,
                                   T* maximumModMeanRatio,
                                   T* meanModMeanRatio);


//! Calculate modified condition number function statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename VertRAIter, typename ISInIter>
void
computeModifiedConditionNumberStatistics(VertRAIter vertices,
      ISInIter indexedSimplicesBeginning,
      ISInIter indexedSimplicesEnd,
      T* minimumModCondNum,
      T* maximumModCondNum,
      T* meanModCondNum);

//! Calculate modified condition number function statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename SimpInIter>
void
computeModifiedConditionNumberStatistics(SimpInIter simplicesBeginning,
      SimpInIter simplicesEnd,
      T* minimumModCondNum,
      T* maximumModCondNum,
      T* meanModCondNum);

//! Calculate modified condition number function statistics for the simplices in the mesh.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
void
computeModifiedConditionNumberStatistics(const IndSimpSet<N, M, T>& iss,
      T* minimumModCondNum,
      T* maximumModCondNum,
      T* meanModCondNum);


//! Calculate quality statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename VertRAIter, typename ISInIter>
void
computeQualityStatistics(VertRAIter vertices,
                         ISInIter indexedSimplicesBeginning,
                         ISInIter indexedSimplicesEnd,
                         T* minimumContent,
                         T* maximumContent,
                         T* meanContent,
                         T* minimumDeterminant,
                         T* maximumDeterminant,
                         T* meanDeterminant,
                         T* minimumModMeanRatio,
                         T* maximumModMeanRatio,
                         T* meanModMeanRatio,
                         T* minimumModCondNum,
                         T* maximumModCondNum,
                         T* meanModCondNum);

//! Calculate quality statistics for the simplices in the mesh.
template<std::size_t M, typename T, typename SimpInIter>
void
computeQualityStatistics(SimpInIter simplicesBeginning,
                         SimpInIter simplicesEnd,
                         T* minimumContent,
                         T* maximumContent,
                         T* meanContent,
                         T* minimumDeterminant,
                         T* maximumDeterminant,
                         T* meanDeterminant,
                         T* minimumModMeanRatio,
                         T* maximumModMeanRatio,
                         T* meanModMeanRatio,
                         T* minimumModCondNum,
                         T* maximumModCondNum,
                         T* meanModCondNum);

//! Calculate quality statistics for the simplices in the mesh.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
void
computeQualityStatistics(const IndSimpSet<N, M, T>& iss,
                         T* minimumContent,
                         T* maximumContent,
                         T* meanContent,
                         T* minimumDeterminant,
                         T* maximumDeterminant,
                         T* meanDeterminant,
                         T* minimumModMeanRatio,
                         T* maximumModMeanRatio,
                         T* meanModMeanRatio,
                         T* minimumModCondNum,
                         T* maximumModCondNum,
                         T* meanModCondNum);


//! Print quality statistics for the simplices in the mesh.
template<std::size_t N, std::size_t M, typename T, typename VertRAIter, typename ISInIter>
void
printQualityStatistics(std::ostream& out,
                       VertRAIter verticesBeginning, VertRAIter verticesEnd,
                       ISInIter indexedSimplicesBeginning,
                       ISInIter indexedSimplicesEnd);

//! Print quality statistics for the simplices in the mesh.
template<std::size_t N, std::size_t M, typename T, typename SimpInIter>
void
printQualityStatistics(std::ostream& out,
                       SimpInIter simplicesBeginning, SimpInIter simplicesEnd);

//! Print quality statistics for the simplices in the mesh.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
void
printQualityStatistics(std::ostream& out,
                       const IndSimpSet<N, M, T>& mesh);

//! Print information about the mesh.
/*! \relates IndSimpSet */
template<std::size_t N, std::size_t M, typename T>
void
printInformation(std::ostream& out,
                 const IndSimpSet<N, M, T>& mesh);

//! Print information about the mesh.
/*! \relates IndSimpSetIncAdj */
template<std::size_t N, std::size_t M, typename T>
void
printInformation(std::ostream& out,
                 const IndSimpSetIncAdj<N, M, T>& mesh);

//@}

} // namespace geom
}

#define __geom_mesh_iss_quality_ipp__
#include "stlib/geom/mesh/iss/quality.ipp"
#undef __geom_mesh_iss_quality_ipp__

#endif
