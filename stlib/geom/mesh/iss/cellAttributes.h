// -*- C++ -*-

/*!
  \file geom/mesh/iss/cellAttributes.h
  \brief Implements cell attribute measures for IndSimpSet.
*/

#if !defined(__geom_mesh_iss_cellAttributes_h__)
#define __geom_mesh_iss_cellAttributes_h__

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"
#include "stlib/geom/mesh/iss/topology.h"

#include "stlib/geom/mesh/simplex.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_cellAttributes Cell Attributes
  These functions compute cell attributes for simplicial meshes.
*/
//@{


//----------------------------------------------------------------------------
// Mean ratio.
//----------------------------------------------------------------------------


//! Calculate the mean ratio function for each simplex in the mesh.
template < std::size_t M, typename T, typename VertRAIter, typename ISInIter,
         typename OutputIterator >
void
computeMeanRatio(VertRAIter vertices,
                 ISInIter indexedSimplicesBeginning,
                 ISInIter indexedSimplicesEnd,
                 OutputIterator output);

//! Calculate the mean ratio function for each simplex in the mesh.
template<std::size_t M, typename T, typename SimpInIter, typename OutputIterator>
void
computeMeanRatio(SimpInIter simplicesBeginning,
                 SimpInIter simplicesEnd,
                 OutputIterator output);

//! Calculate the mean ratio function for each simplex in the mesh.
/*! \relates IndSimpSet */
template < std::size_t N, std::size_t M, typename T,
         typename OutputIterator >
void
computeMeanRatio(const IndSimpSet<N, M, T>& iss,
                 OutputIterator output);



//----------------------------------------------------------------------------
// Modified mean ratio.
//----------------------------------------------------------------------------


//! Calculate the modified mean ratio function for each simplex in the mesh.
template < std::size_t M, typename T, typename VertRAIter, typename ISInIter,
         typename OutputIterator >
void
computeModifiedMeanRatio(VertRAIter vertices,
                         ISInIter indexedSimplicesBeginning,
                         ISInIter indexedSimplicesEnd,
                         OutputIterator output);

//! Calculate the modified mean ratio function for each simplex in the mesh.
template<std::size_t M, typename T, typename SimpInIter, typename OutputIterator>
void
computeModifiedMeanRatio(SimpInIter simplicesBeginning,
                         SimpInIter simplicesEnd,
                         OutputIterator output);

//! Calculate the modified mean ratio function for each simplex in the mesh.
/*! \relates IndSimpSet */
template < std::size_t N, std::size_t M, typename T,
         typename OutputIterator >
void
computeModifiedMeanRatio(const IndSimpSet<N, M, T>& iss,
                         OutputIterator output);


//----------------------------------------------------------------------------
// Condition number.
//----------------------------------------------------------------------------


//! Calculate the condition number function for each simplex in the mesh.
template < std::size_t M, typename T, typename VertRAIter, typename ISInIter,
         typename OutputIterator >
void
computeConditionNumber(VertRAIter vertices,
                       ISInIter indexedSimplicesBeginning,
                       ISInIter indexedSimplicesEnd,
                       OutputIterator output);

//! Calculate the condition number function for each simplex in the mesh.
template<std::size_t M, typename T, typename SimpInIter, typename OutputIterator>
void
computeConditionNumber(SimpInIter simplicesBeginning,
                       SimpInIter simplicesEnd,
                       OutputIterator output);

//! Calculate the condition number function for each simplex in the mesh.
/*! \relates IndSimpSet */
template < std::size_t N, std::size_t M, typename T,
         typename OutputIterator >
void
computeConditionNumber(const IndSimpSet<N, M, T>& iss,
                       OutputIterator output);



//----------------------------------------------------------------------------
// Modified condition number.
//----------------------------------------------------------------------------


//! Calculate the modified condition number function for each simplex in the mesh.
template < std::size_t M, typename T, typename VertRAIter, typename ISInIter,
         typename OutputIterator >
void
computeModifiedConditionNumber(VertRAIter vertices,
                               ISInIter indexedSimplicesBeginning,
                               ISInIter indexedSimplicesEnd,
                               OutputIterator output);

//! Calculate the modified condition number function for each simplex in the mesh.
template<std::size_t M, typename T, typename SimpInIter, typename OutputIterator>
void
computeModifiedConditionNumber(SimpInIter simplicesBeginning,
                               SimpInIter simplicesEnd,
                               OutputIterator output);

//! Calculate the modified condition number function for each simplex in the mesh.
/*! \relates IndSimpSet */
template < std::size_t N, std::size_t M, typename T,
         typename OutputIterator >
void
computeModifiedConditionNumber(const IndSimpSet<N, M, T>& iss,
                               OutputIterator output);


//----------------------------------------------------------------------------
// Content.
//----------------------------------------------------------------------------


//! Calculate the content for each simplex in the mesh.
template < std::size_t M, typename T, typename VertRAIter, typename ISInIter,
         typename OutputIterator >
void
computeContent(VertRAIter vertices,
               ISInIter indexedSimplicesBeginning,
               ISInIter indexedSimplicesEnd,
               OutputIterator output);

//! Calculate the content for each simplex in the mesh.
template<std::size_t M, typename T, typename SimpInIter, typename OutputIterator>
void
computeContent(SimpInIter simplicesBeginning,
               SimpInIter simplicesEnd,
               OutputIterator output);

//! Calculate the content for each simplex in the mesh.
/*! \relates IndSimpSet */
template < std::size_t N, std::size_t M, typename T,
         typename OutputIterator >
void
computeContent(const IndSimpSet<N, M, T>& iss,
               OutputIterator output);



//@}

} // namespace geom
}

#define __geom_mesh_iss_cellAttributes_ipp__
#include "stlib/geom/mesh/iss/cellAttributes.ipp"
#undef __geom_mesh_iss_cellAttributes_ipp__

#endif
