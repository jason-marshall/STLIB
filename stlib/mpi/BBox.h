// -*- C++ -*-

/**
  \file mpi/BBox.h
  \brief Operations on geom::BBox.
*/

#if !defined(__mpi_BBox_h__)
#define __mpi_BBox_h__

#include "stlib/geom/kernel/BBox.h"
#include "stlib/mpi/wrapper.h"

namespace stlib
{
namespace mpi
{


//! Merge the distributed bounding boxes.
/*! On the root process, the output is the merged bounding box. */
template<typename _Float, std::size_t _Dimension>
geom::BBox<_Float, _Dimension>
reduce(geom::BBox<_Float, _Dimension> const& input, MPI_Comm comm,
       int root = 0);


//! Merge the distributed bounding boxes.
/*! On all processes, the output is the merged bounding box. */
template<typename _Float, std::size_t _Dimension>
geom::BBox<_Float, _Dimension>
allReduce(geom::BBox<_Float, _Dimension> const& input, MPI_Comm comm);


} // namespace mpi
} // namespace stlib

#define __mpi_BBox_tcc__
#include "stlib/mpi/BBox.tcc"
#undef __mpi_BBox_tcc__

#endif
