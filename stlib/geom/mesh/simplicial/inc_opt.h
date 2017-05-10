// -*- C++ -*-

/*!
  \file geom/mesh/simplicial/inc_opt.h
  \brief Incidence optimization.
*/

#if !defined(__geom_mesh_simplicial_inc_opt_h__)
#define __geom_mesh_simplicial_inc_opt_h__

#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"
#include "stlib/geom/mesh/simplicial/flip.h"
#include "stlib/geom/mesh/simplicial/geometry.h"

#include "stlib/ads/algorithm/min_max.h"
#include "stlib/ads/iterator/TrivialOutputIterator.h"

#include "stlib/numerical/constants.h"

namespace stlib
{
namespace geom {

//! Modify the topology to optimize the cell-node incidences.
/*!
  \param mesh The simplicial mesh.
  \param norm The norm used in incidence optimization.  May be 0, 1, or 2.
  \param numSweeps The number of sweeps over the faces.  By default, the
  number of sweeps is not limited.

  \return the number of edges flipped.
*/
template < std::size_t N, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
std::size_t
incidenceOptimize(SimpMeshRed<N, 2, T, Node, Cell, Cont>* mesh,
                  std::size_t norm,
                  std::size_t numSweeps = 0);

} // namespace geom
}

//#define __geom_mesh_simplicial_inc_opt2_ipp__
//#include "stlib/geom/mesh/simplicial/inc_opt2.ipp"
//#undef __geom_mesh_simplicial_inc_opt2_ipp__

//#define __geom_mesh_simplicial_inc_opt3_ipp__
//#include "stlib/geom/mesh/simplicial/inc_opt3.ipp"
//#undef __geom_mesh_simplicial_inc_opt3_ipp__

#define __geom_mesh_simplicial_inc_opt_ipp__
#include "stlib/geom/mesh/simplicial/inc_opt.ipp"
#undef __geom_mesh_simplicial_inc_opt_ipp__

#endif
