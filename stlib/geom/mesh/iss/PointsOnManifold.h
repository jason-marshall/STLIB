// -*- C++ -*-

/*!
  \file PointsOnManifold.h
  \brief Represent the features of a mesh.
*/

#if !defined(__geom_PointsOnManifold_h__)
#define __geom_PointsOnManifold_h__

#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/geometry.h"
#include "stlib/geom/mesh/iss/quality.h"

#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"
#include "stlib/geom/mesh/simplicial/geometry.h"
#include "stlib/geom/kernel/SegmentMath.h"

#include "stlib/ads/algorithm/OrderedPair.h"
#include "stlib/container/MultiArray.h"
#include "stlib/numerical/constants.h"

#include <set>
#include <map>

namespace stlib
{
namespace geom {

//! Local closest point to a simplicial complex.
/*!
  \param N is the space dimension.
  \param M is the simplex dimension.
  \param SD is the spline degree.
  \param T is the number type.  By default it is double.
*/
template < std::size_t N, std::size_t M, std::size_t SD, typename T = double >
class PointsOnManifold;

} // namespace geom
}

// Include the implementations for 3-2 and N-1 meshes.
#include "stlib/geom/mesh/iss/PointsOnManifoldN11.h"
#include "stlib/geom/mesh/iss/PointsOnManifold321.h"

#endif
