// -*- C++ -*-

/*!
  \file geom/mesh/simplicial/geometry.h
  \brief Geometric functions.
*/

#if !defined(__geom_mesh_simplicial_geometry_h__)
#define __geom_mesh_simplicial_geometry_h__

#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"
#include "stlib/geom/mesh/simplicial/topology.h"

#include "stlib/geom/mesh/simplex/geometry.h"

#include "stlib/numerical/constants.h"

namespace stlib
{
namespace geom {

//! Return the solid angle accumulated from the incident cells.
/*!
  This function does not check if the node is in the interior or on the
  boundary.  For the sake of efficiency, only call this function for boundary
  nodes.
*/
template<class SMR>
typename SMR::Number
computeIncidentCellsAngle(const typename SMR::Node* node);



//! Compute the dihedral angle at the specified edge.
/*!
  The dihedral angle is accumulated from the incident cells.
*/
template<class SMR>
typename SMR::Number
computeDihedralAngle(typename SMR::ConstEdge edge);



//! Return the cosine of the interior angle at the specified 1-face.
/*!
  \pre The 1-face must have two incident simplices.
*/
template<class SMR>
typename SMR::Number
computeCosineAngle(typename SMR::FaceConstIterator face);



//! Functor that returns true iff the boundary angle is not sharp.
template<class SMR>
class IsNotSharpAngle :
   public std::unary_function<const typename SMR::Node*, bool> {
   // Types.
private:
   typedef std::unary_function<const typename SMR::Node*, bool>
   Base;

public:
   //! The number type.
   typedef typename SMR::Number Number;
   //! The argument type is a node const iterator.
   typedef typename Base::argument_type argument_type;
   //! The result type is a Boolean.
   typedef typename Base::result_type result_type;

private:
   // Data.
   Number _minAngle;

   // Not Implemented.
   IsNotSharpAngle();
   IsNotSharpAngle& operator=(const IsNotSharpAngle&);

public:

   //! Copy constructor.
   IsNotSharpAngle(const IsNotSharpAngle& x) :
      _minAngle(x._minAngle) {}

   //! Angle constructor.
   IsNotSharpAngle(const Number minAngle) :
      _minAngle(minAngle) {}

   //! Return true iff the boundary angle is not sharp.
   /*!
     This does not check if the node is on the boundary.
   */
   result_type
   operator()(argument_type x) const {
      Number a = computeIncidentCellsAngle<SMR>(x);
      return _minAngle <= a &&
             a <= 2.0 * numerical::Constants<Number>::Pi() - _minAngle;
   }
};



//! Compute the normal to the surface.
/*!
  \param node Must be a boundary node.
  \param normal Set to the node normal.
*/
template<class SMR>
void
computeNodeNormal(const typename SMR::Node* node,
                  typename SMR::Vertex* normal);


//! Return the normal to the surface.
/*!
  \param node Must be a boundary node.
*/
template<class SMR>
inline
typename SMR::Vertex
computeNodeNormal(const typename SMR::Node* node) {
   typename SMR::Vertex normal;
   computeNodeNormal<SMR>(node, &normal);
   return normal;
}


//! Compute the cell normal.
/*!
  \note The space dimension must be one more than the simplex dimension,
  N == M + 1.
*/
template<class SMR>
void
computeCellNormal(typename SMR::CellConstIterator cell,
                  typename SMR::Vertex* normal);


//! Return the normal to the surface.
/*!
  \note The space dimension must be one more than the simplex dimension,
  N == M + 1.
*/
template<class SMR>
inline
typename SMR::Vertex
computeCellNormal(typename SMR::CellConstIterator cell) {
   typename SMR::Vertex normal;
   computeCellNormal<SMR>(cell, &normal);
   return normal;
}


//! Compute the face normal.
/*!
  \note The space dimension must be equal to the simplex dimension,
  N == M.
*/
template<class SMR>
void
computeFaceNormal(typename SMR::CellConstIterator cell, std::size_t i,
                  typename SMR::Vertex* x);




//! Project the line segments to 1-D and collect them.
template < typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         typename OutputIterator >
void
projectAndGetSimplices(const SimpMeshRed<2, 1, T, Node, Cell, Cont>& mesh,
                       OutputIterator simplices);


//! Project the triangle simplices to 2-D and collect them.
template < typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         typename OutputIterator >
void
projectAndGetSimplices(const SimpMeshRed<3, 2, T, Node, Cell, Cont>& mesh,
                       OutputIterator simplices);


} // namespace geom
}

#define __geom_mesh_simplicial_geometry_ipp__
#include "stlib/geom/mesh/simplicial/geometry.ipp"
#undef __geom_mesh_simplicial_geometry_ipp__

#endif
