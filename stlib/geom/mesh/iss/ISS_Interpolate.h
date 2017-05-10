// -*- C++ -*-

/*!
  \file ISS_Interpolate.h
  \brief Interpolation on an indexed simplex set.
*/

#if !defined(__geom_ISS_Interpolate_h__)
#define __geom_ISS_Interpolate_h__

#include "stlib/geom/mesh/iss/ISS_SimplexQuery.h"
#include "stlib/geom/mesh/iss/ISS_VertexField.h"

namespace stlib
{
namespace geom {

//! Interpolation for an indexed simplex set with fields at the vertices.
/*!
  \param ISS is the indexed simplex set.
  \param F is the field type.  By default it is the number type of the mesh.

  This class stores an ISS_SimplexQuery for determining
  which which simplex to use in the interpolation and
  an ISS_VertexField for performing the linear interpolation.
*/
template < class ISS,
         typename F = typename ISS::Number >
class ISS_Interpolate {
   //
   // Private types.
   //

private:

   //! The indexed simplex set.
   typedef ISS IssType;

   //
   // Public types.
   //

public:

   //! The number type.
   typedef typename IssType::Number Number;
   //! A vertex.
   typedef typename IssType::Vertex Vertex;

   //
   // Field types.
   //

   //! The field type.
   typedef F Field;

   //
   // Data.
   //

private:

   //! The simplex query data structure.
   ISS_SimplexQuery<ISS> _simplexQuery;
   //! The interpolation data structure.
   ISS_VertexField<ISS, Field> _vertexField;

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   ISS_Interpolate();

   //! Assignment operator not implemented.
   ISS_Interpolate&
   operator=(const ISS_Interpolate&);

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //! @{

   //! Construct from the indexed simplex set and the fields.
   /*!
     \param iss is the indexed simplex set.
     \param fields is the array of fields.
   */
   ISS_Interpolate(const IssType& iss, const Field* fields) :
      _simplexQuery(iss),
      _vertexField(iss, fields) {}

   //! Copy constructor.
   ISS_Interpolate(const ISS_Interpolate& other) :
      _simplexQuery(other._simplexQuery),
      _vertexField(other._vertexField) {}

   //! Destructor.
   ~ISS_Interpolate() {}

   //! @}
   //--------------------------------------------------------------------------
   //! \name Mathematical Functions.
   //! @{

   //! Return the interpolated field for the point \c x.
   Field
   operator()(const Vertex& x) const {
      return _vertexField.interpolate
             (_simplexQuery.computeMinimumDistanceIndex(x), x);
   }

   //! @}
};

} // namespace geom
}

#endif
