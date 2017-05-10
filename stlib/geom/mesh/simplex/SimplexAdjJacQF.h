// -*- C++ -*-

/*!
  \file SimplexAdjJacQF.h
  \brief Quality function of the Jacobian and its adjoint.
*/

#if !defined(__geom_SimplexAdjJacQF_h__)
#define __geom_SimplexAdjJacQF_h__

#include "stlib/geom/mesh/simplex/SimplexJacQF.h"
#include "stlib/geom/mesh/simplex/SimplexAdjJac.h"

namespace stlib
{
namespace geom {

//! Quality function of the Jacobian and its adjoint.
/*!
  \param N is the dimension.
  \param T is the number type.  By default it is double.

  This is a base class for simplex quality functions that use the Jacobian
  matrix and its adjoint.  It has member functions for setting these
  matrices.
  - \c setFunction() sets the Jacobian matrix and its adjoint.
  - \c set() sets the Jacobian matrix and its adjoint and their gradients.
*/
template < std::size_t N, typename T = double >
class SimplexAdjJacQF :
   public SimplexJacQF<N, T> {
private:

   typedef SimplexJacQF<N, T> Base;
   typedef SimplexAdjJac<N, T> Adjoint;

public:

   //
   // Public types.
   //

   //! The number type.
   typedef T Number;

   //! The class for a vertex.
   typedef typename Base::Vertex Vertex;

   //! The simplex type.
   typedef typename Base::Simplex Simplex;

   //! An NxN matrix.
   typedef typename Base::Matrix Matrix;

protected:

   //
   // Member data.
   //

   //! The adjoint of the Jacobian.
   Adjoint _adjoint;

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //! @{

   //! Default constructor.  Un-initialized memory.
   SimplexAdjJacQF() :
      Base(),
      _adjoint() {}

   //! Copy constructor.
   SimplexAdjJacQF(const SimplexAdjJacQF& other) :
      Base(other),
      _adjoint(other._adjoint) {}

   //! Construct from a simplex.
   SimplexAdjJacQF(const Simplex& s) :
      Base(s),
      _adjoint(getMatrix()) {}

   //! Assignment operator.
   SimplexAdjJacQF&
   operator=(const SimplexAdjJacQF& other) {
      if (&other != this) {
         Base::operator=(other);
         _adjoint = other._adjoint;
      }
      return *this;
   }

   //! Trivial destructor.
   ~SimplexAdjJacQF() {}

   //! @}
   //--------------------------------------------------------------------------
   /*! \name Accessors.
   */
   //! @{

   //
   // Inherited from SimplexJacQF.
   //

   //! Return a const reference to the Jacobian matrix.
   using Base::getMatrix;

   //! Return a const reference to the gradient of the Jacobian matrix.
   using Base::getGradientMatrix;

   //! Return the determinant of the Jacobian matrix.
   using Base::getDeterminant;

   //! Return a const reference to the gradient of the determinant of the Jacobian matrix.
   using Base::getGradientDeterminant;

   //! Return the content (hypervolume) of the simplex.
   using Base::computeContent;

   //! Calculate or return the gradient of the content (hypervolume) of the simplex.
   using Base::computeGradientContent;

   //! Return the space dimension.
   using Base::getDimension;

   //
   // From SimplexAdjJac.
   //

   //! Return a const reference to the adjoint Jacobian matrix.
   const Matrix&
   getAdjointMatrix() const {
      return _adjoint.getMatrix();
   }

   //! Return a const reference to the gradient of the adjoint Jacobian matrix.
   const std::array<Matrix, N>&
   getAdjointGradientMatrix() const {
      return _adjoint.getGradientMatrix();
   }

   //! @}
   //--------------------------------------------------------------------------
   //! \name Manipulators
   //! @{

   //! Set the vertices in preparation for a function call.
   void
   setFunction(const Simplex& s) {
      Base::setFunction(s);
      _adjoint.setFunction(getMatrix());
   }

   //! Set the vertices in preparation for a function call or a gradient call.
   void
   set(const Simplex& s) {
      Base::set(s);
      _adjoint.set(getMatrix());
   }

   //! Set the vertices in preparation for a function call.
   /*!
     This first projects the simplex to N-D and then call the above
     set_function().
   */
   void
   setFunction(const std::array < std::array < Number, N + 1 > , N + 1 > & s) {
      Simplex t;
      projectToLowerDimension(s, &t);
      setFunction(t);
   }

   //! Set the vertices in preparation for a function call or a gradient call.
   /*!
     This first projects the simplex to N-D and then call the above set().
   */
   void
   set(const std::array < std::array < Number, N + 1 > , N + 1 > & s) {
      Simplex t;
      projectToLowerDimension(s, &t);
      set(t);
   }

   //! @}
};

} // namespace geom
}

#endif
