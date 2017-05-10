// -*- C++ -*-

/*!
  \file EikonalScheme.h
  \brief Base class for finite difference schemes for the eikonal equation.
*/

#if !defined(__hj_EikonalScheme_h__)
#define __hj_EikonalScheme_h__

#include "stlib/hj/status.h"

namespace stlib
{
namespace hj {

//! Base class for finite differences for the eikonal equation.
/*!
  \param N is the space dimension.
  \param T is the number type.

  The \c Eikonal class defined the finite difference operations.
  Here we hold copies of the solution array and the status array and
  store the speed array.  Classes which derive from \c EikonalScheme
  will use these operations and arrays to implement finite
  difference schemes.
*/
template<std::size_t N, typename T>
class EikonalScheme {
protected:

   //
   // Member data
   //

   //! A reference for the solution array.
   container::MultiArrayRef<T, N> _solution;

   //! A reference for the status array.
   container::MultiArrayRef<Status, N> _status;

   //! The inverse speeed array.
   container::MultiArray<T, N> _inverseSpeed;

private:

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   EikonalScheme();
   //! Copy constructor not implemented.
   EikonalScheme(const EikonalScheme&);
   //! Assignment operator not implemented.
   EikonalScheme&
   operator=(const EikonalScheme&);

public:

   //
   // Constructors
   //

   //! Constructor.
   /*!
     \param solution is the solution array.
     \param status is the status array.
   */
   EikonalScheme(container::MultiArrayRef<T, N>& solution,
                 container::MultiArrayRef<Status, N>& status) :
      _solution(solution),
      _status(status),
      _inverseSpeed(solution.extents(), solution.bases()) {
   }

   //
   // Manipulators.
   //

   //! Return a reference to the inverse speed array.
   container::MultiArray<T, N>&
   getInverseSpeed() {
      return _inverseSpeed;
   }

   //
   // File I/O
   //

   //! Write the inverse speed array.
   void
   put(std::ostream& out) const;
};

//
// File I/O
//

//! Write to a file stream.
template<std::size_t N, typename T>
inline
std::ostream&
operator<<(std::ostream& out, const EikonalScheme<N, T>& x) {
   x.put(out);
   return out;
}

} // namespace hj
}

#define __hj_EikonalScheme_ipp__
#include "stlib/hj/EikonalScheme.ipp"
#undef __hj_EikonalScheme_ipp__

#endif
