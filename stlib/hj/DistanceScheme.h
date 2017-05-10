// -*- C++ -*-

/*!
  \file DistanceScheme.h
  \brief Base class for finite difference schemes for computing distance.
*/

#if !defined(__hj_DistanceScheme_h__)
#define __hj_DistanceScheme_h__

#include "stlib/hj/status.h"

#include "stlib/container/MultiArray.h"

namespace stlib
{
namespace hj {

//! Base class for finite differences for computing distance.
/*!
  \param N is the space dimension.
  \param T is the number type.

  The \c Distance class defined the finite difference operations.
  Here we hold copies of the solution array and the status array.
  Classes which derive from \c DistanceScheme will use these operations
  and arrays to implement finite difference schemes.
*/
template<std::size_t N, typename T>
class DistanceScheme {
protected:

   //
   // Member data
   //

   //! A reference for the solution array.
   container::MultiArrayRef<T, N> _solution;

   //! A reference for the status array.
   container::MultiArrayRef<Status, N> _status;

private:

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   DistanceScheme();
   //! Copy constructor not implemented.
   DistanceScheme(const DistanceScheme&);
   //! Assignment operator not implemented.
   DistanceScheme&
   operator=(const DistanceScheme&);

public:

   //
   // Constructors
   //

   //! Constructor.
   /*!
     \param solution is the solution array.
     \param status is the status array.
   */
   DistanceScheme(container::MultiArrayRef<T, N>& solution,
                  container::MultiArrayRef<Status, N>& status) :
      _solution(solution),
      _status(status) {}

   //
   // File I/O
   //

   //! Write that distance is being computed.
   void
   put(std::ostream& out) const {
      out << "This is an equation for computing distance.\n";
   }

};

//
// File I/O
//

//! Write to a file stream.
template<std::size_t N, typename T>
inline
std::ostream&
operator<<(std::ostream& out, const DistanceScheme<N, T>& x) {
   x.put(out);
   return out;
}

} // namespace hj
}

#endif
