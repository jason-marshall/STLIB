// -*- C++ -*-

/*!
  \file stlib/container/EquilateralArrayRef.h
  \brief A multi-array that has equal extents in each dimension.
*/

#if !defined(__container_EquilateralArrayRef_h__)
#define __container_EquilateralArrayRef_h__

#include "stlib/container/EquilateralArrayImp.h"
#include "stlib/container/FixedArrayRef.h"

namespace stlib
{
namespace container
{

//! A multi-array that has equal extents in each dimension.
/*!
  \param _T The value type.
  \param _D The dimension
  \param N The extent in each dimension.
*/
template<typename _T, std::size_t _D, std::size_t N>
class EquilateralArrayRef :
  public EquilateralArrayImp<_T, _D, N, FixedArrayRef<_T, numerical::Exponentiation < std::size_t, N, _D >::Result> >
{
  //
  // Types.
  //
private:

  //! The base class.
  typedef EquilateralArrayImp<_T, _D, N, FixedArrayRef<_T, numerical::Exponentiation < std::size_t, N, _D >::Result> >
  Base;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor, assignment operator, and
    destructor. */
  // @{
public:

  //! Default constructor. Invalid data pointer.
  EquilateralArrayRef() :
    Base()
  {
  }

  //! Construct from a pointer to the array data.
  EquilateralArrayRef(typename Base::iterator data) :
    Base()
  {
    Base::setData(data);
  }

  // @}
};


} // namespace container
}

#endif
