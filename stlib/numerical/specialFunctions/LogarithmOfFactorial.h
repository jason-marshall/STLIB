// -*- C++ -*-

/*!
  \file numerical/specialFunctions/LogarithmOfFactorial.h
  \brief The logarithm of the factorial function.
*/

#if !defined(__numerical_LogarithmOfFactorial_h__)
#define __numerical_LogarithmOfFactorial_h__

#include <functional>

#include <cassert>
#include <cmath>

namespace stlib
{
namespace numerical
{

//-----------------------------------------------------------------------------
//! \defgroup numerical_specialFunctions_LogarithmOfFactorial The factorial function.
//@{

//! Return log(n!).
/*!
  \c T is the number type.  Note that you must specify the type explicitly
  as a template parameter.  It cannot be deduced.
*/
template<typename T>
T
computeLogarithmOfFactorial(int n);



//! The factorial functor.
/*!
  \c T is the number type.  By default it is double.
*/
template < typename T = double >
class LogarithmOfFactorial :
  public std::unary_function<int, T>
{
private:

  //
  // Private types.
  //

  typedef std::unary_function<int, T> Base;

public:

  //
  //Public types.
  //

  //! The argument type.
  typedef typename Base::argument_type argument_type;
  //! The result type.
  typedef typename Base::result_type result_type;

  // The default constructor, copy, assignment, and destructor are fine.

  //
  // Functor.
  //

  //! Return n!.
  result_type
  operator()(const argument_type n) const
  {
    return computeLogarithmOfFactorial<result_type>(n);
  }
};


//! Convenience function for constructing a \c LogarithmOfFactorial.
/*!
  \relates LogarithmOfFactorial
*/
template<typename T>
inline
LogarithmOfFactorial<T>
constructLogarithmOfFactorial()
{
  return LogarithmOfFactorial<T>();
}


//@}


} // namespace numerical
}

#define __numerical_specialFunctions_LogarithmOfFactorial_ipp__
#include "stlib/numerical/specialFunctions/LogarithmOfFactorial.ipp"
#undef __numerical_specialFunctions_LogarithmOfFactorial_ipp__

#endif
