// -*- C++ -*-

/*!
  \file numerical/specialFunctions/LogarithmOfFactorialCached.h
  \brief The logarithm of the factorial function. Uses a table of values.
*/

#if !defined(__numerical_LogarithmOfFactorialCached_h__)
#define __numerical_LogarithmOfFactorialCached_h__

#include <functional>
#include <numeric>
#include <vector>

#include <cassert>
#include <cmath>

namespace stlib
{
namespace numerical
{

//! The logarithm of the factorial function. Uses a table of values.
/*!
  \c T is the number type.  By default it is double.

  I do not provide a default constructor.  You must specify the size
  of the table of cached function values in the constructor.  You can
  only evaluate the
  function for these values, i.e. arguments in the range [0..size).
  Since copying the table of values is expensive, I do not provide
  a copy constructor or an assignment operator.  They should not be
  necessary; if used as an argument, this class
  should be passed by reference or const reference.

  CONTINUE: How accurate are the function values?
*/
template<typename T = double>
class LogarithmOfFactorialCached :
  public std::unary_function<std::size_t, T>
{
private:

  //
  // Private types.
  //

  typedef std::unary_function<std::size_t, T> Base;

public:

  //
  // Public types.
  //

  //! The argument type.
  typedef typename Base::argument_type argument_type;
  //! The result type.
  typedef typename Base::result_type result_type;
  //! The number type.
  typedef T Number;

  //
  // Member data.
  //

private:

  std::vector<Number> _values;

  //
  // Not implemented.
  //

private:

  //! Default constructor not implemented.
  LogarithmOfFactorialCached();

  //! Copy constructor not implemented.
  LogarithmOfFactorialCached(const LogarithmOfFactorialCached&);

  //! Assignment operator not implemented.
  LogarithmOfFactorialCached&
  operator=(const LogarithmOfFactorialCached&);

public:

  //! Size constructor.
  LogarithmOfFactorialCached(std::size_t size);

  //! Return log(n!).
  result_type
  operator()(const argument_type n) const
  {
#ifdef STLIB_DEBUG
    // If debugging is turned on, then the index range is checked.
    assert(n < _values.size());
#endif
    return _values[n];
  }
};


//@}


} // namespace numerical
}

#define __numerical_specialFunctions_LogarithmOfFactorialCached_ipp__
#include "stlib/numerical/specialFunctions/LogarithmOfFactorialCached.ipp"
#undef __numerical_specialFunctions_LogarithmOfFactorialCached_ipp__

#endif
