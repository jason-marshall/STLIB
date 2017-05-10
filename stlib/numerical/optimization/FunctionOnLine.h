// -*- C++ -*-

/*!
  \file numerical/optimization/FunctionOnLine.h
  \brief Evaluate a multi-dimension function along a parametrized line.
*/

#if !defined(__numerical_optimization_FunctionOnLine_h__)
#define __numerical_optimization_FunctionOnLine_h__

#include "stlib/ext/vector.h"

#include <sstream>
#include <stdexcept>

namespace stlib
{
namespace numerical
{

//! Evaluate a multi-dimension function along a parametrized line.
/*!
  This class stores a reference to the supplied function and constant
  references to the base, and tangent. Thus the user is responsible for
  making sure that these are valid as long as this class is used.
*/
template<class _Function>
class FunctionOnLine
{
  //
  // Types.
  //
public:
  //! The function to evaluate.
  typedef _Function Function;
  //! A vector in N-D space.
  typedef typename Function::argument_type Vector;
  //! The argument type is a scalar.
  typedef typename Function::result_type argument_type;
  //! The result type.
  typedef typename Function::result_type result_type;

  //
  // Member data.
  //
private:

  //! Reference to the function defined on a vector space.
  Function& _function;
  const Vector& _base;
  const Vector& _tangent;
  Vector _x;
  Vector _gradient;

  //
  // Not implemented.
  //
private:

  // Default constructor not implemented.
  FunctionOnLine();

  // Copy constructor not implemented.
  FunctionOnLine(const FunctionOnLine&);

  // Assignment operator not implemented.
  FunctionOnLine&
  operator=(const FunctionOnLine&);

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    The default constructor, the copy constructor and the assignment
    operator are not implemented. We use the default destructor.
  */
  // @{
public:

  //! Construct from the function, a point on the line, and the tangent vector.
  FunctionOnLine(Function& function, const Vector& base,
                 const Vector& tangent) :
    _function(function),
    _base(base),
    _tangent(tangent),
    _x(base.size()),
    _gradient(base.size())
  {
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Function evaluation.
  // @{
public:

  //! Evaluate the function and return the result.
  result_type
  operator()(argument_type t)
  {
    for (std::size_t i = 0; i != _x.size(); ++i) {
      _x[i] = _base[i] + t * _tangent[i];
    }
    return _function(_x);
  }

  //! Evaluate the derivative.
  result_type
  derivative(argument_type t)
  {
    for (std::size_t i = 0; i != _x.size(); ++i) {
      _x[i] = _base[i] + t * _tangent[i];
    }
    _function(_x, &_gradient);
    return ext::dot(_gradient, _tangent);
  }

  // @}
};

} // namespace numerical
}

#endif
