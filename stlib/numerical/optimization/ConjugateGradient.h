// -*- C++ -*-

/*!
  \file numerical/optimization/ConjugateGradient.h
  \brief The Fletcher-Reeves-Polak-Ribiere variant of the conjugate gradient method.
*/

#if !defined(__numerical_optimization_ConjugateGradient_h__)
#define __numerical_optimization_ConjugateGradient_h__

#include "stlib/numerical/optimization/Brent.h"
#include "stlib/numerical/optimization/DBrent.h"
#include "stlib/numerical/optimization/FunctionOnLine.h"
#include "stlib/numerical/optimization/ObjectiveFunction.h"

#include "stlib/ext/vector.h"

#include <cassert>
#include <cmath>

namespace stlib
{
namespace numerical
{

USING_STLIB_EXT_VECTOR_MATH_OPERATORS;

//! The Fletcher-Reeves-Polak-Ribiere variant of the conjugate gradient method.
/*!
  \param _Function is the functor to minimize. The argument type of this
  functor must be \c std::vector<double>, and the return type \c double.
  The functor will be wrapped with an ObjectiveFunction. See the documentation
  for that class for the required interface.
  \param _Minimizer1D is class which is templated on a 1-D function and
  performs minimization of that function.
*/
template<class _Function, template<class> class _Minimizer1D = DBrent>
class ConjugateGradient
{
  //
  // Public types.
  //
public:

  //! The objective function.
  typedef _Function Function;
  //! The number type.
  typedef typename Function::result_type Number;
  //! The vector of coordinates.
  typedef typename Function::argument_type Vector;

  //
  // Member data.
  //
private:


  //! Reference to the objective function.
  ObjectiveFunction<Function> _function;
  //! The fractional tolerance in evaluations of the objective function.
  /*! Minimization is complete when a line search fails to decrease the
    function value by more than this amount.*/
  const Number _fractionalTolerance;

  //
  // Not implemented.
  //
private:

  // Default constructor not implemented.
  ConjugateGradient();
  // Copy constructor not implemented.
  ConjugateGradient(const ConjugateGradient&);
  // Assignment operator not implemented.
  ConjugateGradient&
  operator=(const ConjugateGradient&);

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    The default constructor, the copy constructor and the assignment
    operator are not implemented. We use the default destructor.
  */
  // @{
public:

  //! Construct from the objective function and optionally the fractional tolerance.
  /*! The default value of the fractional tolerance is \c 3e-8. */
  ConjugateGradient(Function& function,
                    const Number fractionalTolerance = 3e-8) :
    _function(function, std::numeric_limits<std::size_t>::max()),
    _fractionalTolerance(fractionalTolerance)
  {
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Minimization.
  // @{
public:

  //! Find the minimum.
  /*!
    \param x The input value is the starting point.  The output value
    is the minimum point found.
    \return The value of the objective function at the minimum point.
  */
  Number
  minimize(Vector* x);

private:

  //! Perform a line minimization.
  /*!
    \return The function value of the minima along the line.

    Update the minima \c *x and set \c *direction to the step.
  */
  Number
  lineMinimization(Vector* x, Vector* direction);

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  //! Return a constant reference to the objective function.
  const Function&
  function() const
  {
    return _function.function();
  }

  //! Return the maximum allowed number of function calls.
  std::size_t
  maxFunctionCalls()
  {
    return _function.maxFunctionCalls();
  }

  //! Return the number of function calls required to find the minimum.
  std::size_t
  numFunctionCalls() const
  {
    return _function.numFunctionCalls();
  }

  // @}
};

} // namespace numerical
}

#define __ConjugateGradient_ipp__
#include "stlib/numerical/optimization/ConjugateGradient.ipp"
#undef __ConjugateGradient_ipp__

#endif
