// -*- C++ -*-

/*!
  \file numerical/optimization/CoordinateDescentHookeJeeves.h
  \brief The coordinate descent method of Hooke and Jeeves.
*/

#if !defined(__numerical_optimization_CoordinateDescentHookeJeeves_h__)
#define __numerical_optimization_CoordinateDescentHookeJeeves_h__

#include "stlib/numerical/optimization/ObjectiveFunction.h"

#include "stlib/ext/vector.h"

#include <limits>

#include <cassert>
#include <cmath>

namespace stlib
{
namespace numerical
{

USING_STLIB_EXT_VECTOR_MATH_OPERATORS;

//! The coordinate descent method of Hooke and Jeeves.
/*!
  \param _Function is the functor to minimize. The argument type of this
  functor must be \c std::vector<double>, and the return type \c double.
  The functor will be wrapped with an ObjectiveFunction. See the documentation
  for that class for the required interface.
*/
template<class _Function>
class CoordinateDescentHookeJeeves
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
  // Member data that the user can set.
  //
private:

  //! Const reference to the objective function.
  ObjectiveFunction<Function> _function;
  //! The initial step size.
  const Number _initialStepSize;
  //! The stepsize at which to halt optimization.
  const Number _finalStepSize;
  //! The stepsize reduction factor.
  const Number _stepSizeReductionFactor;

  //
  // Other member data.
  //

  //! The number of steps taken before the step size is increased.
  const std::size_t _stepLimit;
  //! The step size.
  Number _stepSize;

  //
  // Not implemented.
  //
private:

  // Default constructor not implemented.
  CoordinateDescentHookeJeeves();
  // Copy constructor not implemented.
  CoordinateDescentHookeJeeves(const CoordinateDescentHookeJeeves&);
  // Assignment operator not implemented.
  CoordinateDescentHookeJeeves&
  operator=(const CoordinateDescentHookeJeeves&);

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    The default constructor, the copy constructor and the assignment
    operator are not implemented. We use the default destructor.
  */
  // @{
public:

  //! Construct from the objective function.
  CoordinateDescentHookeJeeves
  (Function& function,
   Number initialStepSize
   = std::pow(std::numeric_limits<Number>::epsilon(), 0.25),
   Number finalStepSize = std::sqrt(std::numeric_limits<Number>::epsilon()));

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

protected:

  // Find a descent direction by moving in the coordinate directions.
  /*
    \param x is the starting point.  It will be set to the new position
    found by the search.
    \param value is initially equal to _function(x).  It will be set to
    the value of the objective function at the new position.
    \param delta will be set to the difference between the new and old
    position.

    \return Return true if the search takes one or more successful steps.
    Otherwise return false.
  */
  bool
  descentDirection(Vector* x, Number* value, Vector* delta);

  // Search in the specified coordinate and direction.
  /*
    \param x is the starting point.  It will be set to the new position
    found by the search.
    \param value is initially equal to _function(x).  It will be set to
    the value of the objective function at the new position.
    \param delta will be set to the difference between the new and old
    position.
    \param i is the coordinate in which to search.
    \param sign is the direction to search (1 or -1).

    \return Return true if the search takes one or more successful steps.
    Otherwise return false.
  */
  bool
  coordinateSearch(Vector* x, Number* value, Vector* delta,
                   std::size_t i, int sign);

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

  //! Return the initial step size.
  Number
  getInitialStepSize() const
  {
    return _initialStepSize;
  }

  //! Return the final step size.
  Number
  getFinalStepSize() const
  {
    return _finalStepSize;
  }

  //! Return the stepsize reduction factor.
  Number
  getStepSizeReductionFactor() const
  {
    return _stepSizeReductionFactor;
  }

  // @}
};

} // namespace numerical
}

#define __CoordinateDescentHookeJeeves_ipp__
#include "stlib/numerical/optimization/CoordinateDescentHookeJeeves.ipp"
#undef __CoordinateDescentHookeJeeves_ipp__

#endif
