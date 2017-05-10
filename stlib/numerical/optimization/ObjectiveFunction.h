// -*- C++ -*-

/*!
  \file numerical/optimization/ObjectiveFunction.h
  \brief Objective function that checks the number of function calls.
*/

#if !defined(__numerical_optimization_ObjectiveFunction_h__)
#define __numerical_optimization_ObjectiveFunction_h__

#include "stlib/numerical/optimization/Exceptions.h"

#include <sstream>

namespace stlib
{
namespace numerical
{

//! Objective function that checks the number of function calls.
/*!
  \param _Function is the functor to minimize.

  This class wraps the function call operator in order to count the number
  of function and gradient calls. If the maximum allowed number of calls is
  exceeded (which indicates that the optimization method is converging too
  slowly, or not converging at all) then a \c OptMaxObjFuncCallsError
  exception is thrown.

  The given functor must supply the following member function for
  evaluating the function value.
  \code
  result_type operator()(const argument_type& x);
  \endcode

  If the functor can also compute gradients it must supply the following
  member function.
  \code
  void gradient(const argument_type& x, argument_type* x);
  \endcode

  This class stores a reference to the supplied functor. Thus the
  user is responsible for making sure that the functor is valid as long as
  this class is used.
*/
template<class _Function>
class ObjectiveFunction
{
  //
  // Types.
  //
private:
  //! The objective function.
  typedef _Function Function;
public:
  //! The argument type.
  typedef typename Function::argument_type argument_type;
  //! The result type.
  typedef typename Function::result_type result_type;

  //
  // Member data.
  //
private:

  //! Reference to the objective function.
  Function& _function;
  //! The maximum allowed number of function calls.
  std::size_t _maxFunctionCalls;
  //! The number of function calls required to find the minimum.
  std::size_t _numFunctionCalls;

  //
  // Not implemented.
  //
private:

  // Default constructor not implemented.
  ObjectiveFunction();

  // Copy constructor not implemented.
  ObjectiveFunction(const ObjectiveFunction&);

  // Assignment operator not implemented.
  ObjectiveFunction&
  operator=(const ObjectiveFunction&);

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    The default constructor, the copy constructor and the assignment
    operator are not implemented. We use the default destructor.
  */
  // @{
public:

  //! Construct from the objective function and the maximum allowed function calls.
  ObjectiveFunction(Function& function, const std::size_t maxFunctionCalls) :
    _function(function),
    _maxFunctionCalls(maxFunctionCalls),
    // Do this so they remember to call resetNumFunctionCalls().
    _numFunctionCalls(maxFunctionCalls) {}

  // @}
  //--------------------------------------------------------------------------
  /*! \name Calling the objective function.
    Each of these increment the count of the number of function calls. */
  // @{
public:

  //! Evaluate the objective function and return the result.
  result_type
  operator()(const argument_type& x)
  {
    incrementFunctionCalls();
    return _function(x);
  }

  //! Evaluate the gradient and return the objective function value.
  result_type
  operator()(const argument_type& x, argument_type* gradient)
  {
    incrementFunctionCalls();
    return _function(x, gradient);
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  //! Return a constant reference to the objective function.
  const Function&
  function() const
  {
    return _function;
  }

  //! Return the maximum allowed number of function calls.
  std::size_t
  maxFunctionCalls()
  {
    return _maxFunctionCalls;
  }

  //! Return the number of function calls required to find the minimum.
  std::size_t
  numFunctionCalls() const
  {
    return _numFunctionCalls;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{
public:

  //! Set the maximum number of function calls allowed per optimization.
  void
  setMaxFunctionCalls(const std::size_t maxFunctionCalls)
  {
    _maxFunctionCalls = maxFunctionCalls;
  }

  //! Reset the number of function calls to zero.
  void
  resetNumFunctionCalls()
  {
    _numFunctionCalls = 0;
  }

private:

  void
  incrementFunctionCalls()
  {
    if (_numFunctionCalls >= _maxFunctionCalls) {
      std::ostringstream message;
      message << "The maximum number of function calls " <<
              _maxFunctionCalls << " has been exceeded in the optimization.";
      throw OptMaxObjFuncCallsError(message.str());
    }
    ++_numFunctionCalls;
  }

  // @}
};

} // namespace numerical
}

#endif
