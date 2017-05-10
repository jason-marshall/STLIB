// -*- C++ -*-

/*!
  \file numerical/optimization/OptimizationBase.h
  \brief Base class for optimization methods.
*/

#if !defined(__numerical_optimization_OptimizationBase_h__)
#define __numerical_optimization_OptimizationBase_h__

#include <limits>
#include <sstream>
#include <stdexcept>

namespace stlib
{
namespace numerical
{

//! Base class for optimization methods.
/*!
  \param _Function is the functor to minimize.
*/
template<class _Function>
class OptimizationBase
{
  //
  // Types.
  //
protected:

  //! The objective function.
  typedef _Function Function;
  //! The argument type.
  typedef typename Function::argument_type argument_type;
  //! The result type.
  typedef typename Function::result_type result_type;
  //! The number type.
  typedef result_type Number;
  //! The vector of coordinates.
  typedef argument_type Vector;

  //
  // Member data.
  //
protected:

  //! Const reference to the objective function.
  const Function& _function;
  //! The maximum allowed number of function calls.
  std::size_t _maxFunctionCalls;
  //! The number of function calls required to find the minimum.
  std::size_t _numFunctionCalls;

  //
  // Not implemented.
  //
private:

  // Default constructor not implemented.
  OptimizationBase();

  // Copy constructor not implemented.
  OptimizationBase(const OptimizationBase&);

  // Assignment operator not implemented.
  OptimizationBase&
  operator=(const OptimizationBase&);

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    The default constructor, the copy constructor and the assignment
    operator are not implemented.  The implemented constructor and
    destructor are protected.
  */
  // @{
protected:

  //! Construct from the objective function.
  OptimizationBase(const Function& function,
                   // CONTINUE: Should there be a default?
                   const std::size_t maxFunctionCalls
                   = std::numeric_limits<std::size_t>::max()) :
    _function(function),
    _maxFunctionCalls(maxFunctionCalls),
    // Do this so they remember to call reset_numFunctionCalls().
    _numFunctionCalls(std::numeric_limits<std::size_t>::max()) {}

  // Use the default destructor.

  // @}
  //--------------------------------------------------------------------------
  /*! \name Calling the objective function.
    Each of these increment the count of the number of function calls. */
  // @{
protected:

  //! Evaluate the objective function and return the result.
  result_type
  evaluateFunction(const argument_type& x)
  {
    incrementFunctionCalls();
    return _function(x);
  }

  //! Evaluate the function and the gradient.
  result_type
  evaluateFunctionAndGradient(const argument_type& x,
                              argument_type* gradient)
  {
    incrementFunctionCalls();
    return _function(x, gradient);
  }

  //! Evaluate the gradient given the function value.
  /*! For some functions it is more efficient to evaluate the gradient if
    the function value is known. */
  void
  evaluateGradient(const argument_type& x, const Number value,
                   argument_type* gradient)
  {
    incrementFunctionCalls();
    _function.gradient(x, value, gradient);
  }

  //! Evaluate the gradient.
  void
  evaluateGradient(const argument_type& x, argument_type* gradient)
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

protected:

  //! Reset the number of function calls to zero.
  void
  resetNumFunctionCalls()
  {
    _numFunctionCalls = 0;
  }

private:

  void
  incrementFunctionCalls(const std::size_t increment = 1)
  {
    if (_numFunctionCalls >= _maxFunctionCalls) {
      std::ostringstream message;
      message << "The maximum number of function calls " <<
              _maxFunctionCalls << " has been exceeded in the optimization.";
      throw std::runtime_error(message.str());
    }
    _numFunctionCalls += increment;
  }

  // @}
};

} // namespace numerical
}

#endif
