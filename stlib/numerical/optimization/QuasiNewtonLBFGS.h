// -*- C++ -*-

/*!
  \file numerical/optimization/QuasiNewtonLBFGS.h
  \brief Limited-memory Broyden-Fletcher-Goldfarb-Shanno quasi-Newton method.
*/

/*
 *      C library of Limited memory BFGS (L-BFGS).
 *
 * Copyright (c) 1990, Jorge Nocedal
 * Copyright (c) 2007,2008,2009 Naoaki Okazaki
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#if !defined(__numerical_optimization_QuasiNewtonLBFGS_h__)
#define __numerical_optimization_QuasiNewtonLBFGS_h__

#include "stlib/numerical/optimization/ObjectiveFunction.h"
#include "stlib/numerical/optimization/FunctionOfSelectedCoordinates.h"
#include "stlib/numerical/optimization/ConjugateGradient.h"

#include "stlib/ext/vector.h"
#include "stlib/performance/Timer.h"

#include <cassert>
#include <cmath>

namespace stlib
{
namespace numerical
{

//! Limited-memory Broyden-Fletcher-Goldfarb-Shanno quasi-Newton method.
/*!
  <!------------------------------------------------------------------------>
  \section QuasiNewtonLBFGS_Overview Overview

  \param _Function is the objective function, the functor to minimize.

  \par Implementation.
  This class is adapted from
  <a href="http://www.chokkan.org/software/liblbfgs/">libLBFGS</a>:
  a library of Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS),
  version 1.9, which is distributed under the
  <a href="http://opensource.org/licenses/mit-license.php">MIT license</a>.

  \par The objective function.
  The \c argument_type of the objective function must be \c std::vector<T>,
  and the \c result_type must be \c T,
  the floating-point number type. This class stores a reference to the
  functor; it is not copied. Thus the user must ensure that it is available
  until the minimization is complete. The class for the objective function
  must overload the function call operator to provide one of the following
  member functions:

  \code
  result_type
  operator()(const argument_type& x, argument_type* gradient);

  result_type
  operator()(const argument_type& x, argument_type* gradient) const;
  \endcode

  \par
  Below we show how to define an appropriate functor to implement the
  objective function and how to calculate a local minimum. Note that the
  objective function inherits from \c std::unary_function. This base class
  defines \c argument_type and \c result_type.

  \note The objective function is not a unary function; its second argument
  is a pointer to the gradient, which is set by the function. However, there is
  no classification of such functions in the STL. \c std::unary_function
  is the best fit.

  \code
  ...
  #include "stlib/ext/vector.h"

  struct Quadratic :
     public std::unary_function<std::vector<double>, double> {
     result_type
     operator()(const argument_type& x, argument_type* gradient) const {
        for (std::size_t i = 0; i != x.size(); ++i) {
           (*gradient)[i] = 2. * x[i];
        }
        return dot(x, x);
     }
  };

  int
  main() {
     // The objective function.
     Quadratic f;
     // The function argument.
     std::vector<double> x(3, 1.);
     // Construct the minimizer.
     numerical::QuasiNewtonLBFGS<Quadratic> opt(f);
     try {
        const double value = opt.minimize(&x);
        std::cout << "The function attains a value of " << value
                  << " at its minimum: ";
        std::copy(x.begin(), x.end(), std::ostream_iterator<double>(std::cout, " "));
        std::cout << '\n';
     }
     catch (numerical::OptError& error) {
        std::cerr << error.what() << '\n';
     }
     return 0;
  }
  \endcode

  <!------------------------------------------------------------------------>
  \section QuasiNewtonLBFGS_Stopping Stopping Criteria

  \par Default stopping criterion.
  By default the minimization search terminates when the L<sub>2</sub> norm
  of the gradient becomes sufficiently small relative to the norm of \e x:
  \f$\|g\| < \epsilon \max(1, \|x\|)\f$. One may get or set the value of
  \f$\epsilon\f$ with the member functions getRelativeGradientTolerance() or
  setRelativeGradientTolerance(). The default value of the relative gradient
  tolerance is defined by the static member function
  getDefaultRelativeGradientTolerance().
  Increasing this tolerance will lead to less accurate minima at a
  decreased computational cost. To disable the relative gradient criterion
  set the tolerance to zero.

  \par
  Note that when the relative gradient tolerance is met, the current location
  may not be close to a local minimum. Even disregarding the finite tolerance
  and truncation errors, a vanishing gradient is a necessary, but not a
  sufficient condition for a minimum. That being said, the relative gradient
  criterion typically works well in practice.

  \par Other stopping criteria.
  One may enable additional stopping criteria to terminate the minimization.
  These either use the gradient or the rate of decrease in the value of
  the objective function. These may be useful when one has specific objectives
  for the accuracy in the minimum or when one simply wants to decrease the
  value of the function and not necessarily converge to a minimum. Note that
  when multiple criteria are employed the minimization terminates as soon
  as any one of the criteria are satisfied.

  \par RMS gradient.
  The root mean square (RMS) gradient criterion terminates the minimization
  when the RMS of the gradient is sufficiently small:
  \f$\|g\| / \sqrt{n} < \epsilon\f$. The member functions
  getRmsGradientTolerance() and setRmsGradientTolerance() allow one to
  get and set the tolerance. The default tolerance value, which
  is given by getDefaultRmsGradientTolerance(), is zero. To enable
  this criterion set the tolerance to a positive value. Note that this
  criterion is very similar to the relative gradient criterion. The difference
  is that the latter uses the positions to determine an appropriate scale
  for measuring the norm of the gradient.

  \par Max gradient.
  One may also terminate a minimization when the absolute value of every
  component of the gradient is less than a specified tolerance:
  \f$\max_i |g_i| < \epsilon\f$. The relevant accessor and manipulator are
  getMaxGradientTolerance() and setMaxGradientTolerance(). The default
  value of the tolerance, which is given by getDefaultMaxGradientTolerance()
  is zero.

  \par Rate of decrease.
  In addition to the gradient criteria one may use a rate of decrease
  criterion. The minimization terminates when \f$f' - f < \epsilon f\f$
  where \f$f'\f$ is the value of the objective function a specified number of
  iterations ago. The accessors areUsingRateOfDecrease() and
  getNumberOfIterationsForRateOfDecrease() tell wether the rate of decrease
  criterion is being used and the number of iterations in the past to
  use in the test. The rate of decrease criterion is not used by default.
  To enable it use the setNumberOfIterationsForRateOfDecrease() to set
  the number of iterations to a nonzero value. One may then change the
  default tolerance, given by getDefaultRateOfDecreaseTolerance(),
  with the setRateOfDecreaseTolerance() manipulator.

  \par
  The gradient criteria are based on mathematical properties of local
  minima. By contrast the rate of decrease criterion is based only on
  economics. It allows normal termination in cases where finding a
  minimum may be prohibitively expensive. More precisely it may indicate
  that actually converging to the minimum may give little benefit compared
  to the computational cost of doing so. Thus it is apparent why
  only the gradient criterion is used by default.

  \par
  In the example code below we demonstrate how to customize some of the stopping
  criteria introduced above.

  \code
  // Construct the minimizer.
  numerical::QuasiNewtonLBFGS<Quadratic> opt(f);
  // Increase the gradient tolerance.
  opt.setRelativeGradientTolerance(1e-3);
  // Enable the rate of decrease criterion and set the tolerance.
  opt.setNumberOfIterationsForRateOfDecrease(2);
  opt.setRateOfDecreaseTolerance(1e-3);
  \endcode

  <!------------------------------------------------------------------------>
  \section QuasiNewtonLBFGS_Exceptions Exceptions

  \par Exception class heirarchy.
  If the minimum cannot be determined to within the desired tolerances, the
  minimization routine will throw an exception. All of the utilized exception
  classes inherit from numerical::OptError, which inherits from
  \c std::runtime_error. There are a number of reasons
  why determining a local minimum may fail. We will consider these in turn.

  \par
  It is often useful to limit the allowed amount of computation in a
  minimization. If this limit is exceeded then the minimization throws an
  exception. The simplest way to impose such a limit
  is to set the maximum allowed time (in seconds) using setMaxTime(). By
  default there is no time limit. You can query the maximum allowed time
  with getMaxTime(). If the maximum allowed time is exceeded, a
  numerical::OptMaxTimeError will be thrown.

  \par
  One may set the maximum number of objective function evaluations with
  setMaxFunctionCalls(). The corresponding accessor is getMaxFunctionCalls().
  If the maximum number of evaluations is exceeded a
  numerical::OptMaxObjFuncCallsError will be thrown.

  \par
  A numerical::OptMaxIterationsError will be thrown in the following
  circumstances:
  - The maximum allowed number of line searches is exceeded.
  - The maximum number of steps within a line search is exceeded.
  - The maximum number of allowed resets for an ill-conditioned problem is
  exceeded.

  \par
  Note that each of numerical::OptMaxTimeError,
  numerical::OptMaxObjFuncCallsError, and numerical::OptMaxIterationsError
  inherit from numerical::OptMaxComputationError. Thus you can catch any of
  the above errors with this class.

  \par
  Sometimes an error is encountered when attempting a step In which case
  a numerical::OptStepError will be thrown. These include
  the following scenarios.
  - The calculated step size is non-positive.
  - The function does not decrease in the calculated descent direction.
  - The step size becomes too small or too large.
  .
  For all other errors that may be encountered the minimization routine will
  throw a numerical::OptError with a descriptive message.

  \par Handling exceptions.
  When minimizing
  a function, it is best to use a try/catch block because it is not uncommon
  for the minimization procedure to fail. In this case one may or may not
  want to use the result. (The function value at the returned argument will
  never be greater than the value at the initial position.) Below we show
  an example of catching each kind of error. The contents of some of the
  catch blocks has been omitted. All of the errors that the minimization
  routine may generate are handled. If some other type of error is encountered,
  perhaps a floating-point exception or a failed memory allocation, then
  an error message is printed and the exception is rethrown.

  \code
  try {
     opt.minimize(&x);
  }
  catch (numerical::OptMaxTimeError&) {
     ...
  }
  catch (numerical::OptMaxObjFuncCallsError&) {
     ...
  }
  catch (numerical::OptMaxIterationsError&) {
     ...
  }
  catch (numerical::OptStepError& error) {
     std::cerr << error.what() << '\n';
  }
  catch (numerical::OptError& error) {
     std::cerr << "Encountered a optimization error in QuasiNewtonLBFGS. "\
                  " Check that the objective function is implemented correctly.\n"
               << error.what() << '\n';
  }
  catch (...) {
     std::cerr << "Unexpected error.\n"
     throw;
  }
  \endcode

  \par
  Of course one does not have to distinguish between all of the different
  types of errors. Below is a simpler example. Here we don't catch unexpected
  errors. Either the calling function will handle such an error or the program
  will terminate with an error message.

  \code
  try {
     opt.minimize(&x);
  }
  catch (numerical::OptMaxComputationError&) {
     // Ignore max computation errors.
  }
  catch (numerical::OptError& error) {
     // Print an error message for other expected errors.
     std::cerr << error.what() << '\n';
  }
  // Don't catch unexpected errors.
  \endcode

  \par
  Below is an even simpler use case.

  \code
  try {
     opt.minimize(&x);
  }
  catch (numerical::OptError&) {
     // Ignore expected errors.
  }
  // Don't catch unexpected errors.
  \endcode

  \par Ignoring exceptions.
  One may enable/disable the throwing of exceptions with
  enableExceptions()/ disableExceptions().
  If exceptions are disabled then they
  are handled internally. That is minimize() and minimizeIllConditioned()
  terminate normally even when optimization exceptions are encountered.
  If exceptions are
  enabled one may ignore maximum computation errors by calling
  disableMaxComputationExceptions(). We demonstrate this in the example below.

  \code
  // Construct the minimizer.
  numerical::QuasiNewtonLBFGS<Quadratic> opt(f);
  // Allow no more than 50 objective function evaluations.
  opt.setMaxObjFuncCalls(50);
  // Allow no more than 2 seconds of computation to find the minimum.
  opt.setMaxTime(2);
  // Disable maximum computation exceptions.
  opt.disableMaxComputationExceptions();
  // Find a minimum.
  try {
     opt.minimize(&x);
  }
  catch (numerical::OptError& error) {
     // Print an error message for errors that are not a result of exceeding
     // the maximum amount of computation.
     std::cerr << error.what() << '\n';
  }
  // Don't catch unexpected errors.
  \endcode

  \par
  Below we ignore all optimization errors.

  \code
  // Construct the minimizer.
  numerical::QuasiNewtonLBFGS<Quadratic> opt(f);
  // Disable optimization exceptions.
  opt.disableExceptions();
  // Find a minimum.
  opt.minimize(&x);
  \endcode
*/
template<class _Function>
class QuasiNewtonLBFGS
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
  // Nested classes.
  //
private:

  class IterationData
  {
  public:
    Number alpha;
    Number ys;
    Vector s;
    Vector y;

    IterationData(const std::size_t n) :
      alpha(0),
      ys(0),
      s(n, 0.),
      y(n, 0.)
    {
    }
  };

  //
  // Member data.
  //
private:

  //! Reference to the objective function.
  ObjectiveFunction<Function> _function;
  //! The number of corrections to approximate the inverse hessian matrix.
  /*! The L-BFGS routine stores the computation results of previous \ref _m
   iterations to approximate the inverse hessian matrix of the current
   iteration. This parameter controls the size of the limited memories
   (corrections). The default value is \c 6. Values less than \c 3 are
   not recommended. Large values will result in excessive computing time. */
  const std::size_t _m;
  //! The relative gradient tolerance for the convergence test.
  /*! This parameter determines the accuracy with which the solution is to
    be found. A minimization terminates when
        ||g|| < \ref _relativeGradientTolerance * max(1, ||x||),
    where ||.|| denotes the Euclidean (L2) norm. The default value is
    given by getDefaultRelativeGradientTolerance() */
  Number _relativeGradientTolerance;
  //! The root mean square (RMS) gradient tolerance for the convergence test.
  /*! This parameter determines the accuracy with which the solution is to
    be found. A minimization terminates when
        ||g|| / sqrt(n) < \ref _rmsGradientTolerance
    where ||.|| denotes the Euclidean (L2) norm. The default value is
    given by getDefaultRmsGradientTolerance() */
  Number _rmsGradientTolerance;
  //! The maximum gradient tolerance for the convergence test.
  /*! This parameter determines the accuracy with which the solution is to
    be found. A minimization terminates when
        max_i(|g_i|) < \ref _maxGradientTolerance
    The default value is given by getDefaultMaxGradientTolerance() */
  Number _maxGradientTolerance;
  //! Distance for delta-based convergence test.
  /*! This parameter determines the distance, in iterations, to compute
    the rate of decrease of the objective function. If the value of this
    parameter is zero, the library does not perform the delta-based
    convergence test. The default value is \c 0. */
  std::size_t _past;
  //! Delta for convergence test.
  /*!  This parameter determines the minimum rate of decrease of the
    objective function. The library stops iterations when the
    following condition is met:
        (f' - f) / f < \ref _delta,
    where f' is the objective value of \ref _past iterations ago, and f is
    the objective value of the current iteration.
    The default value is \c 1e-5. */
  Number _delta;
  //! The maximum number of iterations.
  /*! The minimize() function terminates an optimization process with
    an OptMaxIterationsError exception when the iteration count
    exceeds this parameter. Setting this parameter to zero continues an
    optimization process until a convergence or error. The default value
    is \c 0. */
  const std::size_t _maxIterations;
  //! The maximum number of trials for the line search.
  /*! This parameter controls the number of function and gradients evaluations
    per iteration for the line search routine. The default value is \c 40. */
  const std::size_t _maxLinesearch;
  //! The maximum number of times to reset an ill-conditioned problem.
  /*! The minimizeIllConditioned() function terminates an optimization
    process with a OptMaxIterationsError exception when the number of resets
    exceeds this parameter. The default value is \c 100. */
  const std::size_t _maxResetIllConditioned;
  //! The minimum step of the line search routine.
  /*! The default value is \c 1e-20. This value need not be modified unless
    the exponents are too large for the machine being used, or unless the
    problem is extremely badly scaled (in which case the exponents should
    be increased). */
  const Number _minStep;
  //! The maximum step of the line search.
  /*! The default value is \c 1e+20. This value need not be modified unless
    the exponents are too large for the machine being used, or unless the
    problem is extremely badly scaled (in which case the exponents should
    be increased). */
  const Number _maxStep;
  //! A parameter to control the accuracy of the line search routine.
  /*! The default value is \c 1e-4. This parameter should be greater
    than zero and smaller than \c 0.5. */
  const Number _ftol;
  //! A parameter to control the accuracy of the line search routine.
  /*! The default value is \c 0.9. If the function and gradient
    evaluations are inexpensive with respect to the cost of the
    iteration (which is sometimes the case when solving very large
    problems) it may be advantageous to set this parameter to a small
    value. A typical small value is \c 0.1. This parameter should be
    greater than the \ref ftol parameter (\c 1e-4) and smaller than
    \c 1.0. */
  const Number _gtol;
  //! The machine precision for floating-point values.
  /*! This parameter must be a positive value set by a client program to
    estimate the machine precision. The line search routine will throw
    an OptError exception if the relative width
    of the interval of uncertainty is less than this parameter. */
  const Number _xtol;
  //! The maximum allowed time (in seconds) for a minimization.
  Number _maxTime;
  //! Whether other optimization errors should be thrown.
  bool _areThrowingExceptions;
  //! Whether maximum computation optimization errors should be thrown.
  bool _areThrowingMaxComputationExceptions;

  //
  // Not implemented.
  //
private:

  // Default constructor not implemented.
  QuasiNewtonLBFGS();
  // Copy constructor not implemented.
  QuasiNewtonLBFGS(const QuasiNewtonLBFGS&);
  // Assignment operator not implemented.
  QuasiNewtonLBFGS&
  operator=(const QuasiNewtonLBFGS&);

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    The default constructor, the copy constructor and the assignment
    operator are not implemented. We use the default destructor.
  */
  // @{
public:

  //! Construct from the objective function.
  QuasiNewtonLBFGS(Function& function) :
    _function(function, std::numeric_limits<std::size_t>::max()),
    _m(6),
    _relativeGradientTolerance(getDefaultRelativeGradientTolerance()),
    _rmsGradientTolerance(getDefaultRmsGradientTolerance()),
    _maxGradientTolerance(getDefaultMaxGradientTolerance()),
    _past(0),
    _delta(getDefaultRateOfDecreaseTolerance()),
    _maxIterations(0),
    _maxLinesearch(40),
    _maxResetIllConditioned(100),
    _minStep(1e-20),
    _maxStep(1e20),
    _ftol(1e-4),
    _gtol(0.9),
    _xtol(std::numeric_limits<Number>::epsilon()),
    _maxTime(std::numeric_limits<double>::max()),
    _areThrowingExceptions(true),
    _areThrowingMaxComputationExceptions(true)
  {
    // Check the parameters for errors.
    assert(_relativeGradientTolerance >= 0.);
    assert(_rmsGradientTolerance >= 0.);
    assert(_maxGradientTolerance >= 0.);
    assert(_delta >= 0.);
    assert(_minStep >= 0.);
    assert(_maxStep >= _minStep);
    assert(_ftol >= 0.);
    assert(_gtol >= 0.);
    assert(_xtol >= 0.);
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

  //! Attempt to find a local minimum for an ill-conditioned problem.
  /*! If the normal minimization process fails, a minimization will be
    performed on the group of variables with the largest gradients. */
  Number
  minimizeIllConditioned(Vector* x, std::size_t groupSize = 1);

private:

  //! Find the minimum.
  Number
  _minimize(Vector* x);

  Number
  _minimizeIllConditioned(Vector* x, std::size_t groupSize);

  bool
  hasGradientConverged(const Vector& x, const Vector& g) const;

  std::size_t
  findGroupWithMaximumGradient(const Vector& x, const std::size_t groupSize);

  void
  lineSearch(Vector* x, Number* f, Vector* g, const Vector& s,
             Number* stp, const Vector& xp);

  void
  updateTrialInterval(Number* x, Number* fx, Number* dx,
                      Number* y, Number* fy, Number* dy,
                      Number* t, Number* ft, Number* dt,
                      const Number tmin, const Number tmax, bool* isBracketed);

  //! Find a minimizer of an interpolated quadratic function.
  /*!
    \param u The value of one point, u.
    \param fu The value of f(u).
    \param du The value of f'(u).
    \param v The value of another point, v.
    \param fv The value of f(v).

    \return The minimizer of the interpolated quadratic.
  */
  Number
  quadraticMinimizer(const Number u, const Number fu, const Number du,
                     const Number v, const Number fv) const;

  //! Find a minimizer of an interpolated quadratic function.
  /*!
  \param u The value of one point, u.
  \param du The value of f'(u).
  \param v The value of another point, v.
  \param dv The value of f'(v).

  \return The minimizer of the interpolated quadratic.
  */
  Number
  quadraticMinimizer(const Number u, const Number du, const Number v,
                     const Number dv) const;

  //! Find a minimizer of an interpolated cubic function.
  /*!
   \param u The value of one point, u.
   \param fu The value of f(u).
   \param du The value of f'(u).
   \param v The value of another point, v.
   \param fv The value of f(v).
   \param du The value of f'(v).

   \return The minimizer of the interpolated cubic.
  */
  Number
  cubicMinimizer(const Number u, const Number fu, const Number du,
                 const Number v, const Number fv, const Number dv) const;

  //! Find a minimizer of an interpolated cubic function.
  /*!
   \param u The value of one point, u.
   \param fu The value of f(u).
   \param du The value of f'(u).
   \param v The value of another point, v.
   \param fv The value of f(v).
   \param du The value of f'(v).
   \param xmin The minimum value.
   \param xmax The maximum value.

   \return The minimizer of the interpolated cubic.
  */
  Number
  cubicMinimizer(const Number u, const Number fu, const Number du,
                 const Number v, const Number fv, const Number dv,
                 const Number xmin, const Number xmax) const;

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

  //! Return the maximum allowed number of objective function calls.
  std::size_t
  getMaxObjFuncCalls()
  {
    return _function.maxFunctionCalls();
  }

  //! Return the number of function calls required to find the minimum.
  std::size_t
  numFunctionCalls() const
  {
    return _function.numFunctionCalls();
  }

  //! Return the maximum allowed time for a minimization.
  double
  getMaxTime()
  {
    return _maxTime;
  }

  //! Return the relative gradient tolerance that is used to test for convergence.
  /*! The minimization terminates when \f$\|g\| < \epsilon \max(1, \|x\|)\f$.*/
  Number
  getRelativeGradientTolerance() const
  {
    return _relativeGradientTolerance;
  }

  //! Return the default relative gradient tolerance that is used to test for convergence.
  /*! The minimization terminates when \f$\|g\| < \epsilon \max(1, \|x\|)\f$.
   The default value of \f$\epsilon\f$ is 1e-5. */
  static
  Number
  getDefaultRelativeGradientTolerance()
  {
    return 1e-5;
  }

  //! Return the RMS gradient tolerance that is used to test for convergence.
  /*! The minimization terminates when \f$\|g\| / \sqrt{n} < \epsilon\f$.*/
  Number
  getRmsGradientTolerance() const
  {
    return _rmsGradientTolerance;
  }

  //! Return the default RMS gradient tolerance that is used to test for convergence.
  /*! The minimization terminates when \f$\|g\| / \sqrt{n} < \epsilon\f$.
    The default value of \f$\epsilon\f$ is 0. Thus this test is not used by
    default. */
  static
  Number
  getDefaultRmsGradientTolerance()
  {
    return 0;
  }

  //! Return the maximum gradient tolerance that is used to test for convergence.
  /*! The minimization terminates when \f$\max_i |g_i| < \epsilon\f$.*/
  Number
  getMaxGradientTolerance() const
  {
    return _maxGradientTolerance;
  }

  //! Return the default maximum gradient tolerance that is used to test for convergence.
  /*! The minimization terminates when \f$\max_i |g_i| < \epsilon\f$.
    The default value of \f$\epsilon\f$ is 0. Thus this test is not used by
    default. */
  static
  Number
  getDefaultMaxGradientTolerance()
  {
    return 0;
  }

  //! Return true if the rate of decrease tolerance is being used.
  bool
  areUsingRateOfDecrease() const
  {
    return _past != 0;
  }

  //! Return the number of iterations in the past used in the rate of decrease criterion.
  /*! If the number of iterations is zero the rate of decrease criterion is
    not used. */
  std::size_t
  getNumberOfIterationsForRateOfDecrease() const
  {
    return _past;
  }

  //! Return the rate of decrease tolerance that is used to test for convergence.
  /*! The minimization terminates when \f$f' - f < \delta f\f$ where
    \e f' is the value of the objective function \e past iterations ago. */
  Number
  getRateOfDecreaseTolerance() const
  {
    return _delta;
  }

  //! Return the default rate of decrease tolerance that is used to test for convergence.
  /*! The minimization terminates when \f$f' - f < \delta f\f$ where
    \e f' is the value of the objective function \e past iterations ago.
    The default value of \f$\delta\f$ is 1e-5. */
  static
  Number
  getDefaultRateOfDecreaseTolerance()
  {
    return 1e-5;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{
public:

  //! Set the maximum number of objective function calls allowed per optimization.
  void
  setMaxObjFuncCalls(const std::size_t maxFunctionCalls)
  {
    _function.setMaxFunctionCalls(maxFunctionCalls);
  }

  //! Set the maximum allowed time for a minimization.
  void
  setMaxTime(const double time)
  {
    _maxTime = time;
  }

  //! Set the relative gradient tolerance that is used to test for convergence.
  /*! The minimization terminates when \f$\|g\| < \epsilon \max(1, \|x\|)\f$.*/
  void
  setRelativeGradientTolerance(const Number epsilon)
  {
    _relativeGradientTolerance = epsilon;
  }

  //! Set the RMS gradient tolerance that is used to test for convergence.
  /*! The minimization terminates when \f$\|g\| / \sqrt{n} < \epsilon\f$.*/
  void
  setRmsGradientTolerance(const Number epsilon)
  {
    _rmsGradientTolerance = epsilon;
  }

  //! Set the maximum gradient tolerance that is used to test for convergence.
  /*! The minimization terminates when \f$\max_i |g_i| < \epsilon\f$.*/
  void
  setMaxGradientTolerance(const Number epsilon)
  {
    _maxGradientTolerance = epsilon;
  }

  //! Set the number of iterations in the past used in the rate of decrease criterion.
  /*! If the number of iterations is zero the rate of decrease criterion is
    not used. */
  void
  setNumberOfIterationsForRateOfDecrease(const std::size_t n)
  {
    _past = n;
  }

  //! Set the rate of decrease tolerance that is used to test for convergence.
  /*! The minimization terminates when \f$f' - f < \delta f\f$ where
    \e f' is the value of the objective function \e past iterations ago. */
  void
  setRateOfDecreaseTolerance(const Number delta)
  {
    _delta = delta;
  }

  //! Enable exceptions.
  /*! \note If exceptions are enabled, one may ignore maximum computation
   errors by calling disableMaxComputationExceptions(). */
  void
  enableExceptions()
  {
    _areThrowingExceptions = true;
  }

  //! Disable all exceptions.
  /*! This manipulation overides enableMaxComputationExceptions(). */
  void
  disableExceptions()
  {
    _areThrowingExceptions = false;
  }

  //! Disable maximum computation exceptions.
  void
  disableMaxComputationExceptions()
  {
    _areThrowingMaxComputationExceptions = false;
  }

  // @}
};

} // namespace numerical
}

#define __QuasiNewtonLBFGS_ipp__
#include "stlib/numerical/optimization/QuasiNewtonLBFGS.ipp"
#undef __QuasiNewtonLBFGS_ipp__

#endif
