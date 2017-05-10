// -*- C++ -*-

/*!
  \file numerical/derivative/ridders.h
  \brief Numerical differentiation with centered difference schemes.
*/

#if !defined(__numerical_derivative_ridders_h__)
#define __numerical_derivative_ridders_h__

#include "stlib/ads/tensor/SquareMatrix.h"

#include <limits>

#include <cmath>

namespace stlib
{
namespace numerical
{

//-----------------------------------------------------------------------------
/*! \defgroup derivative_ridders Derivatives and Gradients with Ridders' Method

\par
These functions numerically evaluate derivatives or gradients of a
functor using \ref geom_ridders1982 "Ridders"' method. The implementations
are adapted from the \ref geom_press2007 "Numerical Recipes" function
\c dfridr().

\par Derivatives.
To calculate the derivative of a function, supply the functor that evaluates
the function and the function argument. The argument type and result type
for the functor must be \c double.

\code
class Functor : public std::unary_function<double, double> {
private:
   typedef std::unary_function<double, double> Base;
public:
   typedef typename Base::argument_type argument_type;
   typedef typename Base::result_type result_type;

   result_type
   operator()(const argument_type x) const {
      return 0.5 * x * x;
   }
};
...
Functor f;
double x = 1.;
double derivative = numerical::derivativeRidders(f, x);
\endcode

\par Scaling issues.
In evaluating the derivative the algorithm will use the value of the function
argument to attempt to determine an
appropriate range of arguments over which to evaluate the function.
These values are used to extrapolate the value of the derivative.
If the function is reasonably scaled, the selected range of arguments will
probably be appropriate. However, the selection criterion may fail if the
function varies on a very small or very large argument scale. In this case
one must indicate the appropriate scale in the derivative calculation.
This scale will be used as an initial step size in evaluating the
centered difference formula and should not necessarilly be very small.
Instead it should be an indication of the scale
over which the function changes appreciably.

\code
class Functor : public std::unary_function<double, double> {
private:
   typedef std::unary_function<double, double> Base;
public:
   typedef typename Base::argument_type argument_type;
   typedef typename Base::result_type result_type;

   result_type
   operator()(const argument_type x) const {
      return std::cos(1e6 * x);
   }
};
...
double scale = 1e-6;
double derivative = numerical::derivativeRidders(f, x, scale);
\endcode

\par Estimated error.
One may obtain an estimate of the %numerical error in value of the derivative
by passing an additional argument.

\code
double error;
double derivative = numerical::derivativeRidders(f, x, scale, &error);
\endcode

\par Gradients.
To calculate the gradient of a function, supply the functor that evaluates
the function, the argument vector, and a pointer to the gradient vector.
The L-2 norm of the estimated error in the gradient is returned.

\code
class Functor : public std::unary_function<std::vector<double>, double> {
private:
   typedef std::unary_function<std::vector<double>, double> Base;
public:
   typedef typename Base::argument_type argument_type;
   typedef typename Base::result_type result_type;

   result_type
   operator()(const argument_type& x) const {
      result_type result = 0;
      for (std::size_t n = 0; n != x.size(); ++n) {
         result += 0.5 * x[n] * x[n];
      }
      return result;
   }
};
...
Functor f;
std::vector<double> x(N);
...
std::vector<double> gradient(N);
double error = numerical::gradientRidders(f, x, &gradient);
\endcode

\par
The vector type must support indexing and have a size() member function.
Additionally the value type must be \c double. For example both
\c std::vector<double> and \c std::array<double,N> are suitable
types. The argument type of the supplied functor must be \c double
and the result type must be the vector type used for the function argument
and gradient.

\par Scaling issues.
As with evaluating the derivative, one may indicate scales when evaluating
the gradient. This is done for each dimension.

\code
class Functor : public std::unary_function<std::vector<double>, double> {
private:
   typedef std::unary_function<std::vector<double>, double> Base;
public:
   typedef typename Base::argument_type argument_type;
   typedef typename Base::result_type result_type;

   result_type
   operator()(const argument_type& x) const {
      assert(x.size() == 2);
      return std::cos(1e6 * x[0]) + std::sin(1e-6 * x[1]);
   }
};
...
std::vector<double> scales(2);
scales[0] = 1e-6;
scales[1] = 1e6;
double error = numerical::gradientRidders(f, x, &gradient, scales);
\endcode
*/
//@{


//----------------------------------------------------------------------------
// Derivative


//! Calculate the derivative using Ridders' method.
/*!
  \param f The function to differentiate.
  \param x The function argument.
  \param scale The scale over which the function changes appreciably.
  \param error The estimated error in the derivative.
  \param a The table used to evaluate polynomial approximations of the
  derivative.

  \return The approximate value of the derivative.
*/
template<typename _Functor>
typename _Functor::result_type
derivativeRidders(_Functor& f, typename _Functor::result_type x,
                  typename _Functor::result_type scale,
                  typename _Functor::result_type* error,
                  ads::SquareMatrix<10, typename _Functor::result_type>* a);


//! Calculate the derivative using Ridders' method.
/*!
  \param f The function to differentiate.
  \param x The function argument.
  \param scale The scale over which the function changes appreciably.
  \param error The estimated error in the derivative.

  \return The approximate value of the derivative.
*/
template<typename _Functor>
inline
typename _Functor::result_type
derivativeRidders(_Functor& f, const typename _Functor::result_type x,
                  const typename _Functor::result_type scale,
                  typename _Functor::result_type* error)
{
  ads::SquareMatrix<10, typename _Functor::result_type> a;
  return derivativeRidders(f, x, scale, error, &a);
}


//! Calculate the derivative using Ridders' method.
/*!
  \param f The function to differentiate.
  \param x The function argument.
  \param scale The scale over which the function changes appreciably.

  \return The approximate value of the derivative.
*/
template<typename _Functor>
inline
typename _Functor::result_type
derivativeRidders(_Functor& f, const typename _Functor::result_type x,
                  const typename _Functor::result_type scale)
{
  typename _Functor::result_type error;
  return derivativeRidders(f, x, scale, &error);
}


//! Calculate the derivative using Ridders' method.
/*!
  \param f The function to differentiate.
  \param x The function argument.

  \return The approximate value of the derivative.
*/
template<typename _Functor>
inline
typename _Functor::result_type
derivativeRidders(_Functor& f, const typename _Functor::result_type x)
{
  // Take a guess at an appropriate length scale over which the function
  // changes appreciably.
  const typename _Functor::result_type scale =
    std::max(0.001 * std::abs(x), 0.001);
  return derivativeRidders(f, x, scale);
}


//----------------------------------------------------------------------------
// Gradient.


//! Calculate the gradient using Ridders' method.
/*!
  \param f The function to differentiate.
  \param x The function argument.
  \param gradient The approximate value of the gradient.
  \param scales The scales over which the function changes appreciably.
  \param a The table used to evaluate polynomial approximations of the
  derivative.

  \return The l2-norm of the estimated error in the gradient.
*/
template<typename _Functor, typename _Vector>
typename _Functor::result_type
gradientRidders(_Functor& f, const _Vector& x, _Vector* gradient,
                const _Vector& scales,
                ads::SquareMatrix<10, typename _Functor::result_type>* a);


//! Calculate the gradient using Ridders' method.
/*!
  \param f The function to differentiate.
  \param x The function argument.
  \param gradient The approximate value of the gradient.
  \param scales The scales over which the function changes appreciably.

  \return The l2-norm of the estimated error in the gradient.
*/
template<typename _Functor, typename _Vector>
inline
typename _Functor::result_type
gradientRidders(_Functor& f, const _Vector& x, _Vector* gradient,
                const _Vector& scales)
{
  ads::SquareMatrix<10, typename _Functor::result_type> a;
  return gradientRidders(f, x, gradient, scales, &a);
}


//! Calculate the gradient using Ridders' method.
/*!
  \param f The function to differentiate.
  \param x The function argument.
  \param gradient The approximate value of the gradient.

  \return The l2-norm of the estimated error in the gradient.
*/
template<typename _Functor, typename _Vector>
typename _Functor::result_type
gradientRidders(_Functor& f, const _Vector& x, _Vector* gradient);


//@}

} // namespace numerical
}

#define __numerical_derivative_ridders_ipp__
#include "stlib/numerical/derivative/ridders.ipp"
#undef __numerical_derivative_ridders_ipp__

#endif
