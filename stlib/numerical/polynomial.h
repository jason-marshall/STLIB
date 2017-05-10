// -*- C++ -*-

/*!
  \file numerical/polynomial.h
  \brief Polynomials.
*/

#if !defined(__numerical_polynomial_h__)
#define __numerical_polynomial_h__

#include "stlib/numerical/polynomial/Polynomial.h"

#endif

namespace stlib
{
namespace numerical
{

/*!
  \page numerical_polynomial Polynomials

  \par
  This package has functions for evaluating polynomials and their derivatives.
  They are grouped according to whether the polynomial order is
  \ref numerical_polynomial_static "known at compile-time" or
  whether the
  \ref numerical_polynomial_generic "order is generic".
  The classes
  numerical::Polynomial and numerical::PolynomialGenericOrder provide
  functors for these capabilities.
  There is also a function for
  \ref numerical_polynomial_differentiating "differentiating polynomials".

  \par
  There is a performance advantage if the order of the polynomial is known
  at compile time. Below is a table of execution times in nanoseconds
  for evaluating constant through cubic polynomials with the static
  generic order algorithms. Using the order-generic algorithm takes more
  than twice as long as the order-specific algorithm. This is because
  the compiler uses loop unrolling for the latter.

  <table border = "1" rules = "all">
  <tr> <th> Order <th> 0 <th> 1 <th> 2 <th> 3
  <tr> <th> Static <td> 1.586 <td> 1.311 <td> 1.802 <td> 2.386
  <tr> <th> Generic <td> 3.942 <td> 3.826 <td> 4.361 <td> 5.412
  </table>
*/

} // namespace numerical
}
