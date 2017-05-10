// -*- C++ -*-

/*!
  \file numerical/random/poisson/PoissonPdfCdfAtTheMode.h
  \brief Probability density function and cumulative distribution function evaluated at the mode for the Poisson distribution.
*/

#if !defined(__numerical_PoissonPdfCdfAtTheMode_h__)
#define __numerical_PoissonPdfCdfAtTheMode_h__

#include "stlib/numerical/random/poisson/PoissonCdf.h"

#include "stlib/numerical/interpolation/hermite.h"

#include <vector>

namespace stlib
{
namespace numerical {

//! Probability density function and cumulative distribution function evaluated at the mode for the Poisson distribution.
/*!
  \param T The number type.  By default it is double.

  CONTINUE.
*/
template < typename T = double >
class PoissonPdfCdfAtTheMode {
   //
   // Public types.
   //
public:

   //! The number type.
   typedef T Number;

   //
   // Member data.
   //
private:

   // The lower bound of the range of means.
   Number _lowerBound;
   // Factor that will scale the argument to the index.
   Number _scaleToIndex;
   // Polynomial coefficients.
   std::vector<Number> _coefficients;

   //
   // Not implemented.
   //
private:

   //! Default constructor not implemented.
   PoissonPdfCdfAtTheMode();

public:

   //! Construct from the range of means and the number of patches per unit.
   PoissonPdfCdfAtTheMode(int closedLowerBound, int openUpperBound,
                          int numberOfPatchesPerUnit);

   //! Copy constructor.
   /*!
     \note This function is expensive.
   */
   PoissonPdfCdfAtTheMode(const PoissonPdfCdfAtTheMode& other);

   //! Assignment operator.
   /*!
     \note This function is expensive.
   */
   PoissonPdfCdfAtTheMode&
   operator=(const PoissonPdfCdfAtTheMode& other);

   //! Destructor.
   ~PoissonPdfCdfAtTheMode() {}

   //! Evaluate the probability density function and cumulative distribution function at the mode.
   void
   evaluate(Number mean, Number* pdf, Number* cdf) const;
};


} // namespace numerical
}

#define __numerical_random_PoissonPdfCdfAtTheMode_ipp__
#include "stlib/numerical/random/poisson/PoissonPdfCdfAtTheMode.ipp"
#undef __numerical_random_PoissonPdfCdfAtTheMode_ipp__

#endif
