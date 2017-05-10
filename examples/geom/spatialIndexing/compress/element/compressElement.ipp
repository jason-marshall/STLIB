// -*- C++ -*-

/*!
  \file examples/geom/spatialIndexing/compressElement.ipp
  \brief Define merging and coarsening using only the element.
*/

#ifndef __examples_geom_spatialIndexing_compressElement_ipp__
#error This is an implementation detail.
#endif

#include "stlib/ads/array/FixedArray.h"

namespace {

//
// Types.
//

//! The number type.
typedef double Number;

//
// Functors.
//

//! Average the child values.
struct Average {
   template<int _NumberOfOrthants>
   Number
   operator()(const ads::FixedArray<_NumberOfOrthants, const Number*>& elements) const {
      Number average = *elements[0];
      for (int n = 1; n != _NumberOfOrthants; ++n) {
         average += *elements[n];
      }
      average /= _NumberOfOrthants;
      return average;
   }
};

//! Return true if the element should be coarsened.
class Coarsen {
public:
   //! The result type.
   typedef bool result_type;

private:
   Number _maximumAllowedVariation;

public:

   void
   set(const Number maximumAllowedVariation) {
      _maximumAllowedVariation = maximumAllowedVariation;
   }

   template<int _NumberOfOrthants>
   result_type
   operator()(const ads::FixedArray<_NumberOfOrthants, const Number*>& elements) const {
      Number minimum = *elements[0];
      Number maximum = *elements[0];
      for (int n = 1; n != _NumberOfOrthants; ++n) {
         if (*elements[n] < minimum) {
            minimum = *elements[n];
         }
         else if (*elements[n] > maximum) {
            maximum = *elements[n];
         }
      }
      return maximum - minimum <= _maximumAllowedVariation;
   }
};

}
