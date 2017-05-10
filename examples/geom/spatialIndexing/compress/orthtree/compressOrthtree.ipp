// -*- C++ -*-

/*!
  \file examples/geom/spatialIndexing/compressOrthtree.ipp
  \brief Define merging and coarsening using the orthtree.
*/

#ifndef __examples_geom_spatialIndexing_compressOrthtree_ipp__
#error This is an implementation detail.
#endif

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
   template<typename _Orthtree>
   typename _Orthtree::Element
   operator()(const _Orthtree& orthtree,
              typename _Orthtree::const_iterator i) const {
      typename _Orthtree::Element average = i->second;
      ++i;
      for (int n = 1; n != _Orthtree::NumberOfOrthants; ++n, ++i) {
         average += i->second;
      }
      average /= _Orthtree::NumberOfOrthants;
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

   template<typename _Orthtree>
   result_type
   operator()(const _Orthtree& orthtree,
              typename _Orthtree::const_iterator i) const {
      Number minimum = i->second;
      Number maximum = i->second;
      ++i;
      for (int n = 1; n != _Orthtree::NumberOfOrthants; ++n, ++i) {
         if (i->second < minimum) {
            minimum = i->second;
         }
         else if (i->second > maximum) {
            maximum = i->second;
         }
      }
      return maximum - minimum <= _maximumAllowedVariation;
   }
};

}
