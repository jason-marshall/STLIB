// -*- C++ -*-

/*!
  \file ads/functor.h
  \brief Includes the functor files.
*/

/*!
  \page ads_functor Functor Package

  The functor package has:
  - functors for
  taking the \ref functor_address "address" of objects in Address.h .
  - a functor for
  \ref functor_dereference "dereferencing a handle"
  in Dereference.h .
  - a functor for
  \ref functor_handle_to_pointer "converting a handle to a pointer"
  in HandleToPointer.h .
  - the \ref functor_identity "identity" functor.
  - functors for
  \ref functor_compare_handle "comparing objects"
  by their handles in compare_handle.h .
  - functions and functors for
  \ref functor_composite_compare "comparing composite numbers"
  (Cartesian coordinates) in composite_compare.h .
  \ref functor_coordinateCompare "comparing coordinates of points"
  in coordinateCompare.h .
  - functors for
  \ref functor_compose "function composition"
  in compose.h
  - \ref functor_constant "constant functors" in constant.h .
  - \ref functor_linear "linear functors" in linear.h .
  - functors for \ref functor_index "indexing" in index.h .
*/

#if !defined(__ads_functor_h__)
#define __ads_functor_h__

#include "stlib/ads/functor/Address.h"
#include "stlib/ads/functor/Dereference.h"
#include "stlib/ads/functor/HandleToPointer.h"
#include "stlib/ads/functor/Identity.h"
#include "stlib/ads/functor/compare_handle.h"
#include "stlib/ads/functor/compose.h"
#include "stlib/ads/functor/composite_compare.h"
#include "stlib/ads/functor/constant.h"
#include "stlib/ads/functor/coordinateCompare.h"
#include "stlib/ads/functor/index.h"
#include "stlib/ads/functor/linear.h"
#include "stlib/ads/functor/select.h"

#endif
