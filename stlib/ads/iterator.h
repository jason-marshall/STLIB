// -*- C++ -*-

/*!
  \file iterator.h
  \brief Includes the iterator files.
*/

/*!
  \page ads_iterator Iterator Package

  ads::AdaptedIterator is a base class for all adapted iterators.

  The ads::IndirectIterator class allows you to treat a container of iterators
  to widgets like a container of widgets.

  ads::IntIterator is a random access iterator over an integer type.

  ads::MemFunIterator is an iterator that calls a member function in
  dereferencing.

  The ads::TransformIterator class dereferences and then applies a transform
  in the \c operator*() member function.

  ads::TrivialAssignable is a trivial assignable object.

  There is an implementation of a trivial output iterator
  in ads::TrivialOutputIterator .
*/

#if !defined(__ads_iterator_h__)
#define __ads_iterator_h__

#include "stlib/ads/iterator/AdaptedIterator.h"
#include "stlib/ads/iterator/IndirectIterator.h"
#include "stlib/ads/iterator/IntIterator.h"
#include "stlib/ads/iterator/MemFunIterator.h"
#include "stlib/ads/iterator/TransformIterator.h"
#include "stlib/ads/iterator/TrivialAssignable.h"
#include "stlib/ads/iterator/TrivialOutputIterator.h"

#endif
