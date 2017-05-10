// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueHashing.h"
#include "stlib/ads/indexedPriorityQueue/HashingChainingGeneric.h"
#include <list>

using namespace stlib;

typedef std::vector<double>::const_iterator Iter;
typedef ads::IndexedPriorityQueueHashing
<double, ads::HashingChainingGeneric<Iter, std::list<Iter> > >
IndexedPriorityQueue;

#define HASHING

#define __ads_IndexedPriorityQueue_main_ipp__
#include "main.ipp"
#undef __ads_IndexedPriorityQueue_main_ipp__
