// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueHashing.h"
#include "stlib/ads/indexedPriorityQueue/HashingChainingVector.h"

using namespace stlib;

typedef ads::IndexedPriorityQueueHashing<double, ads::HashingChainingVector<std::vector<double>::const_iterator> >
IndexedPriorityQueue;

#define HASHING

#define __ads_IndexedPriorityQueue_main_ipp__
#include "main.ipp"
#undef __ads_IndexedPriorityQueue_main_ipp__
