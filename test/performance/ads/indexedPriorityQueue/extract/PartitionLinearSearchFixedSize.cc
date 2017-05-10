// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueuePartitionLinearSearchFixedSize.h"

using namespace stlib;

typedef ads::IndexedPriorityQueuePartitionLinearSearchFixedSize<>
IndexedPriorityQueue;

#define BALANCE_COSTS

#define __ads_IndexedPriorityQueue_main_ipp__
#include "main.ipp"
#undef __ads_IndexedPriorityQueue_main_ipp__
