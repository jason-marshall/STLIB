// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueuePartitionLinearSearchSizeAdaptive.h"

typedef ads::IndexedPriorityQueuePartitionLinearSearchSizeAdaptive<>
IndexedPriorityQueue;

#define BALANCE_COSTS

#define __stochastic_nextReaction_main_ipp__
#include "main.ipp"
#undef __stochastic_nextReaction_main_ipp__
