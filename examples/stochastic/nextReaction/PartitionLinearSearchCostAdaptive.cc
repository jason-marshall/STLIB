// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueuePartitionLinearSearchCostAdaptive.h"

typedef ads::IndexedPriorityQueuePartitionLinearSearchCostAdaptive<>
IndexedPriorityQueue;

#define BALANCE_COSTS

#define __stochastic_nextReaction_main_ipp__
#include "main.ipp"
#undef __stochastic_nextReaction_main_ipp__
