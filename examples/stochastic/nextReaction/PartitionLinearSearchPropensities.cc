// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueuePartitionLinearSearchPropensities.h"

typedef ads::IndexedPriorityQueuePartitionLinearSearchPropensities<>
IndexedPriorityQueue;

#define STOCHASTIC_NEXTREACTION_PROPENSITIES_TIME
#define BALANCE_COSTS

#define __stochastic_nextReaction_main_ipp__
#include "main.ipp"
#undef __stochastic_nextReaction_main_ipp__
