// -*- C++ -*-

#include "ads/indexedPriorityQueue/IndexedPriorityQueuePartitionLinearSearchCostAdaptive.h"

typedef ads::IndexedPriorityQueuePartitionLinearSearchCostAdaptive<>
IndexedPriorityQueue;

#define COST_CONSTANT

#define __HomogeneousNextReaction_ipp__
#include "HomogeneousNextReaction.ipp"
#undef __HomogeneousNextReaction_ipp__
