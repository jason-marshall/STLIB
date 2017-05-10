// -*- C++ -*-

#include "ads/indexedPriorityQueue/IndexedPriorityQueuePartitionLinearSearchSizeAdaptive.h"

typedef ads::IndexedPriorityQueuePartitionLinearSearchSizeAdaptive<>
IndexedPriorityQueue;

#define COST_CONSTANT

#define __HomogeneousNextReaction_ipp__
#include "HomogeneousNextReaction.ipp"
#undef __HomogeneousNextReaction_ipp__
