// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueLinearSearchPartitionPropensities.h"

typedef ads::IndexedPriorityQueueLinearSearchPartitionPropensities<> IndexedPriorityQueue;

#define STOCHASTIC_NEXTREACTION_PROPENSITIES_TIME

#define __stochastic_nextReaction_main_ipp__
#include "main.ipp"
#undef __stochastic_nextReaction_main_ipp__
