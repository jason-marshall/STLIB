// -*- C++ -*-

#define GIBSON_BRUCK_UPDATE

#include "ads/indexedPriorityQueue/IndexedPriorityQueueBinaryHeapPair.h"

typedef ads::IndexedPriorityQueueBinaryHeapPair<> IndexedPriorityQueue;

#define __HomogeneousNextReaction_ipp__
#include "HomogeneousNextReaction.ipp"
#undef __HomogeneousNextReaction_ipp__
