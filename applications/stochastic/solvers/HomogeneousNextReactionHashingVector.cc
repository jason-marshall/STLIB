// -*- C++ -*-

#include "ads/indexedPriorityQueue/IndexedPriorityQueueHashing.h"
#include "ads/indexedPriorityQueue/HashingChainingVector.h"

typedef ads::IndexedPriorityQueueHashing
<double, ads::HashingChainingVector<std::vector<double>::const_iterator> > IndexedPriorityQueue;

#define HASHING

#define __HomogeneousNextReaction_ipp__
#include "HomogeneousNextReaction.ipp"
#undef __HomogeneousNextReaction_ipp__
