// -*- C++ -*-

#include "ads/indexedPriorityQueue/IndexedPriorityQueueHashing.h"
#include "ads/indexedPriorityQueue/HashingChainingGeneric.h"

typedef std::vector<double>::const_iterator Iterator;
typedef ads::IndexedPriorityQueueHashing
<double, ads::HashingChainingGeneric<Iterator> > IndexedPriorityQueue;

#define HASHING

#define __HomogeneousNextReaction_ipp__
#include "HomogeneousNextReaction.ipp"
#undef __HomogeneousNextReaction_ipp__
