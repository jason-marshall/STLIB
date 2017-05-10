// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueHashing.h"
#include "stlib/ads/indexedPriorityQueue/HashingChainingVector.h"

typedef ads::IndexedPriorityQueueHashing
<double, ads::HashingChainingVector<ads::Array<1, double>::const_iterator> > IndexedPriorityQueue;

#define STOCHASTIC_NEXTREACTION_HASHING

#define __stochastic_nextReaction_main_ipp__
#include "main.ipp"
#undef __stochastic_nextReaction_main_ipp__
