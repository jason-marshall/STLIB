// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueHashing.h"
#include "stlib/ads/indexedPriorityQueue/HashingChainingGeneric.h"
#include <list>

typedef ads::Array<1, double>::const_iterator Iter;
typedef ads::IndexedPriorityQueueHashing
<double, ads::HashingChainingGeneric<Iter, std::list<Iter> > > IndexedPriorityQueue;

#define STOCHASTIC_NEXTREACTION_HASHING

#define __stochastic_nextReaction_main_ipp__
#include "main.ipp"
#undef __stochastic_nextReaction_main_ipp__
