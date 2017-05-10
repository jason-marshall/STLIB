// -*- C++ -*-

#include "ads/indexedPriorityQueue/IndexedPriorityQueueHashing.h"
#include "ads/indexedPriorityQueue/HashingChainingGeneric.h"
#include <list>

typedef std::vector<double>::const_iterator Iterator;
typedef ads::IndexedPriorityQueueHashing
<double, ads::HashingChainingGeneric<Iterator, std::list<Iterator> > > IndexedPriorityQueue;

#define HASHING

#define __HomogeneousNextReaction_ipp__
#include "HomogeneousNextReaction.ipp"
#undef __HomogeneousNextReaction_ipp__
