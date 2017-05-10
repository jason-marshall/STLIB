// -*- C++ -*-

#include "ads/indexedPriorityQueue/IndexedPriorityQueueHashing.h"

typedef ads::IndexedPriorityQueueHashing<> IndexedPriorityQueue;

#define HASHING

#define __HomogeneousNextReaction_ipp__
#include "HomogeneousNextReaction.ipp"
#undef __HomogeneousNextReaction_ipp__
