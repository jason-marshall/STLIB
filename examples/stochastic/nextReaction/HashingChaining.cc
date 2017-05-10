// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueHashing.h"

typedef ads::IndexedPriorityQueueHashing<> IndexedPriorityQueue;

#define STOCHASTIC_NEXTREACTION_HASHING

#define __stochastic_nextReaction_main_ipp__
#include "main.ipp"
#undef __stochastic_nextReaction_main_ipp__
