// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueLinearSearch.h"
#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueActive.h"

typedef ads::IndexedPriorityQueueLinearSearch <
ads::IndexedPriorityQueueActive<> > IndexedPriorityQueue;

#define __stochastic_nextReaction_main_ipp__
#include "main.ipp"
#undef __stochastic_nextReaction_main_ipp__
