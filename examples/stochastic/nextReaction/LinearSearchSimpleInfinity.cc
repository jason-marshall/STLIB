// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueLinearSearchSimple.h"

typedef ads::IndexedPriorityQueueLinearSearchSimple<double, true> IndexedPriorityQueue;

#define STOCHOSTIC_NEXTREACTION_USE_INFINITY

#define __stochastic_nextReaction_main_ipp__
#include "main.ipp"
#undef __stochastic_nextReaction_main_ipp__
