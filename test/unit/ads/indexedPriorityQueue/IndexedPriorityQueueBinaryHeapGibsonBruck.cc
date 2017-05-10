// -*- C++ -*-

#define GIBSON_BRUCK_UPDATE

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueBinaryHeapPair.h"

using namespace stlib;

int
main()
{
  {
    typedef ads::IndexedPriorityQueueBinaryHeapPair<>
    IndexedPriorityQueue;
#define __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
#include "IndexedPriorityQueue.ipp"
#undef __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
  }

  return 0;
}
