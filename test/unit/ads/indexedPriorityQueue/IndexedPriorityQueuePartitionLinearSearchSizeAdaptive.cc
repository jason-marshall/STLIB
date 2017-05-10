// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueuePartitionLinearSearchSizeAdaptive.h"

using namespace stlib;

int
main()
{
  {
    typedef ads::IndexedPriorityQueuePartitionLinearSearchSizeAdaptive<>
    IndexedPriorityQueue;
#define __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
#include "IndexedPriorityQueue.ipp"
#undef __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
  }

  return 0;
}
