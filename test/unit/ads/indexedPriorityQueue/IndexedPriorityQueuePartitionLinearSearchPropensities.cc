// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueuePartitionLinearSearchPropensities.h"

#define PROPENSITIES

using namespace stlib;

int
main()
{
  {
    typedef ads::IndexedPriorityQueuePartitionLinearSearchPropensities<>
    IndexedPriorityQueue;
#define __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
#include "IndexedPriorityQueue.ipp"
#undef __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
  }

  return 0;
}
