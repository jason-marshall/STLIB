// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueLinearSearch.h"

using namespace stlib;

int
main()
{
  {
    typedef ads::IndexedPriorityQueueLinearSearch<>
    IndexedPriorityQueue;
#define __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
#include "IndexedPriorityQueue.ipp"
#undef __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
  }

  return 0;
}
