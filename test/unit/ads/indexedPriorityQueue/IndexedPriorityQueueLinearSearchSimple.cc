// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueLinearSearchSimple.h"

using namespace stlib;

int
main()
{
  {
    typedef ads::IndexedPriorityQueueLinearSearchSimple<>
    IndexedPriorityQueue;
#define __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
#include "IndexedPriorityQueue.ipp"
#undef __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
  }

  // CONTINUE
#if 0
  {
    typedef ads::IndexedPriorityQueueLinearSearchSimple<double, false>
    IndexedPriorityQueue;
#define __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
#include "IndexedPriorityQueue.ipp"
#undef __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
  }
#endif
  return 0;
}
