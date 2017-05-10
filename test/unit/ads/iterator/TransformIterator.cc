// -*- C++ -*-

#include "stlib/ads/iterator/TransformIterator.h"

#include "stlib/ads/array/Array.h"

#include <algorithm>
#include <vector>

using namespace stlib;

int
main()
{
  {
    ads::Array<1, int> v(4);
    for (int n = 0; n != v.size(); ++n) {
      v[n] = n;
    }

    {
      typedef ads::TransformIterator < ads::Array<1, int>::const_iterator,
              std::negate<int> > TI;

      // Forward iteration.
      TI i(v.begin());
      for (ads::Array<1, int>::const_iterator j = v.begin(); j != v.end();
           ++i, ++j) {
        assert(*i == - *j);
      }

      // Random access iteration.
      i = v.begin();
      for (int n = 0; n != v.size(); ++n) {
        assert(i[n] == - v[n]);
      }
    }

    {
      typedef ads::Array<1, int>::const_iterator It;
      typedef std::negate<int> Tr;
      typedef ads::TransformIterator<It, Tr> TI;

      // Constructors and instatiation.
      {
        TI ti;
        ti = ads::constructTransformIterator<It, Tr>();
      }
      {
        It i(0);
        TI ti(i);
        ti = ads::constructTransformIterator<It, Tr>(i);
      }
      {
        Tr t;
        TI ti(t);
        ti = ads::constructTransformIterator<It, Tr>(t);
      }
      {
        It i(0);
        Tr t;
        TI ti(i, t);
        ti = ads::constructTransformIterator<It, Tr>(i, t);
      }
    }

    {
      typedef ads::TransformIterator < ads::Array<1, int>::iterator,
              std::negate<int> > TI;

      // Forward iteration.
      TI i(v.begin());
      for (ads::Array<1, int>::const_iterator j = v.begin(); j != v.end();
           ++i, ++j) {
        assert(*i == - *j);
      }

      // Random access iteration.
      i = v.begin();
      for (int n = 0; n != v.size(); ++n) {
        assert(i[n] == - v[n]);
      }
    }
  }

  {
    const int size = 10;
    ads::Array< 1, ads::FixedArray<3> > points(size);

    std::vector<double> rand(3 * size);
    for (std::size_t i = 0; i != rand.size(); ++i) {
      rand[i] = double(i) / double(rand.size());
    }
    std::random_shuffle(rand.begin(), rand.end());

    double sum0 = 0;
    {
      std::size_t i = 0;
      for (int n = 0; n != size; ++n) {
        sum0 += points[n][0] = rand[i++];
        points[n][1] = rand[i++];
        points[n][2] = rand[i++];
      }
    }

    // CONTINUE Replace binder2nd.
#if 0
    {
      // Check the sum of the x component.
      typedef ads::IndexConstObject<ads::FixedArray<3> > Index;
      Index index;
      typedef std::binder2nd<Index> IndexN;
      IndexN index0(index, 0);
      typedef ads::TransformIterator <
      ads::Array< 1, ads::FixedArray<3> >::const_iterator,
          IndexN > Index0Iter;
      Index0Iter i(points.begin(), index0);
      Index0Iter i_end(points.end(), index0);
      assert(sum0 == std::accumulate(i, i_end, 0.0));
    }
    {
      // Assign values to the y component.
      typedef ads::IndexObject<ads::FixedArray<3> > Index;
      Index index;
      typedef std::binder2nd<Index> IndexN;
      IndexN index1(index, 1);
      typedef ads::TransformIterator <
      ads::Array< 1, ads::FixedArray<3> >::iterator,
          IndexN > Index1Iter;
      Index1Iter i(points.begin(), index1);
      Index1Iter i_end(points.end(), index1);
      std::fill(i, i_end, 0);
      assert(std::accumulate(i, i_end, 0.0) == 0.0);
    }
#endif
  }

  // Use with IndexIterUnary.
  {
    int index[3] = { 0, 1, 2 };
    double value[3] = { 2, 3, 5 };
    ads::IndexIterUnary<double*> f(value);
    ads::TransformIterator<int*, ads::IndexIterUnary<double*> > i(f);
    i = index;
    for (int n = 0; n != 3; ++n, ++i) {
      assert(*i == value[n]);
    }
  }
  {
    int index[3] = { 0, 1, 2 };
    double value[3] = { 2, 3, 5 };
    ads::TransformIterator<int*, ads::IndexIterUnary<double*> > i =
      ads::constructArrayIndexingIterator<int*, double*>(index, value);
    for (int n = 0; n != 3; ++n, ++i) {
      assert(*i == value[n]);
    }
  }
  {
    int index[3] = { 0, 1, 2 };
    double value[3] = { 2, 3, 5 };
    assert(std::equal(value, value + 3,
                      ads::constructArrayIndexingIterator(index, value)));
  }

  {
    unsigned counting[5] = { 0, 1, 2, 3, 4 };
    unsigned prime[2] = { 2, 3 };
    int index[2] = { 2, 3 };
    assert(std::equal(prime, prime + 2,
                      ads::constructArrayIndexingIterator(index, counting)));
  }

  return 0;
}
