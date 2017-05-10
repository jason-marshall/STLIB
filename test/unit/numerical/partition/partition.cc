// -*- C++ -*-

#include "stlib/numerical/partition.h"


using namespace stlib;

// Partition a vector of weights.
template<typename _T>
void
partitionWeights()
{
  using numerical::computePartitions;

  // Empty vector of weights.
  std::vector<_T> weights;
  {
    // 1 part.
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 1, std::back_inserter(delimiters));
    assert(delimiters.size() == 2);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 0);
  }
  {
    // 2 parts.
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 2, std::back_inserter(delimiters));
    assert(delimiters.size() == 3);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 0);
    assert(delimiters[2] == 0);
  }
  // One weight.
  weights.push_back(1);
  {
    // 1 part.
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 1, std::back_inserter(delimiters));
    assert(delimiters.size() == 2);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 1);
  }
  {
    // 2 parts.
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 2, std::back_inserter(delimiters));
    assert(delimiters.size() == 3);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 0 || delimiters[1] == 1);
    assert(delimiters[2] == 1);
  }
  // 2 weights.
  weights.push_back(2);
  {
    // 1 part.
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 1, std::back_inserter(delimiters));
    assert(delimiters.size() == 2);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 2);
  }
  {
    // 2 parts.
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 2, std::back_inserter(delimiters));
    assert(delimiters.size() == 3);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 1);
    assert(delimiters[2] == 2);
  }
  // 3 weights.
  weights.push_back(3);
  {
    // 1 part.
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 1, std::back_inserter(delimiters));
    assert(delimiters.size() == 2);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 3);
  }
  {
    // 2 parts.
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 2, std::back_inserter(delimiters));
    assert(delimiters.size() == 3);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 2);
    assert(delimiters[2] == 3);
  }
  {
    // 3 parts.
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 3, std::back_inserter(delimiters));
    assert(delimiters.size() == 4);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 2);
    assert(delimiters[1] <= delimiters[2] && delimiters[2] <= delimiters[3]);
    assert(delimiters[3] == 3);
  }

  //
  // Different scales.
  //
  weights.clear();
  weights.push_back(100);
  weights.push_back(1);
  {
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 2, std::back_inserter(delimiters));
    assert(delimiters.size() == 3);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 1);
    assert(delimiters[2] == 2);
  }
  weights.clear();
  weights.push_back(1);
  weights.push_back(100);
  {
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 2, std::back_inserter(delimiters));
    assert(delimiters.size() == 3);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 1);
    assert(delimiters[2] == 2);
  }
  weights.clear();
  weights.push_back(1);
  weights.push_back(100);
  weights.push_back(1);
  {
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 3, std::back_inserter(delimiters));
    assert(delimiters.size() == 4);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 1);
    assert(delimiters[2] == 2);
    assert(delimiters[3] == 3);
  }
  weights.clear();
  weights.push_back(100);
  weights.push_back(1);
  weights.push_back(1);
  {
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 3, std::back_inserter(delimiters));
    assert(delimiters.size() == 4);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 1);
    assert(delimiters[2] == 2);
    assert(delimiters[3] == 3);
  }
  {
    std::vector<std::size_t> delimiters;
    computePartitions(weights, 4, std::back_inserter(delimiters));
    assert(delimiters.size() == 5);
    assert(delimiters[0] == 0);
    assert(delimiters[1] == 1);
    assert(delimiters[2] == 2);
    assert(delimiters[3] == 3);
    assert(delimiters[4] == 3);
  }
}


int
main()
{
  using numerical::getPartition;
  using numerical::getPartitionRange;
  using numerical::computePartitions;

  int a, b;

  //
  // x = 0
  //
  {
    const int x = 0;
    // n = 1
    {
      const int n = 1;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 0);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 0);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 0);
    }
    // n = 2
    {
      const int n = 2;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 0);
      assert(getPartition(x, n, 1) == 0);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 0);
      getPartitionRange(x, n, 1, &a, &b);
      assert(a == 0 && b == 0);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 0);
      assert(delimiters[2] == 0);
    }
  }

  //
  // x = 1
  //
  {
    const int x = 1;

    // n = 1
    {
      const int n = 1;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 1);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 1);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
    }
    // n = 2
    {
      const int n = 2;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 1);
      assert(getPartition(x, n, 1) == 0);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 1);
      getPartitionRange(x, n, 1, &a, &b);
      assert(a == 1 && b == 1);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
      assert(delimiters[2] == 1);
    }
    // n = 3
    {
      const int n = 3;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 1);
      assert(getPartition(x, n, 1) == 0);
      assert(getPartition(x, n, 2) == 0);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 1);
      getPartitionRange(x, n, 1, &a, &b);
      assert(a == 1 && b == 1);
      getPartitionRange(x, n, 2, &a, &b);
      assert(a == 1 && b == 1);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
      assert(delimiters[2] == 1);
      assert(delimiters[3] == 1);
    }
  }

  //
  // x = 2
  //
  {
    const int x = 2;

    // n = 1
    {
      const int n = 1;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 2);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 2);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 2);
    }
    // n = 2
    {
      const int n = 2;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 1);
      assert(getPartition(x, n, 1) == 1);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 1);
      getPartitionRange(x, n, 1, &a, &b);
      assert(a == 1 && b == 2);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
      assert(delimiters[2] == 2);
    }
    // n = 3
    {
      const int n = 3;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 1);
      assert(getPartition(x, n, 1) == 1);
      assert(getPartition(x, n, 2) == 0);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 1);
      getPartitionRange(x, n, 1, &a, &b);
      assert(a == 1 && b == 2);
      getPartitionRange(x, n, 2, &a, &b);
      assert(a == 2 && b == 2);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
      assert(delimiters[2] == 2);
      assert(delimiters[3] == 2);
    }
  }

  //
  // x = 3
  //
  {
    const int x = 3;

    // n = 1
    {
      const int n = 1;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 3);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 3);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 3);
    }
    // n = 2
    {
      const int n = 2;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 2);
      assert(getPartition(x, n, 1) == 1);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 2);
      getPartitionRange(x, n, 1, &a, &b);
      assert(a == 2 && b == 3);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 2);
      assert(delimiters[2] == 3);
    }
    // n = 3
    {
      const int n = 3;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 1);
      assert(getPartition(x, n, 1) == 1);
      assert(getPartition(x, n, 2) == 1);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 1);
      getPartitionRange(x, n, 1, &a, &b);
      assert(a == 1 && b == 2);
      getPartitionRange(x, n, 2, &a, &b);
      assert(a == 2 && b == 3);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
      assert(delimiters[2] == 2);
      assert(delimiters[3] == 3);
    }
    // n = 4
    {
      const int n = 4;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 1);
      assert(getPartition(x, n, 1) == 1);
      assert(getPartition(x, n, 2) == 1);
      assert(getPartition(x, n, 3) == 0);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 1);
      getPartitionRange(x, n, 1, &a, &b);
      assert(a == 1 && b == 2);
      getPartitionRange(x, n, 2, &a, &b);
      assert(a == 2 && b == 3);
      getPartitionRange(x, n, 3, &a, &b);
      assert(a == 3 && b == 3);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
      assert(delimiters[2] == 2);
      assert(delimiters[3] == 3);
      assert(delimiters[4] == 3);
    }
  }

  //
  // x = 4
  //
  {
    const int x = 4;

    // n = 1
    {
      const int n = 1;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 4);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 4);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 4);
    }
    // n = 2
    {
      const int n = 2;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 2);
      assert(getPartition(x, n, 1) == 2);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 2);
      getPartitionRange(x, n, 1, &a, &b);
      assert(a == 2 && b == 4);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 2);
      assert(delimiters[2] == 4);
    }
    // n = 3
    {
      const int n = 3;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 2);
      assert(getPartition(x, n, 1) == 1);
      assert(getPartition(x, n, 2) == 1);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 2);
      getPartitionRange(x, n, 1, &a, &b);
      assert(a == 2 && b == 3);
      getPartitionRange(x, n, 2, &a, &b);
      assert(a == 3 && b == 4);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 2);
      assert(delimiters[2] == 3);
      assert(delimiters[3] == 4);
    }
    // n = 4
    {
      const int n = 4;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 1);
      assert(getPartition(x, n, 1) == 1);
      assert(getPartition(x, n, 2) == 1);
      assert(getPartition(x, n, 3) == 1);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 1);
      getPartitionRange(x, n, 1, &a, &b);
      assert(a == 1 && b == 2);
      getPartitionRange(x, n, 2, &a, &b);
      assert(a == 2 && b == 3);
      getPartitionRange(x, n, 3, &a, &b);
      assert(a == 3 && b == 4);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
      assert(delimiters[2] == 2);
      assert(delimiters[3] == 3);
      assert(delimiters[4] == 4);
    }
    // n = 5
    {
      const int n = 5;
      int delimiters[n + 1];

      assert(getPartition(x, n, 0) == 1);
      assert(getPartition(x, n, 1) == 1);
      assert(getPartition(x, n, 2) == 1);
      assert(getPartition(x, n, 3) == 1);
      assert(getPartition(x, n, 4) == 0);

      getPartitionRange(x, n, 0, &a, &b);
      assert(a == 0 && b == 1);
      getPartitionRange(x, n, 1, &a, &b);
      assert(a == 1 && b == 2);
      getPartitionRange(x, n, 2, &a, &b);
      assert(a == 2 && b == 3);
      getPartitionRange(x, n, 3, &a, &b);
      assert(a == 3 && b == 4);
      getPartitionRange(x, n, 4, &a, &b);
      assert(a == 4 && b == 4);

      computePartitions(x, n, delimiters);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
      assert(delimiters[2] == 2);
      assert(delimiters[3] == 3);
      assert(delimiters[4] == 4);
      assert(delimiters[5] == 4);
    }
  }

  partitionWeights<int>();
  partitionWeights<std::size_t>();
  partitionWeights<float>();

  // Partition weights using std::size_t.
  {
    std::vector<std::size_t> weights;

    {
      std::vector<std::size_t> delimiters;
      computePartitions(weights, 1, std::back_inserter(delimiters));
      assert(delimiters.size() == 2);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 0);
    }

    {
      std::vector<std::size_t> delimiters;
      computePartitions(weights, 2, std::back_inserter(delimiters));
      assert(delimiters.size() == 3);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 0);
      assert(delimiters[2] == 0);
    }

    weights.push_back(1);

    {
      std::vector<std::size_t> delimiters;
      computePartitions(weights, 1, std::back_inserter(delimiters));
      assert(delimiters.size() == 2);
      assert(delimiters[0] == 0);
      assert(delimiters.back() == weights.size());
    }

    {
      std::vector<std::size_t> delimiters;
      computePartitions(weights, 2, std::back_inserter(delimiters));
      assert(delimiters.size() == 3);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
      assert(delimiters.back() == weights.size());
    }

    weights.push_back(1);

    {
      std::vector<std::size_t> delimiters;
      computePartitions(weights, 1, std::back_inserter(delimiters));
      assert(delimiters.size() == 2);
      assert(delimiters[0] == 0);
      assert(delimiters.back() == weights.size());
    }

    {
      std::vector<std::size_t> delimiters;
      computePartitions(weights, 2, std::back_inserter(delimiters));
      assert(delimiters.size() == 3);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
      assert(delimiters.back() == weights.size());
    }

    {
      std::vector<std::size_t> delimiters;
      computePartitions(weights, 3, std::back_inserter(delimiters));
      assert(delimiters.size() == 4);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
      assert(delimiters[2] == 2);
      assert(delimiters.back() == weights.size());
    }

    weights.push_back(1);

    {
      std::vector<std::size_t> delimiters;
      computePartitions(weights, 1, std::back_inserter(delimiters));
      assert(delimiters.size() == 2);
      assert(delimiters[0] == 0);
      assert(delimiters.back() == weights.size());
    }

    {
      std::vector<std::size_t> delimiters;
      computePartitions(weights, 2, std::back_inserter(delimiters));
      assert(delimiters.size() == 3);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
      assert(delimiters.back() == weights.size());
    }

    {
      std::vector<std::size_t> delimiters;
      computePartitions(weights, 3, std::back_inserter(delimiters));
      assert(delimiters.size() == 4);
      assert(delimiters[0] == 0);
      assert(delimiters[1] == 1);
      assert(delimiters[2] == 2);
      assert(delimiters.back() == weights.size());
    }

    weights.resize(100);
    std::fill(weights.begin(), weights.end(), 1);

    {
      std::vector<std::size_t> delimiters;
      computePartitions(weights, 10, std::back_inserter(delimiters));
      assert(delimiters.size() == 11);
      assert(delimiters[0] == 0);
      assert(delimiters.back() == weights.size());
      for (std::size_t i = 0; i != 10; ++i) {
        assert(delimiters[i + 1] - delimiters[i] == 10);
      }
    }

    for (std::size_t i = 0; i != weights.size(); ++i) {
      weights[i] = i;
    }

    {
      std::vector<std::size_t> delimiters;
      computePartitions(weights, 10, std::back_inserter(delimiters));
      assert(delimiters.size() == 11);
      assert(delimiters[0] == 0);
      assert(delimiters.back() == weights.size());
      for (std::size_t i = 0; i != 10; ++i) {
        assert(delimiters[i + 1] > delimiters[i]);
      }
    }
  }

  return 0;
}
