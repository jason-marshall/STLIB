// -*- C++ -*-

#include "stlib/container/PackedArrayOfArrays.h"

#include <cstdint>

using namespace stlib;

template<typename _Integer>
void
testTransposeInteger()
{
  container::PackedArrayOfArrays<_Integer> x;
  container::PackedArrayOfArrays<_Integer> t;

  transpose(x, &t);
  assert(t.empty());
  assert(t.numArrays() == 0);

  // 0 |
  x.pushArray();
  transpose(x, &t);
  assert(t.empty());
  assert(t.numArrays() == 0);

  transpose(x, 3, &t);
  assert(t.empty());
  assert(t.numArrays() == 3);

  // 0 | 2
  // 1 |
  // 2 | 2 3
  x.push_back(2);
  x.pushArray();
  x.pushArray();
  x.push_back(2);
  x.push_back(3);
  transpose(x, &t);
  assert(t.size() == x.size());
  assert(t.numArrays() == 4);
  assert(t.size(0) == 0);
  assert(t.size(1) == 0);
  assert(t.size(2) == 2);
  assert(t(2, 0) == 0);
  assert(t(2, 1) == 2);
  assert(t.size(3) == 1);
  assert(t(3, 0) == 2);
}


void
testTransposeIntegerOverflow()
{
  typedef std::uint8_t Integer;
  container::PackedArrayOfArrays<Integer> x;
  container::PackedArrayOfArrays<Integer> t;

  for (std::size_t i = 0;
       i != std::size_t(std::numeric_limits<Integer>::max()) + 2; ++i) {
    x.pushArray();
  }
  try {
    transpose(x, 1, &t);
    throw std::runtime_error("Expected an overflow error.");
  }
  catch (std::overflow_error) {
  }
}


template<typename _Integer>
void
testTransposePair()
{
  typedef std::pair<_Integer, _Integer> Pair;
  container::PackedArrayOfArrays<Pair> x;
  container::PackedArrayOfArrays<Pair> t;

  transpose(x, &t);
  assert(t.empty());
  assert(t.numArrays() == 0);

  // 0 |
  x.pushArray();
  transpose(x, &t);
  assert(t.empty());
  assert(t.numArrays() == 0);

  transpose(x, 3, &t);
  assert(t.empty());
  assert(t.numArrays() == 3);

  // x = 
  // 0 | (2, 3)
  // 1 |
  // 2 | (2, 3) (3, 4)
  // t = 
  // 0 | 
  // 1 | 
  // 2 | (0, 3) (2, 3)
  // 3 | (2, 4)
  x.push_back(Pair(2, 3));
  x.pushArray();
  x.pushArray();
  x.push_back(Pair(2, 3));
  x.push_back(Pair(3, 4));
  transpose(x, &t);
  assert(t.size() == x.size());
  assert(t.numArrays() == 4);
  assert(t.size(0) == 0);
  assert(t.size(1) == 0);
  assert(t.size(2) == 2);
  assert(t(2, 0) == Pair(0, 3));
  assert(t(2, 1) == Pair(2, 3));
  assert(t.size(3) == 1);
  assert(t(3, 0) == Pair(2, 4));
}


void
testTransposePairOverflow()
{
  typedef std::uint8_t Integer;
  typedef std::pair<Integer, Integer> Pair;
  container::PackedArrayOfArrays<Pair> x;
  container::PackedArrayOfArrays<Pair> t;

  for (std::size_t i = 0;
       i != std::size_t(std::numeric_limits<Integer>::max()) + 2; ++i) {
    x.pushArray();
  }
  try {
    transpose(x, 1, &t);
    throw std::runtime_error("Expected an overflow error.");
  }
  catch (std::overflow_error) {
  }
}


int
main()
{
  {
    // Default constructor.
    container::PackedArrayOfArrays<double> x;
    assert(x.numArrays() == 0);
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.max_size() > 0);
    assert(x.begin() == x.end());

    x.append(x);
    assert(x.numArrays() == 0);
    assert(x.empty());

    x.pushArray();
    x.push_back(2);
    x.pushArray();
    x.push_back(3);
    x.push_back(5);
    x.append(x);
    assert(x.numArrays() == 4);
    assert(x.size(0) == 1);
    assert(x.size(1) == 2);
    assert(x.size(2) == 1);
    assert(x.size(3) == 2);
    assert(x(0, 0) == 2);
    assert(x(1, 0) == 3);
    assert(x(1, 1) == 5);
    assert(x(2, 0) == 2);
    assert(x(3, 0) == 3);
    assert(x(3, 1) == 5);

    x.pushArrays(4);
    assert(x.numArrays() == 8);
    for (std::size_t i = 4; i != 8; ++i) {
      assert(x.empty(i));
    }

    x.shrink_to_fit();
    assert(x.capacity() == x.size());
  }
  // Rebuild from a vector of packed arrays.
  {
    container::PackedArrayOfArrays<double> x;
    // No parts.
    {
      std::vector<container::PackedArrayOfArrays<double> > parts;
      x.rebuild(parts);
      assert(x.empty());
      assert(x.numArrays() == 0);
    }
    // Empty parts.
    {
      std::vector<container::PackedArrayOfArrays<double> > parts(7);
      x.rebuild(parts);
      assert(x.empty());
      assert(x.numArrays() == 0);
    }
    // Non-empty parts.
    {
      std::vector<container::PackedArrayOfArrays<double> > parts(3);
      parts[0].pushArray();
      parts[0].push_back(2);
      parts[1].pushArray();
      parts[1].push_back(3);
      parts[2].pushArray();
      parts[2].push_back(5);

      x.rebuild(parts);
      assert(x.size() == 3);
      assert(x.numArrays() == 3);
      assert(x(0, 0) == 2);
      assert(x(1, 0) == 3);
      assert(x(2, 0) == 5);
    }
  }
  {
    // Construct from a container of containers.
    std::vector<std::vector<double> > cc(7);
    cc[1].push_back(1);
    cc[1].push_back(1);
    cc[3].push_back(2);
    cc[3].push_back(3);
    cc[3].push_back(5);
    cc[5].push_back(8);
    cc[5].push_back(13);
    cc[5].push_back(21);
    cc[5].push_back(34);
    cc[5].push_back(55);
    const std::size_t numElements = 10;
    container::PackedArrayOfArrays<double> x(cc);

    assert(x.numArrays() == cc.size());
    assert(x.size() == numElements);
    assert(! x.empty());
    assert(x.max_size() > 0);
    assert(x.begin() + x.size() == x.end());
    assert(x.rbegin() + x.size() == x.rend());

    for (std::size_t n = 0; n != cc.size(); ++n) {
      assert(x.size(n) == cc[n].size());
      assert(x.empty(n) == cc[n].empty());
      assert(x.begin(n) + x.size(n) == x.end(n));
      assert(x.rbegin(n) + x.size(n) == x.rend(n));
    }

    std::vector<double> values(x.begin(), x.end());
    assert(values.size() == x.size());
    assert(std::equal(x.begin(), x.end(), values.begin()));
    assert(std::equal(x.rbegin(), x.rend(), values.rbegin()));

    assert(x(1, 0) == cc[1][0]);
    assert(x(1, 1) == cc[1][1]);

    assert(x(3, 0) == cc[3][0]);
    assert(x(3, 1) == cc[3][1]);
    assert(x(3, 2) == cc[3][2]);

    assert(x(5, 0) == cc[5][0]);
    assert(x(5, 1) == cc[5][1]);
    assert(x(5, 2) == cc[5][2]);
    assert(x(5, 3) == cc[5][3]);
    assert(x(5, 4) == cc[5][4]);
  }
  {
    // Construct from sizes and values.
    const std::size_t numArrays = 7;
    const std::size_t numElements = 10;
    const std::size_t sizes[numArrays] = { 0, 2, 0, 3, 0, 5, 0 };
    const double values[numElements] = { 1, 1,
                                         2, 3, 5,
                                         8, 13, 21, 34, 55
                                       };
    container::PackedArrayOfArrays<double> x(sizes, sizes + numArrays,
        values, values + numElements);

    assert(x.numArrays() == numArrays);
    assert(x.size() == numElements);
    assert(! x.empty());
    assert(x.max_size() > 0);
    assert(x.begin() + x.size() == x.end());

    for (std::size_t n = 0; n != numArrays; ++n) {
      assert(x.size(n) == sizes[n]);
      assert(x.empty(n) == ! sizes[n]);
      assert(x.begin(n) + x.size(n) == x.end(n));
    }

    assert(x(1, 0) == values[0]);
    assert(x(1, 1) == values[1]);

    assert(x(3, 0) == values[2]);
    assert(x(3, 1) == values[3]);
    assert(x(3, 2) == values[4]);

    assert(x(5, 0) == values[5]);
    assert(x(5, 1) == values[6]);
    assert(x(5, 2) == values[7]);
    assert(x(5, 3) == values[8]);
    assert(x(5, 4) == values[9]);

    // Copy constructor.
    {
      container::PackedArrayOfArrays<double> y(x);
      assert(x == y);
    }

    // Swap.
    {
      container::PackedArrayOfArrays<double> y(x);
      container::PackedArrayOfArrays<double> z;
      z.swap(y);
      assert(x == z);
    }

    // Assignment operator
    {
      container::PackedArrayOfArrays<double> y;
      y = x;
      assert(x == y);
    }

    // Build
    {
      container::PackedArrayOfArrays<double> y;
      y.rebuild(sizes, sizes + numArrays, values, values + numElements);
      assert(x == y);
    }

    // Build
    {
      container::PackedArrayOfArrays<double> y;
      y.rebuild(sizes, sizes + numArrays);
      std::copy(values, values + numElements, y.begin());
      assert(x == y);
    }

    //
    // Swap.
    //
    {
      container::PackedArrayOfArrays<double> a(x);
      container::PackedArrayOfArrays<double> b;
      a.swap(b);
      assert(x == b);
    }

    // Ascii I/O.
    {
      std::stringstream s;
      s << x;
      container::PackedArrayOfArrays<double> a;
      s >> a;
      assert(x == a);
    }
  }
  // Push and pop.
  {
    container::PackedArrayOfArrays<double> x;
    assert(x.numArrays() == 0);

    x.pushArray();
    assert(x.numArrays() == 1);
    assert(x.size(0) == 0);

    x.push_back(2);
    assert(x.numArrays() == 1);
    assert(x.size(0) == 1);
    assert(x(0, 0) == 2);
    assert(x[0] == 2);

    x.pop_back();
    assert(x.numArrays() == 1);
    assert(x.size(0) == 0);

    x.popArray();
    assert(x.numArrays() == 0);


    x.pushArray();
    x.pushArray();
    x.push_back(2);
    x.pushArray();
    x.push_back(3);
    x.push_back(5);
    assert(x.numArrays() == 3);
    assert(x.size(0) == 0);
    assert(x.size(1) == 1);
    assert(x.size(2) == 2);

    x.popArray();
    assert(x.numArrays() == 2);
    assert(x.size(0) == 0);
    assert(x.size(1) == 1);

    x.popArray();
    assert(x.numArrays() == 1);
    assert(x.size(0) == 0);

    x.popArray();
    assert(x.numArrays() == 0);
    assert(x.empty());

    std::vector<double> data;
    data.push_back(2);
    data.push_back(3);
    data.push_back(5);
    x.pushArray(data.begin(), data.end());
    assert(x.numArrays() == 1);
    assert(x.size(0) == data.size());
  }

  testTransposeInteger<std::uint8_t>();
  testTransposeInteger<std::uint64_t>();

  testTransposeIntegerOverflow();

  testTransposePair<std::uint8_t>();
  testTransposePair<std::uint64_t>();

  testTransposePairOverflow();

  return 0;
}
