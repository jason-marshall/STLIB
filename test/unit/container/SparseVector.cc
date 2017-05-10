// -*- C++ -*-

#include "stlib/container/SparseVector.h"

#include <iostream>

#include <cassert>
#include <sstream>

using namespace stlib;

int
main()
{
  typedef container::SparseVector<double> SparseVector;
  typedef SparseVector::value_type value_type;
  typedef SparseVector::iterator iterator;
  typedef SparseVector::const_iterator const_iterator;
  {
    // Vector constructor.
    {
      std::vector<value_type> data;
      data.push_back(value_type(3, 1.));
      data.push_back(value_type(2, 0.));
      SparseVector x(data);
      assert(x.isValid());
      SparseVector y;
      y.rebuild(data);
      assert(y.isValid());
    }

    // Default constructor.
    SparseVector x;
    assert(x.size() == 0);
    assert(x.empty());
    assert(x.max_size() > 0);
    assert(x.begin() == x.end());
    assert(x.rbegin() == x.rend());
    assert(x.keyBegin() == x.keyEnd());
    assert(x.mappedBegin() == x.mappedEnd());
    assert(x.find(0) == x.end());
    assert(x.count(0) == 0);
    assert(x.lower_bound(0) == x.end());
    assert(x.upper_bound(0) == x.end());

    // Insert.
    {
      std::pair<iterator, bool> result = x.insert(value_type(2, 4));
      assert(result.first != x.end());
      assert(result.first->first == 2);
      assert(result.first->second == 4);
      assert(result.second);
      assert(x.isValid());
      assert(x.size() == 1);
    }
    // Append
    {
      x.append(3, 9);
      assert(x.isValid());
      assert(x.size() == 2);
    }

    // Index.
    assert(x[2] == 4 && x[3] == 9);
    x[5] = 25;
    assert(x.size() == 3);
    assert(x.isValid());
    assert(x[5] == 25);

    // Copy constructor.
    {
      SparseVector y(x);
      assert(x == y);
    }

    // Assignment operator.
    {
      SparseVector y;
      y = x;
      assert(x == y);
    }

    // Find.
    {
      iterator i = x.find(2);
      assert(i->first == 2 && i->second == 4);
      i = x.find(4);
      assert(i == x.end());
    }
    {
      const SparseVector& y = x;
      const_iterator i = y.find(2);
      assert(i->first == 2 && i->second == 4);
      i = y.find(4);
      assert(i == y.end());
    }

    // Lower bound.
    {
      iterator i = x.lower_bound(0);
      assert(i != x.end());
      assert(i->first == 2 && i->second == 4);

      i = x.lower_bound(2);
      assert(i != x.end());
      assert(i->first == 2 && i->second == 4);

      i = x.lower_bound(4);
      assert(i != x.end());
      assert(i->first == 5 && i->second == 25);

      i = x.lower_bound(6);
      assert(i == x.end());
    }
    {
      const SparseVector& y = x;
      const_iterator i = y.lower_bound(0);
      assert(i != y.end());
      assert(i->first == 2 && i->second == 4);

      i = y.lower_bound(2);
      assert(i != y.end());
      assert(i->first == 2 && i->second == 4);

      i = y.lower_bound(4);
      assert(i != y.end());
      assert(i->first == 5 && i->second == 25);

      i = y.lower_bound(6);
      assert(i == y.end());
    }

    // Upper bound.
    {
      iterator i = x.upper_bound(0);
      assert(i != x.end());
      assert(i->first == 2 && i->second == 4);

      i = x.upper_bound(2);
      assert(i != x.end());
      assert(i->first == 3 && i->second == 9);

      i = x.upper_bound(4);
      assert(i != x.end());
      assert(i->first == 5 && i->second == 25);

      i = x.upper_bound(6);
      assert(i == x.end());
    }
    {
      const SparseVector& y = x;
      const_iterator i = y.upper_bound(0);
      assert(i != y.end());
      assert(i->first == 2 && i->second == 4);

      i = y.upper_bound(2);
      assert(i != y.end());
      assert(i->first == 3 && i->second == 9);

      i = y.upper_bound(4);
      assert(i != y.end());
      assert(i->first == 5 && i->second == 25);

      i = y.upper_bound(6);
      assert(i == y.end());
    }

    // Swap.
    {
      SparseVector y;
      x.swap(y);
      assert(x.size() == 0 && y.size() == 3);
      x.swap(y);
      assert(x.size() == 3 && y.size() == 0);
    }

    // File I/O.
    {
      std::ostringstream out;
      out << x;
      SparseVector y;
      std::istringstream in(out.str());
      in >> y;
      assert(x == y);
    }

    // Mathematical functions.
    assert(sum(x) == 4 + 9 + 25);
    assert(product(x) == 4 * 9 * 25);
    assert(min(x) == 4);
    assert(max(x) == 25);

    // Operations with vectors and sparse vectors.
    {
      std::vector<double> v(7, 0.);
      v += x;
      assert(v[0] == 0 && v[1] == 0 && v[2] == 4 && v[3] == 9 && v[4] == 0 &&
             v[5] == 25 && v[6] == 0);
    }
    {
      std::vector<double> v(7, 0.);
      v -= x;
      assert(v[0] == 0 && v[1] == 0 && v[2] == -4 && v[3] == -9 && v[4] == 0 &&
             v[5] == -25 && v[6] == 0);
    }
    {
      std::vector<double> v(7, 1.);
      v *= x;
      assert(v[0] == 1 && v[1] == 1 && v[2] == 4 && v[3] == 9 && v[4] == 1 &&
             v[5] == 25 && v[6] == 1);
    }
    {
      std::vector<double> v(7, 0.);
      scaleAdd(&v, 2, x);
      assert(v[0] == 0 && v[1] == 0 && v[2] == 8 && v[3] == 18 && v[4] == 0 &&
             v[5] == 50 && v[6] == 0);
    }

    // Operations with two sparse vectors.
    {
      SparseVector y;
      y.append(1, 2);
      y.append(2, 4);
      y.append(3, 6);
      y.append(4, 8);
      // +
      {
        SparseVector z;
        z.append(1, 0 + 2);
        z.append(2, 4 + 4);
        z.append(3, 9 + 6);
        z.append(4, 0 + 8);
        z.append(5, 25 + 0);
        assert(z == x + y);
      }
      // -
      {
        SparseVector z;
        z.append(1, 0 - 2);
        //z.append(2, 4 - 4);
        z.append(3, 9 - 6);
        z.append(4, 0 - 8);
        z.append(5, 25 - 0);
        assert(z == x - y);
      }
      // *
      {
        SparseVector z;
        //z.append(1, 0 * 2);
        z.append(2, 4 * 4);
        z.append(3, 9 * 6);
        //z.append(4, 0 * 8);
        //z.append(5, 25 * 0);
        assert(z == x * y);
      }
    }

    // Erase.
    x.erase(x.begin());
    assert(x.size() == 2);
    assert(x.isValid());
    assert(x[3] == 9 && x[5] == 25);

    assert(x.erase(3));
    assert(x.size() == 1);
    assert(x.isValid());
    assert(x[5] == 25);

    assert(! x.erase(4));
  }
  return 0;
}
