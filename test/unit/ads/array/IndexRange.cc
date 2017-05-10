// -*- C++ -*-

#include "stlib/ads/array/IndexRange.h"

#include <cassert>

using namespace stlib;

int
main()
{
  //
  // 1-D
  //

  {
    typedef ads::IndexRange<1> IndexRange;
    //
    // Constructor: default
    //
    {
      IndexRange a;
      assert(a.lbound() == 0);
      assert(a.ubound() == 0);
      assert(a.extent() == 0);
      assert(a.empty());
    }
    //
    // Construct from bounds.
    //
    {
      IndexRange a(-1, 7);
      assert(a.lbound() == -1);
      assert(a.ubound() == 7);
      assert(a.extent() == 8);
      assert(! a.empty());
    }
    //
    // Set the bounds.
    //
    {
      IndexRange a;
      a.set_lbound(-1);
      a.set_ubound(7);
      assert(a.lbound() == -1);
      assert(a.ubound() == 7);
      assert(a.extent() == 8);
      assert(! a.empty());
      assert(a.is_in(-1));
      assert(a.is_in(6));
      assert(! a.is_in(-2));
      assert(! a.is_in(7));
    }
    //
    // Equality.
    //
    {
      IndexRange a(2, 3), b(5, 7);
      assert(a == a);
      assert(a != b);
    }
    //
    // Copy constructor.
    //
    {
      IndexRange a(-1, 7);
      IndexRange b(a);
      assert(a == b);
    }
    //
    // Assignment.
    //
    {
      IndexRange a(-1, 7);
      IndexRange b;
      b = a;
      assert(a == b);
    }
  }





  //
  // N-D
  //

  //
  // Constructor: default
  //
  {
    typedef ads::IndexRange<1> IndexRange;
    typedef IndexRange::multi_index_type MI;
    IndexRange a;
    assert(a.dimension() == 1);
    MI x;
    x = 0;
    assert(a.lbounds() == x);
    assert(a.ubounds() == x);
    assert(a.extents() == x);
    assert(a.empty());
  }
  {
    typedef ads::IndexRange<3> IndexRange;
    typedef IndexRange::multi_index_type MI;
    IndexRange a;
    assert(a.dimension() == 3);
    MI x;
    x = 0;
    assert(a.lbounds() == x);
    assert(a.ubounds() == x);
    assert(a.extents() == x);
    assert(a.empty());
  }
  //
  // Construct from bounds.
  //
  {
    typedef ads::IndexRange<3> IndexRange;
    typedef IndexRange::multi_index_type MI;
    MI lb(-2, -3, -5), ub(7, 11, 13);
    IndexRange a(lb, ub);

    assert(a.dimension() == 3);
    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert((a.extents() == ub - lb));
    assert(! a.empty());

    assert(a.is_in(lb));
    assert(a.is_in(MI(0, 0, 0)));
    assert(! a.is_in(ub));
    assert(! a.is_in(MI(7, 0, 0)));
    assert(! a.is_in(MI(0, 11, 0)));
    assert(! a.is_in(MI(0, 0, 13)));

    assert(a.is_in(-2, -3, -5));
    assert(a.is_in(0, 0, 0));
    assert(! a.is_in(7, 11, 13));
    assert(! a.is_in(7, 0, 0));
    assert(! a.is_in(0, 11, 0));
    assert(! a.is_in(0, 0, 13));
  }
  //
  // Construct from number_type bounds.
  //
  {
    typedef ads::IndexRange<3> IndexRange;
    typedef IndexRange::multi_index_type MI;
    MI lb(-2, -3, -5), ub(7, 11, 13);
    IndexRange a(-2, -3, -5, 7, 11, 13);

    assert(a.dimension() == 3);
    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert((a.extents() == ub - lb));
    assert(! a.empty());
  }
  //
  // Set the bounds.
  //
  {
    typedef ads::IndexRange<3> IndexRange;
    typedef IndexRange::multi_index_type MI;
    MI lb(-2, -3, -5), ub(7, 11, 13);
    IndexRange a;

    a.set_lbounds(lb);
    a.set_ubounds(ub);

    assert(a.dimension() == 3);
    assert(a.lbounds() == lb);
    assert(a.ubounds() == ub);
    assert((a.extents() == ub - lb));
    assert(! a.empty());
  }
  //
  // Equality.
  //
  {
    typedef ads::IndexRange<2> IndexRange;
    typedef IndexRange::multi_index_type MI;
    IndexRange
    a(MI(2, 3), MI(5, 7)),
    b(MI(1, 2), MI(3, 4));
    assert(a == a);
    assert(a != b);
  }
  //
  // Copy constructor.
  //
  {
    typedef ads::IndexRange<3> IndexRange;
    typedef IndexRange::multi_index_type MI;
    IndexRange a(MI(1, 2, 3), MI(11, 22, 33));
    IndexRange b(a);
    assert(a == b);
  }
  //
  // Assignment.
  //
  {
    typedef ads::IndexRange<3> IndexRange;
    typedef IndexRange::multi_index_type MI;
    IndexRange a(MI(1, 2, 3), MI(11, 22, 33));
    IndexRange b;
    b = a;
    assert(a == b);
  }
  //
  // File I/O.
  //
  {
    typedef ads::IndexRange<3> IndexRange;
    typedef IndexRange::multi_index_type MI;
    IndexRange a(MI(1, 2, 3), MI(11, 22, 33));
    std::ostringstream out;
    out << a;
    std::istringstream in(out.str());
    IndexRange b;
    in >> b;
    assert(a == b);
  }
  {
    typedef ads::IndexRange<3> IndexRange;
    typedef IndexRange::multi_index_type MI;
    IndexRange a(MI(0, 0, 0), MI(11, 22, 33));
    std::istringstream in("11 22 33");
    IndexRange b;
    in >> b;
    assert(a == b);
  }

  return 0;
}
