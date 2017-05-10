// -*- C++ -*-

#include "stlib/hj/DiffSchemeAdj.h"

#include "stlib/hj/DistanceAdj1st.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

int
main()
{

  //
  // 2-D
  //
  {
    typedef container::MultiArray<double, 2> MultiArray;
    typedef MultiArray::SizeList SizeList;
    typedef MultiArray::IndexList IndexList;
    typedef MultiArray::size_type size_type;
    typedef MultiArray::Index Index;
    typedef MultiArray::Range Range;

    const SizeList extents = {{10, 10}};
    const IndexList bases = {{0, 0}};
    const Range range(extents, bases);
    MultiArray solution(extents + size_type(2), bases - Index(1));
    const double dx = 1;
    const bool is_concurrent = false;
    typedef hj::DistanceAdj1st<2, double> Equation;
    hj::DiffSchemeAdj<2, double, Equation>
    x(range, solution, dx, is_concurrent);
  }
  //
  // 3-D
  //
  {
    typedef container::MultiArray<double, 3> MultiArray;
    typedef MultiArray::SizeList SizeList;
    typedef MultiArray::IndexList IndexList;
    typedef MultiArray::size_type size_type;
    typedef MultiArray::Index Index;
    typedef MultiArray::Range Range;

    const SizeList extents = {{10, 10, 10}};
    const IndexList bases = {{0, 0, 0}};
    const Range range(extents, bases);
    MultiArray solution(extents + size_type(2), bases - Index(1));
    const double dx = 1;
    const bool is_concurrent = false;
    typedef hj::DistanceAdj1st<3, double> Equation;
    hj::DiffSchemeAdj<3, double, Equation>
    x(range, solution, dx, is_concurrent);
  }

  return 0;
}
