// -*- C++ -*-

#include "stlib/hj/GridMCC.h"

#include "stlib/hj/DistanceAdjDiag1st.h"
#include "stlib/hj/DiffSchemeAdjDiag.h"

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

    const SizeList extents = {{10, 10}};
    const IndexList bases = {{0, 0}};
    MultiArray solution(extents + size_type(2), bases - Index(1));
    const double dx = 1;
    typedef hj::DistanceAdjDiag1st<2, double> Equation;
    typedef hj::DiffSchemeAdjDiag<2, double, Equation> DifferenceScheme;
    hj::GridMCC<2, double, DifferenceScheme> x(solution, dx);
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

    const SizeList extents = {{10, 10, 10}};
    const IndexList bases = {{0, 0, 0}};
    MultiArray solution(extents + size_type(2), bases - Index(1));
    const double dx = 1;
    typedef hj::DistanceAdjDiag1st<3, double> Equation;
    typedef hj::DiffSchemeAdjDiag<3, double, Equation> DifferenceScheme;
    hj::GridMCC<3, double, DifferenceScheme> x(solution, dx);
  }

  return 0;
}
