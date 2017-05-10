// -*- C++ -*-

#include "stlib/hj/DistanceAdj1st.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{

  //
  // 2-D
  //
  {
    container::MultiArray<double, 2> solution;
    container::MultiArray<hj::Status, 2> status;
    const double dx = 1;
    hj::DistanceAdj1st<2, double> x(solution, status, dx);
  }

  //
  // 3-D
  //
  {
    container::MultiArray<double, 3> solution;
    container::MultiArray<hj::Status, 3> status;
    const double dx = 1;
    hj::DistanceAdj1st<3, double> x(solution, status, dx);
  }

  return 0;
}
