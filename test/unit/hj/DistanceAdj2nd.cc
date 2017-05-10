// -*- C++ -*-

#include "stlib/hj/DistanceAdj2nd.h"

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
    hj::DistanceAdj2nd<2, double> x(solution, status, dx);
  }

  //
  // 3-D
  //
  {
    /*
    ads::Array<3,double> solution;
    ads::Array<3,hj::Status> status;
    const double dx = 1;
    hj::DistanceAdj2nd<3,double> x(solution, status, dx);
    */
  }

  return 0;
}
