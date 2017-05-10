// -*- C++ -*-

/*!
  \file
  \brief Minimum enclosing ball in 3-D.
*/

// Adapted from code in the WildMagic 4 library.

// Geometric Tools, LLC
// Copyright (c) 1998-2010
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
// File Version: 4.10.0 (2009/11/18)

#if !defined(__geom_kernel_MinSphere3_h__)
#define __geom_kernel_MinSphere3_h__

#include "stlib/geom/kernel/Ball.h"

namespace stlib
{
namespace geom
{

//! Compute the minimum volume sphere containing the input set of points.
/*! The algorithm randomly permutes the input points so that the construction
  occurs in 'expected' O(N) time. */
template<typename _T>
class MinBall3
{
  //
  // Types.
  //
public:

  //! A Cartesian point.
  typedef std::array<_T, 3> Point;

private:

  // indices of points that support current minimum volume sphere
  class Support
  {
  public:
    bool
    Contains(int iIndex, Point** apkPoint, _T epsilon)
    {
      for (std::size_t i = 0; i < Quantity; ++i) {
        Point kDiff = *apkPoint[iIndex] - *apkPoint[Index[i]];
        if (kDiff.SquaredLength() < epsilon) {
          return true;
        }
      }
      return false;
    }

    int Quantity;
    int Index[4];
  };

  typedef Ball<_T, 3> (MinBall3<_T>::*UpdateFunction)
  (int, Point**, Support&);

  //
  // Member data.
  //
private:

  _T _epsilon;
  UpdateFunction m_aoUpdate[5];

public:

  // The epsilon value is a floating-point tolerance used for various
  // computations.
  MinBall3(int iQuantity, const Point* akPoint,
           Ball<_T, 3>& rkMinimal, _T epsilon = (_T)1.0e-03);

private:

  // test if point P is inside sphere S
  bool
  Contains(const Point& rkP, const Ball<_T, 3>& rkS,
           _T& rfDistDiff);

  Ball<_T, 3>
  ExactSphere1(const Point& rkP);

  Ball<_T, 3>
  ExactSphere2(const Point& rkP0, const Point& rkP1);

  Ball<_T, 3>
  ExactSphere3(const Point& rkP0, const Point& rkP1,
               const Point& rkP2);

  Ball<_T, 3>
  ExactSphere4(const Point& rkP0, const Point& rkP1,
               const Point& rkP2, const Point& rkP3);

  Ball<_T, 3>
  UpdateSupport1(int i, Point** apkPerm, Support& rkSupp);

  Ball<_T, 3>
  UpdateSupport2(int i, Point** apkPerm, Support& rkSupp);

  Ball<_T, 3>
  UpdateSupport3(int i, Point** apkPerm, Support& rkSupp);

  Ball<_T, 3>
  UpdateSupport4(int i, Point** apkPerm, Support& rkSupp);

};

} // namespace geom
}

#define __geom_kernel_MinBall3_ipp__
#include "stlib/geom/kernel/MinBall3.ipp"
#undef __geom_kernel_MinBall3_ipp__

#endif
