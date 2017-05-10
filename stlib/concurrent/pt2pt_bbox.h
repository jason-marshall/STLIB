// -*- C++ -*-

/*!
  \file concurrent/pt2pt_bbox.h
  \brief Point-to-Point Communications Using Bounding Boxes.
*/

/*!
  \page concurrent_pt2pt_bbox Point-to-Point Communications Using Bounding Boxes

  This package is used to determine the communication pattern to implement
  short-range interactions where the data is distributed over a number
  of processors.  In the Virtual Test Facility (VTF) it is used in the
  Eulerian-Lagrangian coupling algorithm.

  This is a templated class library.
  Thus there is no library to compile or link with.  Just include the
  appropriate header files in your application code when you compile.

  The communication classes are classified according to whether the
  communication occurs within a single group of processors (1Grp) or occurs
  between two groups of processors (2Grp).  They are further classified
  according to whether the processors have a single domain (1Dom)
  or have a data domain and an interest domain (2Dom).

  - concurrent::PtToPt1Grp1Dom
  - concurrent::PtToPt1Grp2Dom
  - concurrent::PtToPt2Grp1Dom
  - concurrent::PtToPt2Grp2Dom
*/

#if !defined(__concurrent_pt2pt_bbox_h__)
#define __concurrent_pt2pt_bbox_h__

#include "stlib/concurrent/pt2pt_bbox/PtToPt1Grp1Dom.h"
#include "stlib/concurrent/pt2pt_bbox/PtToPt1Grp2Dom.h"
#include "stlib/concurrent/pt2pt_bbox/PtToPt2Grp1Dom.h"
#include "stlib/concurrent/pt2pt_bbox/PtToPt2Grp2Dom.h"

#endif
