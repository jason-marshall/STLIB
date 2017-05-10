// -*- C++ -*-

/*!
  \file examples/geom/spatialIndexing/interface2.cc
  \brief Track a boundary described by a level set in 2-D.
*/

namespace {
//
// Constants.
//

static const int Dimension = 2;
static const int MaximumLevel = 8;
static const bool AutomaticBalancing = true;
}

#define __examples_geom_spatialIndexing_interface_ipp__
#include "../interface.ipp"
#undef __examples_geom_spatialIndexing_interface_ipp__
