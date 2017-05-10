// -*- C++ -*-

/*!
  \file examples/geom/spatialIndexing/interface4.cc
  \brief Track a boundary described by a level set in 4-D.
*/

namespace {
//
// Constants.
//

static const int Dimension = 4;
static const int MaximumLevel = 4;
static const bool AutomaticBalancing = true;
}

#define __examples_geom_spatialIndexing_interface_ipp__
#include "../interface.ipp"
#undef __examples_geom_spatialIndexing_interface_ipp__
