// -*- C++ -*-

/*!
  \file examples/geom/spatialIndexing/interface1.cc
  \brief Track a boundary described by a level set in 1-D.
*/

namespace {
//
// Constants.
//

static const int Dimension = 1;
static const int MaximumLevel = 32;
static const bool AutomaticBalancing = false;
}

#define __examples_geom_spatialIndexing_interface_ipp__
#include "../interface.ipp"
#undef __examples_geom_spatialIndexing_interface_ipp__
