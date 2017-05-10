// -*- C++ -*-

/*!
  \file examples/amr/unbalanced/interface2.cc
  \brief Track a boundary described by a level set in 2-D.
*/

#include <cstddef>

namespace {
//
// Constants.
//

#define DIMENSION 2
static const std::size_t MaximumLevel = 8;
static const bool AutomaticBalancing = false;
}

#define __examples_amr_interface_h__
#include "interface.h"
#undef __examples_amr_interface_h__
