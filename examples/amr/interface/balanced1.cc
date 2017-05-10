// -*- C++ -*-
/*!
  \file examples/amr/balanced/interface1.cc
  \brief Track a boundary described by a level set in 1-D.
*/

#include <cstddef>

namespace {
//
// Constants.
//

#define DIMENSION 1
static const std::size_t MaximumLevel = 32;
static const bool AutomaticBalancing = true;
}

#define __examples_amr_interface_h__
#include "interface.h"
#undef __examples_amr_interface_h__
