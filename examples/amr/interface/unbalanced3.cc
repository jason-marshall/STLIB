// -*- C++ -*-

/*!
  \file examples/amr/unbalanced/interface3.cc
  \brief Track a boundary described by a level set in 3-D.
*/

#include <cstddef>

namespace {
//
// Constants.
//

#define DIMENSION 3
static const std::size_t MaximumLevel = 6;
static const bool AutomaticBalancing = false;
}

#define __examples_amr_interface_h__
#include "interface.h"
#undef __examples_amr_interface_h__
