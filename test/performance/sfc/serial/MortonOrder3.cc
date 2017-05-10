// -*- C++ -*-

#include "stlib/sfc/MortonOrder.h"

using namespace stlib;

std::size_t const Dimension = 3;
typedef std::uint64_t Code;
typedef sfc::MortonOrder<Dimension, Code> Order;

#include "Order3.h"
