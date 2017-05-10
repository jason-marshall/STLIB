// -*- C++ -*-

/*!
  \file State.h
  \brief Class for controlling the state of the CPT.
*/

#if !defined(__cpt_State_h__)
#define __cpt_State_h__

#include "stlib/cpt/StateBase.h"

#include "stlib/container/MultiArray.h"

namespace stlib
{
namespace cpt
{

// Hold the state for a closest point transform.
template < std::size_t N, typename T = double >
class State;

} // namespace cpt
}

#define __cpt_State1_ipp__
#include "stlib/cpt/State1.ipp"
#undef __cpt_State1_ipp__

#define __cpt_State2_ipp__
#include "stlib/cpt/State2.ipp"
#undef __cpt_State2_ipp__

#define __cpt_State3_ipp__
#include "stlib/cpt/State3.ipp"
#undef __cpt_State3_ipp__

#endif
