// -*- C++ -*-

/*!
  \file stochastic/InhomogeneousTimeSeriesUniformDirect.h
  \brief Collect time series data with the direct method on time inhomogeneous problems.
*/

#if !defined(__stochastic_InhomogeneousTimeSeriesUniformDirect_h__)
#define __stochastic_InhomogeneousTimeSeriesUniformDirect_h__

#include "stlib/stochastic/InhomogeneousDirect.h"

namespace stlib
{
namespace stochastic
{

//! Collect time series data with the direct method on time inhomogeneous problems.
/*!
  This class just renames InhomogeneousDirect. This base class has all of the
  necessary functionality.
*/
class InhomogeneousTimeSeriesUniformDirect :
  public InhomogeneousDirect
{
  //
  // Private types.
  //
private:

  typedef InhomogeneousDirect Base;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  InhomogeneousTimeSeriesUniformDirect();
  //! Copy constructor not implemented.
  InhomogeneousTimeSeriesUniformDirect
  (const InhomogeneousTimeSeriesUniformDirect&);
  //! Assignment operator not implemented.
  InhomogeneousTimeSeriesUniformDirect&
  operator=(const InhomogeneousTimeSeriesUniformDirect&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  InhomogeneousTimeSeriesUniformDirect(const State& state,
                                       const ReactionSet& reactionSet,
                                       const double maxSteps) :
    Base(state, reactionSet, maxSteps)
  {
  }

  // Default destructor is fine.

  //@}
};

//@}

} // namespace stochastic
}

#endif
