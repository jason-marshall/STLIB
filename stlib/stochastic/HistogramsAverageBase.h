// -*- C++ -*-

/*!
  \file stochastic/HistogramsAverageBase.h
  \brief Collect the average values of species populations.
*/

#if !defined(__stochastic_HistogramsAverageBase_h__)
#define __stochastic_HistogramsAverageBase_h__

#include "stlib/stochastic/Direct.h"
#include "stlib/stochastic/HistogramsAveragePackedArray.h"

namespace stlib
{
namespace stochastic
{

//! Accumulate a histogram of equilibrium state using Gillespie's direct method.
/*!
  \param _DiscreteGenerator Random deviate generator for the discrete,
  finite distribution with reaction propensities as scaled probabilities.
  \param _ExponentialGenerator Random deviate generator for the exponential
  distribution. By default the ziggurat algorithm is used.
  \param _PropensitiesFunctor Can calculate propensities as a function of the
  reaction index and the populations.
*/
template < class _DiscreteGenerator,
           class _ExponentialGenerator =
           numerical::ExponentialGeneratorZiggurat<>,
           class _PropensitiesFunctor = PropensitiesSingle<true> >
class HistogramsAverageBase :
  public Direct < _DiscreteGenerator, _ExponentialGenerator,
  _PropensitiesFunctor >
{
  //
  // Private types.
  //
private:

  typedef Direct < _DiscreteGenerator, _ExponentialGenerator,
          _PropensitiesFunctor > Base;

  //
  // Public types.
  //
public:

  //! The propensities functor.
  typedef typename Base::PropensitiesFunctor PropensitiesFunctor;
  //! The exponential generator.
  typedef typename Base::ExponentialGenerator ExponentialGenerator;
  //! The discrete, finite generator.
  typedef typename Base::DiscreteGenerator DiscreteGenerator;
  //! The discrete, uniform generator.
  typedef typename Base::DiscreteUniformGenerator DiscreteUniformGenerator;

  //
  // Member data.
  //
protected:

  using Base::_state;
  using Base::_error;
  using Base::_time;
  using Base::_tau;
  using Base::_exponentialGenerator;
  using Base::_discreteGenerator;

  //! The species to record.
  std::vector<std::size_t> _recordedSpecies;
  //! Histograms for the recorded species.
  HistogramsAveragePackedArray _histograms;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HistogramsAverageBase();
  //! Copy constructor not implemented.
  HistogramsAverageBase(const HistogramsAverageBase&);
  //! Assignment operator not implemented.
  HistogramsAverageBase&
  operator=(const HistogramsAverageBase&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  HistogramsAverageBase
  (const State& state,
   const PropensitiesFunctor& propensitiesFunctor,
   const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins,
   const std::size_t multiplicity,
   const double maxSteps) :
    Base(state, propensitiesFunctor, reactionInfluence, maxSteps),
    _recordedSpecies(recordedSpecies),
    _histograms(recordedSpecies.size(), numberOfBins, multiplicity)
  {
  }

  // Use the default destructor.

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Let the process equilibrate.
  using Base::equilibrate;

  //! Initialize the state with the initial populations and time.
  void
  initialize(const std::vector<double>& populations, const double startTime)
  {
    // Initialize the state.
    Base::initialize(populations, startTime);
    // Prepare for recording a trajectory.
    _histograms.initialize();
  }

  //! Synchronize the two sets of histograms so that corresponding historams have the same lower bounds and widths.
  void
  synchronize()
  {
    _histograms.synchronize();
  }

protected:

  //! Record a step count error message. Record the current time.
  using Base::setStepCountError;

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return a const reference to the state.
  using Base::getState;

  //! Return a const reference to the discrete, uniform generator.
  using Base::getDiscreteUniformGenerator;

  //! Return the vector of recorded species.
  const std::vector<std::size_t>&
  getRecordedSpecies() const
  {
    return _recordedSpecies;
  }

  //! Return the set of histograms.
  const HistogramsAveragePackedArray&
  getHistograms() const
  {
    return _histograms;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  //! Compute the time to the next reaction.
  using Base::computeTau;

  //@}
};

//@}

} // namespace stochastic
}

#endif
