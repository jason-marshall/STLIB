// -*- C++ -*-

/*!
  \file stochastic/HistogramsBase.h
  \brief Base class for solvers using the direct method to record output in histograms.
*/

#if !defined(__stochastic_HistogramsBase_h__)
#define __stochastic_HistogramsBase_h__

#include "stlib/stochastic/Direct.h"
#include "stlib/stochastic/HistogramsPackedArray.h"

namespace stlib
{
namespace stochastic
{

//! Base class for solvers using the direct method to record output in histograms.
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
class HistogramsBase :
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
  using Base::_time;
  using Base::_discreteUniformGenerator;
  using Base::_exponentialGenerator;
  using Base::_discreteGenerator;

  //! The times at which to accumulate histograms of the state.
  std::vector<double> _frames;
  //! The species to record.
  std::vector<std::size_t> _recordedSpecies;
  //! Histograms for the recorded species.
  HistogramsPackedArray _histograms;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HistogramsBase();
  //! Copy constructor not implemented.
  HistogramsBase(const HistogramsBase&);
  //! Assignment operator not implemented.
  HistogramsBase&
  operator=(const HistogramsBase&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  HistogramsBase
  (const State& state,
   const PropensitiesFunctor& propensitiesFunctor,
   const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
   const std::vector<double>& frames,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins, const std::size_t multiplicity,
   const double maxSteps) :
    Base(state, propensitiesFunctor, reactionInfluence, maxSteps),
    _frames(frames),
    _recordedSpecies(recordedSpecies),
    _histograms(frames.size(), recordedSpecies.size(), numberOfBins,
                multiplicity)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Initialize the state with the initial populations and time.
  void
  initialize(const std::vector<double>& populations, const double time)
  {
    // Initialize the state.
    Base::initialize(populations, time);
    // Prepare for recording a trajectory.
    _histograms.initialize();
  }

  //! Synchronize the two sets of histograms so that corresponding historams have the same lower bounds and widths.
  void
  synchronize()
  {
    _histograms.synchronize();
  }

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
  const HistogramsPackedArray&
  getHistograms() const
  {
    return _histograms;
  }

  //@}
};

//@}

} // namespace stochastic
}

#endif
