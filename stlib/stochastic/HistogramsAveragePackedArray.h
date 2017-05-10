// -*- C++ -*-

/*!
  \file stochastic/HistogramsAveragePackedArray.h
  \brief An array of sets of packed histograms.
*/

#if !defined(__stochastic_HistogramsAveragePackedArray_h__)
#define __stochastic_HistogramsAveragePackedArray_h__

#include "stlib/stochastic/HistogramsAveragePacked.h"

#include <limits>

namespace stlib
{
namespace stochastic
{

//! An array of sets of packed histograms.
/*!
  Call initialize before recording each trajectory.
*/
class HistogramsAveragePackedArray
{
  //
  // Member data.
  //
private:

  // The array of sets of histograms.
  std::vector<HistogramsAveragePacked> _histograms;
  // The current set of histograms.
  std::vector<HistogramsAveragePacked>::iterator _current;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HistogramsAveragePackedArray();
  //! Copy constructor not implemented.
  HistogramsAveragePackedArray(const HistogramsAveragePackedArray&);
  //! Assignment operator not implemented.
  HistogramsAveragePackedArray&
  operator=(const HistogramsAveragePackedArray&);


  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the number of species and the number of bins.
  HistogramsAveragePackedArray(const std::size_t numberOfSpecies,
                               const std::size_t numberOfBins,
                               const std::size_t multiplicity) :
    _histograms(multiplicity),
    _current(_histograms.begin())
  {
    // Initialize the histograms.
    for (std::size_t i = 0; i != _histograms.size(); ++i) {
      _histograms[i].initialize(numberOfSpecies, numberOfBins);
    }
  }

  // Use the default destructor.

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the specified histogram.
  const HistogramReference&
  operator()(const std::size_t species) const
  {
    return (*_current)(species);
  }

  //! Return the multiplicity for the histograms.
  std::size_t
  multiplicity() const
  {
    return _histograms.size();
  }

  //! Return the specified set of packed histograms.
  const HistogramsAveragePacked&
  get(const std::size_t i) const
  {
    return _histograms[i];
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Prepare for recording a trajectory by selecting a set of histograms and incrementing the sum.
  void
  initialize()
  {
    // Loop over the elements and select the one with the fewest trajectories.
    _current = _histograms.begin();
    for (std::vector<HistogramsAveragePacked>::iterator i = _histograms.begin() + 1;
         i != _histograms.end(); ++i) {
      if (i->getNumberOfTrajectories() <
          _current->getNumberOfTrajectories()) {
        _current = i;
      }
    }
    _current->incrementNumberOfTrajectories();
  }

  //! Synchronize the two sets of histograms so that corresponding historams have the same lower bounds and widths.
  void
  synchronize()
  {
    // The number of bins in each histogram.
    const std::size_t numberOfBins = _histograms[0](0).size();
    double lower, upper, width;
    std::vector<double> sums(_histograms.size());
    // For each species.
    for (std::size_t species = 0;
         species != _histograms[0].numberOfSpecies(); ++species) {
      // There are a number (the multiplicity) of histograms that record
      // this probability distribution.
      // First compute the sums of the histograms.
      for (std::size_t i = 0; i != _histograms.size(); ++i) {
        sums[i] = _histograms[i](species).computeSum();
      }
      // Determine a closed lower bound and an open upper bound for
      // all of the events.
      // Check the trivial case that no events have been recorded.
      if (std::accumulate(sums.begin(), sums.end(), 0.) == 0.) {
        lower = 0.;
        upper = 1.;
        width = 1.;
      }
      else {
        // Standard case.
        // Compute lower and upper bounds.
        lower = std::numeric_limits<double>::max();
        upper = 0.;
        for (std::size_t i = 0; i != _histograms.size(); ++i) {
          if (sums[i] != 0.) {
            const HistogramReference& h =
              _histograms[i](species);
            lower = std::min(lower, h.computeMinimumNonzero());
            upper = std::max(upper, h.computeMaximumNonzero());
          }
        }
        assert(lower < upper);
        // Calculate an appropriate width.
        width = 1.;
        while (lower + numberOfBins * width < upper) {
          width *= 2;
        }
      }
      // Set the lower bounds and widths to the same value in each
      // histogram.
      for (std::size_t i = 0; i != _histograms.size(); ++i) {
        _histograms[i](species).setLowerBoundAndWidth(lower,
            width);
      }
    }
#ifdef STLIB_DEBUG
    assert(areSynchronized());
#endif
  }

  //! Return the specified histogram.
  HistogramReference&
  operator()(const std::size_t species)
  {
    return (*_current)(species);
  }

private:

  //! Return true if the histograms are synchronized.
  bool
  areSynchronized()
  {
    // Use the first histogram as the reference.
    HistogramsAveragePacked& h = _histograms[0];
    for (std::size_t i = 1; i != _histograms.size(); ++i) {
      for (std::size_t species = 0; species != h.numberOfSpecies();
           ++species) {
        if (! stochastic::areSynchronized(h(species),
                                          _histograms[i](species))) {
          return false;
        }
      }
    }
    return true;
  }

  //@}
};

//! Write the two sets of histograms in ascii format.
/*!
  \relates HistogramsAveragePackedArray

  \note You must call synchronize() on \c x before calling this function.

  For each species: write a histogram with two bin arrays.
*/
inline
std::ostream&
operator<<(std::ostream& out, const HistogramsAveragePackedArray& x)
{
  // Write the histogram multiplicity.
  out << x.multiplicity() << '\n';
  for (std::size_t species = 0; species != x.get(0).numberOfSpecies();
       ++species) {
    // Write the lower bound, width, and the bins for the first histogram.
    const HistogramReference& h0 = x.get(0)(species);
    out << h0;
    // Write the bins for the rest of the histograms.
    for (std::size_t i = 1; i != x.multiplicity(); ++i) {
      const HistogramReference& h = x.get(i)(species);
      assert(h.getLowerBound() == h0.getLowerBound() &&
             h.getWidth() == h0.getWidth());
      std::copy(h.begin(), h.end(),
                std::ostream_iterator<double>(out, " "));
      out << '\n';
    }
  }
  return out;
}

} // namespace stochastic
}

#endif
