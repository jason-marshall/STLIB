// -*- C++ -*-

/*!
  \file stochastic/HistogramsAveragePacked.h
  \brief Packed histograms for recording the average populations.
*/

#if !defined(__stochastic_HistogramsAveragePacked_h__)
#define __stochastic_HistogramsAveragePacked_h__

#include "stlib/stochastic/HistogramReference.h"

namespace stlib
{
namespace stochastic
{

//! A packed set of histograms.
/*!
  For weighted histograms, the sum of the weighted probabilities is the number
  of events.
*/
class HistogramsAveragePacked
{
private:

  //
  // Member data.
  //

  //! The number of species.
  std::size_t _numberOfSpecies;
  //! The number of trajectories recorded.
  double _numberOfTrajectories;
  //! The histograms.
  std::vector<HistogramReference> _histograms;
  //! The data for the bins.
  std::vector<double> _bins;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor. Invalid data.
  HistogramsAveragePacked() :
    _numberOfSpecies(),
    _numberOfTrajectories(),
    _histograms(),
    _bins()
  {
  }

  //! Construct from the number of species and the number of bins.
  HistogramsAveragePacked(const std::size_t numberOfSpecies,
                          const std::size_t numberOfBins) :
    _numberOfSpecies(numberOfSpecies),
    _numberOfTrajectories(0),
    _histograms(numberOfSpecies),
    _bins(numberOfSpecies* numberOfBins, 0.)
  {
    for (std::size_t i = 0; i != _histograms.size(); ++i) {
      _histograms[i].initialize(numberOfBins, &_bins[0] + i * numberOfBins);
    }
  }

  //! Copy constructor.
  HistogramsAveragePacked(const HistogramsAveragePacked& other) :
    _numberOfSpecies(other._numberOfSpecies),
    _numberOfTrajectories(other._numberOfTrajectories),
    _histograms(other._histograms),
    _bins(other._bins.size())
  {
    if (! _histograms.empty()) {
      const std::size_t numberOfBins = _histograms[0].size();
      double* data = &_bins[0];
      for (std::size_t i = 0; i != _histograms.size();
           ++i, data += numberOfBins) {
        _histograms[i].initialize(numberOfBins, data);
      }
      std::copy(other._bins.begin(), other._bins.end(), _bins.begin());
    }
  }

  //! Assignment operator.
  HistogramsAveragePacked&
  operator=(const HistogramsAveragePacked& other)
  {
    if (&other != this) {
      _numberOfSpecies = other._numberOfSpecies;
      _numberOfTrajectories = other._numberOfTrajectories;
      _histograms = other._histograms;
      _bins = other._bins;
      // Copy the bins and update the memory locations.
      const std::size_t numberOfBins = _histograms[0].size();
      double* data = &_bins[0];
      for (std::size_t i = 0; i != _histograms.size();
           ++i, data += numberOfBins) {
        _histograms[i].copyBins(data);
      }
    }
    return *this;
  }

  // Use the default destructor.

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the number of species.
  std::size_t
  numberOfSpecies() const
  {
    return _numberOfSpecies;
  }

  //! Get the number of trajectories recorded.
  double
  getNumberOfTrajectories() const
  {
    return _numberOfTrajectories;
  }

  //! Return the specified histogram.
  const HistogramReference&
  operator()(const std::size_t species) const
  {
    return _histograms[species];
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Initialize from the number of frames, the number of species, and the number of bins.
  void
  initialize(const std::size_t numberOfSpecies,
             const std::size_t numberOfBins)
  {
    _numberOfSpecies = numberOfSpecies;
    _numberOfTrajectories = 0;
    _histograms.resize(numberOfSpecies);
    _bins.resize(numberOfSpecies * numberOfBins);
    std::fill(_bins.begin(), _bins.end(), 0);
    double* data = &_bins[0];
    for (std::size_t i = 0; i != _histograms.size();
         ++i, data += numberOfBins) {
      _histograms[i].initialize(numberOfBins, data);
    }
  }

  //! Increment the number of trajectories recorded.
  void
  incrementNumberOfTrajectories()
  {
    ++_numberOfTrajectories;
  }

  //! Return the specified histogram.
  HistogramReference&
  operator()(const std::size_t species)
  {
    return _histograms[species];
  }

  //@}
};

//! Write the histograms in ascii format.
/*!
  \relates HistogramsAveragePacked

  For each species, print the histogram.
*/
inline
std::ostream&
operator<<(std::ostream& out, const HistogramsAveragePacked& x)
{
  for (std::size_t species = 0; species != x.numberOfSpecies(); ++species) {
    out << x(species);
  }
  return out;
}

} // namespace stochastic
}

#endif
