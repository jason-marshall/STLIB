// -*- C++ -*-

/*!
  \file stochastic/HistogramsPacked.h
  \brief Packed histograms for recording time series populations.
*/

#if !defined(__stochastic_HistogramsPacked_h__)
#define __stochastic_HistogramsPacked_h__

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
class HistogramsPacked
{
private:

  //
  // Member data.
  //

  //! The number of frames.
  std::size_t _numberOfFrames;
  //! The number of species.
  std::size_t _numberOfSpecies;
  //! The sum of the weighted probabilities.
  double _sum;
  //! The histograms.
  std::vector<HistogramReference> _histograms;
  //! The data for the bins.
  std::vector<double> _bins;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor. Invalid data.
  HistogramsPacked() :
    _numberOfFrames(),
    _numberOfSpecies(),
    _sum(),
    _histograms(),
    _bins()
  {
  }

  //! Construct from the number of frames, the number of species, and the number of bins.
  HistogramsPacked(const std::size_t numberOfFrames,
                   const std::size_t numberOfSpecies,
                   const std::size_t numberOfBins) :
    _numberOfFrames(numberOfFrames),
    _numberOfSpecies(numberOfSpecies),
    _sum(0),
    _histograms(numberOfFrames* numberOfSpecies),
    _bins(numberOfFrames* numberOfSpecies* numberOfBins, 0.)
  {
    double* data = &_bins[0];
    for (std::size_t i = 0; i != _histograms.size();
         ++i, data += numberOfBins) {
      _histograms[i].initialize(numberOfBins, data);
    }
  }

  //! Copy constructor.
  HistogramsPacked(const HistogramsPacked& other) :
    _numberOfFrames(other._numberOfFrames),
    _numberOfSpecies(other._numberOfSpecies),
    _sum(other._sum),
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
  HistogramsPacked&
  operator=(const HistogramsPacked& other)
  {
    if (&other != this) {
      _numberOfFrames = other._numberOfFrames;
      _numberOfSpecies = other._numberOfSpecies;
      _sum = other._sum;
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

  //! Return the number of frames.
  std::size_t
  numberOfFrames() const
  {
    return _numberOfFrames;
  }

  //! Return the number of species.
  std::size_t
  numberOfSpecies() const
  {
    return _numberOfSpecies;
  }

  //! Get the sum of the weighted probabilities.
  double
  getSum() const
  {
    return _sum;
  }

  //! Return the specified histogram.
  const HistogramReference&
  operator()(const std::size_t frame, const std::size_t species) const
  {
    return _histograms[frame * _numberOfSpecies + species];
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Initialize from the number of frames, the number of species, and the number of bins.
  void
  initialize(const std::size_t numberOfFrames,
             const std::size_t numberOfSpecies,
             const std::size_t numberOfBins)
  {
    _numberOfFrames = numberOfFrames;
    _numberOfSpecies = numberOfSpecies;
    _sum = 0;
    _histograms.resize(numberOfFrames * numberOfSpecies);
    _bins.resize(numberOfFrames * numberOfSpecies * numberOfBins);
    std::fill(_bins.begin(), _bins.end(), 0);
    double* data = &_bins[0];
    for (std::size_t i = 0; i != _histograms.size();
         ++i, data += numberOfBins) {
      _histograms[i].initialize(numberOfBins, data);
    }
  }

  //! Increment the sum to indicate that another trajectory will be recorded.
  void
  incrementSum()
  {
    ++_sum;
  }

  //! Return the specified histogram.
  HistogramReference&
  operator()(const std::size_t frame, const std::size_t species)
  {
    return _histograms[frame * _numberOfSpecies + species];
  }

  //@}
};

//! Write the histograms in ascii format.
/*!
  \relates HistogramsPacked

  For each frame and each species, print the histogram.
*/
inline
std::ostream&
operator<<(std::ostream& out, const HistogramsPacked& x)
{
  for (std::size_t frame = 0; frame != x.numberOfFrames(); ++frame) {
    for (std::size_t species = 0; species != x.numberOfSpecies(); ++species) {
      out << x(frame, species);
    }
  }
  return out;
}

} // namespace stochastic
}

#endif
