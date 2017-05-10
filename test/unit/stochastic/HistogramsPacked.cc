// -*- C++ -*-

#include "stlib/stochastic/HistogramsPacked.h"

using namespace stlib;

int
main()
{
  typedef stochastic::HistogramsPacked HistogramsPacked;
  typedef stochastic::HistogramReference HistogramReference;

  const std::size_t NumberOfFrames = 3;
  const std::size_t NumberOfSpecies = 2;

  {
    // Default constructor.
    HistogramsPacked histograms;
    assert(histograms.numberOfFrames() == 0);
    assert(histograms.numberOfSpecies() == 0);
    assert(histograms.getSum() == 0);
    // Initialize.
    const std::size_t Size = 32;
    histograms.initialize(NumberOfFrames, NumberOfSpecies, Size);
    for (std::size_t frame = 0; frame != NumberOfFrames; ++frame) {
      for (std::size_t species = 0; species != NumberOfSpecies; ++species) {
        const HistogramReference& x = histograms(frame, species);
        assert(x.size() == Size);
        assert(x.computeSum() == 0);
        assert(x.getLowerBound() == 0);
        assert(x.getWidth() == 1);
        assert(x.getUpperBound() == Size);
        for (std::size_t i = 0; i != x.size(); ++i) {
          assert(x[i] == 0);
        }
      }
    }
    assert(histograms.getSum() == 0);
    histograms.incrementSum();
    assert(histograms.getSum() == 1);
  }

  {
    // Empty.
    const std::size_t Size = 32;
    HistogramsPacked histograms(NumberOfFrames, NumberOfSpecies, Size);
    for (std::size_t frame = 0; frame != NumberOfFrames; ++frame) {
      for (std::size_t species = 0; species != NumberOfSpecies; ++species) {
        const HistogramReference& x = histograms(frame, species);
        assert(x.size() == Size);
        assert(x.computeSum() == 0);
        assert(x.getLowerBound() == 0);
        assert(x.getWidth() == 1);
        assert(x.getUpperBound() == Size);
        for (std::size_t i = 0; i != x.size(); ++i) {
          assert(x[i] == 0);
        }
      }
    }
    assert(histograms.getSum() == 0);
    histograms.incrementSum();
    assert(histograms.getSum() == 1);
  }

  {
    // Insert within range.
    const std::size_t Size = 32;
    HistogramsPacked histograms(NumberOfFrames, NumberOfSpecies, Size);
    for (std::size_t frame = 0; frame != NumberOfFrames; ++frame) {
      for (std::size_t species = 0; species != NumberOfSpecies; ++species) {
        HistogramReference& x = histograms(frame, species);
        for (std::size_t i = 0; i != Size; ++i) {
          x.accumulate(i, 1);
        }
        assert(x.size() == Size);
        assert(x.computeSum() == Size);
        assert(x.getLowerBound() == 0);
        assert(x.getWidth() == 1);
        assert(x.getUpperBound() == Size);
        for (std::size_t i = 0; i != x.size(); ++i) {
          assert(x[i] == 1);
        }
      }
    }
    // Copy constructor.
    HistogramsPacked copy(histograms);
    for (std::size_t frame = 0; frame != NumberOfFrames; ++frame) {
      for (std::size_t species = 0; species != NumberOfSpecies; ++species) {
        HistogramReference& x = histograms(frame, species);
        HistogramReference& y = copy(frame, species);
        assert(x.size() == y.size());
        assert(x.computeSum() == y.computeSum());
        assert(x.getLowerBound() == y.getLowerBound());
        assert(x.getWidth() == y.getWidth());
        assert(x.getUpperBound() == y.getUpperBound());
        for (std::size_t i = 0; i != x.size(); ++i) {
          assert(x[i] == y[i]);
        }
      }
    }
  }

  {
    // Insert above range.
    const std::size_t Size = 32;
    HistogramsPacked histograms(NumberOfFrames, NumberOfSpecies, Size);
    for (std::size_t frame = 0; frame != NumberOfFrames; ++frame) {
      for (std::size_t species = 0; species != NumberOfSpecies; ++species) {
        HistogramReference& x = histograms(frame, species);
        for (std::size_t i = 0; i != 2 * Size; ++i) {
          x.accumulate(i, 1);
        }
        assert(x.size() == Size);
        assert(x.computeSum() == 2 * Size);
        assert(x.getLowerBound() == 0);
        assert(x.getWidth() == 2);
        assert(x.getUpperBound() == 2 * Size);
        for (std::size_t i = 0; i != x.size(); ++i) {
          assert(x[i] == 2);
        }
      }
    }
  }

  {
    // Insert below range.
    const std::size_t Size = 32;
    HistogramsPacked histograms(NumberOfFrames, NumberOfSpecies, Size);
    for (std::size_t frame = 0; frame != NumberOfFrames; ++frame) {
      for (std::size_t species = 0; species != NumberOfSpecies; ++species) {
        HistogramReference& x = histograms(frame, species);
        for (std::size_t i = 2 * Size; i != 0; --i) {
          x.accumulate(i - 1, 1);
        }
        assert(x.size() == Size);
        assert(x.computeSum() == 2 * Size);
        assert(x.getLowerBound() == 0);
        assert(x.getWidth() == 2);
        assert(x.getUpperBound() == 2 * Size);
        for (std::size_t i = 0; i != x.size(); ++i) {
          assert(x[i] == 2);
        }

        x.reset();
        assert(x.size() == Size);
        assert(x.computeSum() == 0);
        assert(x.getLowerBound() == 0);
        assert(x.getWidth() == 1);
        assert(x.getUpperBound() == Size);
        for (std::size_t i = 0; i != x.size(); ++i) {
          assert(x[i] == 0);
        }
      }
    }
  }

  return 0;
}
