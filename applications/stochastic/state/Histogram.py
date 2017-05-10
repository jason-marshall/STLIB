"""A histogram for a probability mass function."""

import copy
import math
import numpy

def kolmogorovMetric(x, y):
    """For PMF's (probability mass functions) recorded in arrays, the
    Kolmogorov metric is the absolute value of the maximum difference
    between the CMF's (cumulative mass functions). Note that
    PMF's must be valid (sum to unity)."""
    assert len(x) == len(y)
    d = 0.
    f = 0.
    g = 0.
    for i in range(len(x)):
        f += x[i]
        g += y[i]
        d = max(d, abs(f - g))
    return d

def totalVariationMetric(x, y):
    """The total variation metric is half the sum of the absolute value of
    the difference between the PMF's (probability mass functions). Note that
    PMF's must be valid (sum to unity)."""
    assert len(x) == len(y)
    d = 0.
    for i in range(len(x)):
        d += abs(x[i] - y[i])
    return 0.5 * d

class Histogram:
    """A histogram for a probability mass function. It uses a list of bin
    arrays.

    In addition to recording histogram data, we record information so that
    one can compute the mean and variance. Specifically, we record the 
    cardinality, the mean, the summed second centered moment
    sum_i(w_i(x_i-mu)**2), and the sum of the weights.

    The members _width and _inverseWidth are marked as private because they
    must be changed together. Use getWidth() to get the width and setWidth()
    to set the width."""
    
    def __init__(self, size=0, multiplicity=0):
        """Construct an empty histogram."""
        self.cardinality = 0.
        self.sumOfWeights = 0.
        self.mean = 0.
        self.summedSecondCenteredMoment = 0.
        self.lowerBound = 0.
        self.setWidth(1.)
        self.histograms = []
        for i in range(multiplicity):
            self.histograms.append(numpy.zeros(size, numpy.float64))
        # The current histogram.
        self.current = None

    def clear(self):
        self.cardinality = 0.
        self.sumOfWeights = 0.
        self.mean = 0.
        self.summedSecondCenteredMoment = 0.
        self.lowerBound = 0.
        self.setWidth(1.)
        for h in self.histograms:
            h.fill(0)
        self.current = None

    def set(self, i, bins):
        """Set the specified array of bins. This function ensures that the bins
        are represented with a numpy array."""
        self.histograms[i] = numpy.array(bins, numpy.float64)

    def getWidth(self):
        return self._width

    def setWidth(self, width):
        self._width = width
        self._inverseWidth = 1. / width

    def size(self):
        """Return the number of bins."""
        return len(self.histograms[0])

    def multiplicity(self):
        """Return the histogram multiplicity."""
        return len(self.histograms)
    
    def findMinimum(self):
        """Return the index of the histogram with the minimum sum."""
        sums = [sum(h) for h in self.histograms]
        return sums.index(min(sums))

    def setCurrent(self, index):
        """Set the current histogram"""
        self.current = self.histograms[index]

    def setCurrentToMinimum(self):
        """Set the current histogram to the one with the minimum sum."""
        self.current = self.histograms[self.findMinimum()]

    def min(self):
        """Return a closed lower bound."""
        for i in range(len(self.histograms[0])):
            for h in self.histograms:
                if h[i] != 0:
                    return self.lowerBound + i * self._width
        # If the histogram is empty, return infinity.
        return float('inf')

    def max(self):
        """Return an open upper bound."""
        for i in range(len(self.histograms[0])-1, -1, -1):
            for h in self.histograms:
                if h[i] != 0:
                    return self.lowerBound + (i + 1) * self._width
        return 1.

    def _upperBound(self):
        return self.lowerBound + self.size() * self._width

    def __repr__(self):
        """Print the cardinality, sum of weights, mean, summed second centered
        moment, lower bound, width, and the list of bin arrays."""
        return ''.join([repr(self.cardinality), '\n',
                        repr(self.sumOfWeights), '\n',
                        repr(self.mean), '\n',
                        repr(self.summedSecondCenteredMoment), '\n',
                        repr(self.lowerBound), '\n', repr(self._width), '\n',
                        '\n'.join([''.join([repr(x) + ' ' for x in h])
                                   for h in self.histograms])])

    def read(self, stream, multiplicity):
        """Read the cardinality, sum of weights, mean, summed second centered
        moment, lower bound, width, and the list of bin arrays."""
        self.cardinality = float(stream.readline())
        self.sumOfWeights = float(stream.readline())
        self.mean = float(stream.readline())
        self.summedSecondCenteredMoment = float(stream.readline())
        self.lowerBound = float(stream.readline())
        self.setWidth(float(stream.readline()))
        self.histograms = []
        for i in range(multiplicity):
            self.histograms.append(numpy.array([float(x) for x in
                                                stream.readline().split()],
                                               numpy.float64))
        self.current = None

    def accumulate(self, event, weight):
        # Update the statistics.
        if self.cardinality == 0:
            self.cardinality = 1.
            self.sumOfWeights = weight
            self.mean = event
            self.summedSecondCenteredMoment = 0.
        else:
            self.cardinality += 1.
            newSum = self.sumOfWeights + weight
            self.summedSecondCenteredMoment +=\
                self.sumOfWeights * weight * (event - self.mean)**2 / newSum
            self.mean += (event - self.mean) * weight / newSum
            self.sumOfWeights = newSum
        # Update the current histogram.
        self._includeEvent(event)
        index = int((event - self.lowerBound) * self._inverseWidth)
        self.current[index] += weight

    def merge(self, other):
        # Check for the trivial case.
        if other.cardinality == 0:
            return
        # Update the statistics.
        self.cardinality += other.cardinality
        sumOfWeights = self.sumOfWeights + other.sumOfWeights
        self.mean = (self.sumOfWeights * self.mean +
                     other.sumOfWeights * other.mean) / sumOfWeights
        self.summedSecondCenteredMoment += other.summedSecondCenteredMoment
        self.sumOfWeights = sumOfWeights
        # Merge the histograms.
        self._includeHistogram(other)
        for h in other.histograms:
            self.setCurrentToMinimum()
            for i in range(len(h)):
                if h[i] != 0:
                    event = other.lowerBound + i * other._width
                    index = int((event - self.lowerBound) * self._inverseWidth)
                    self.current[index] += h[i]

    def _includeEvent(self, event):
        """If necessary adjust the lower bound and width to include the 
        specified event."""
        # Do nothing if the event will be placed in the current histogram.
        if self.lowerBound <= event and event < self._upperBound():
            return

        # Determine the new closed lower bound.
        if (self.lowerBound < event):
            lower = min(self.min(), event)
        else:
            lower = event
        # Determine the new open upper bound.
        # Add one to get an open upper bound.
        if (event < self._upperBound()):
            upper = max(self.max(), event + 1.)
        else:
            upper = event + 1.
        # Rebuild with the new lower and upper bounds.
        self.rebuild(lower, upper);


    def _includeHistogram(self, other):
        """If necessary adjust the lower bound and width to include the 
        events from the other histogram."""
        # Do nothing if all of the events will be placed in the current
        # histogram.
        if self.lowerBound <= other.lowerBound and\
                self._upperBound() >= other._upperBound():
            return
        # Determine the new closed lower bound.
        lower = min(self.min(), other.min())
        upper = max(self.max(), other.max())
        # Rebuild with the new lower and upper bounds.
        self.rebuild(lower, upper);

    def rebuild(self, low, high):
        assert low >= 0 and low < high

        # Determine the new bounds and a bin width.
        # Note that the width is only allowed to grow.
        newWidth = self._width;
        newLowerBound = math.floor(low / newWidth) * newWidth;
        newUpperBound = newLowerBound + self.size() * newWidth;
        while high > newUpperBound:
            newWidth *= 2
            newLowerBound = math.floor(low / newWidth) * newWidth;
            newUpperBound = newLowerBound + self.size() * newWidth;

        # Rebuild the histogram.
        # Copy the probabilities.
        newInverseWidth = 1. / newWidth;
        newBins = numpy.zeros(self.size(), numpy.float64)
        for bins in self.histograms:
            newBins.fill(0.)
            for i in range(self.size()):
                if bins[i] != 0:
                    event = self.lowerBound + i * self._width;
                    index = int((event - newLowerBound) * newInverseWidth)
                    newBins[index] += bins[i];
            bins[:] = newBins
        # New bounds and width.
        self.lowerBound = newLowerBound;
        self.setWidth(newWidth)

    def getMean(self):
        return self.mean

    def isVarianceDefined(self):
        return self.cardinality > 1

    def getUnbiasedVariance(self):
        if not self.isVarianceDefined():
            return float('inf')
        return self.summedSecondCenteredMoment * self.cardinality /\
            ((self.cardinality - 1) * self.sumOfWeights)

    def getProbabilities(self):
        """Return an array of normalized probabilities."""
        probabilities = numpy.zeros(self.size(), numpy.float64)
        for h in self.histograms:
            probabilities += h
        s = sum(probabilities)
        if s != 0:
            probabilities *= 1. / s
        return probabilities

    def getPmf(self):
        """Return an array of the probability mass function. This is the
        normalized probabilities divided by the bin width."""
        pmf = self.getProbabilities()
        pmf *= self._inverseWidth
        return pmf

    def errorInDistribution(self, metric=totalVariationMetric):
        """Return an estimate of the error in the distribution using the
        specified metric. (The default is the total variation metric.)
        To do this sum the distances between each of
        the histograms and their mean distribution. Then divide by (m - 1)
        where m is the histogram multiplicity to obtain an unbiased estimate
        of the error in each of the histograms. Finally we assume that
        the convergence rate of the metric is 1/sqrt(n) where n is the
        cardinality. Thus we divide by sqrt(m) to obtain an estimate of
        the error in the mean distribution."""
        # If the histogram multiplicity is one, then we cannot estimate the
        # error.
        if len(self.histograms) == 1:
            return 1.
        d = self.getProbabilities()
        # Handle the special case that no events have been recorded.
        if sum(d) == 0.:
            return 1.
        x = numpy.zeros(self.size(), numpy.float64)
        s = 0.
        # The number of non-empty histograms.
        multiplicity = 0
        for i in range(len(self.histograms)):
            x[:] = self.histograms[i]
            # Normalize to get probabilities.
            sx = sum(x)
            if sx != 0:
                multiplicity += 1
                x *= 1./sx
                s += metric(d, x)
        # If the multiplicity is unity, we cannot estimate the error.
        if multiplicity <= 1:
            return 1.
        return s / ((multiplicity - 1) * math.sqrt(multiplicity))

    def writeXml(self, writer, frame=None, species=None):
        """frame is the frame index. species is the recorded species index,
        which is not the same as the species index as not all species may
        be recorded."""
        attributes =\
            {'cardinality':repr(self.cardinality),
             'sumOfWeights':repr(self.sumOfWeights),
             'mean':repr(self.mean),
             'summedSecondCenteredMoment':repr(self.summedSecondCenteredMoment),
             'lowerBound':repr(self.lowerBound),
             'width':repr(self._width)}
        if frame is not None:
            attributes['frame'] = repr(frame)
        if species is not None:
            attributes['species'] = repr(species)
        writer.beginElement('histogram', attributes)
        for h in self.histograms:
            writer.writeElement('histogramElement', {},
                                ' '.join([repr(x) for x in h]))
        writer.endElement() # histogram
        
def coordinate(histograms):
    """Coordinate the list of histograms so they have the same ranges and bin
    widths."""
    low = min([x.min() for x in histograms])
    # Check the case that all histograms are empty.
    if low == float('inf'):
        low = 0
    high = max([x.max() for x in histograms])
    for x in histograms:
        x.rebuild(low, high)
    for x in histograms[1:]:
        assert histograms[0].lowerBound == x.lowerBound and\
            histograms[0]._width == x._width

def histogramDistance(a, b, metric=totalVariationMetric):
    """Return the histogram distance using the specified metric (total
    variation is the default)."""
    # Make copies and synchronize them.
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    coordinate([a, b])
    assert a.lowerBound == b.lowerBound and a._width == b._width and \
        a.size() == b.size()
    # Return the distance.
    return metric(a.getProbabilities(), b.getProbabilities())

def _computeMeanAndVariance(values, weights):
    assert len(values) == len(weights)
    n = len(values)
    assert n > 1
    mean = 0.
    sumOfWeights = 0.
    for i in range(n):
        mean += weights[i] * values[i]
        sumOfWeights += weights[i]
    mean /= sumOfWeights
    variance = 0.
    for i in range(n):
        variance += weights[i] * (values[i] - mean)**2
    variance *= n / ((n - 1) * sumOfWeights)
    return (mean, variance)
