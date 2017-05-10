"""Tests the Histogram class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from StringIO import StringIO
import math
import numpy

from unittest import TestCase, main

from Histogram import Histogram, _computeMeanAndVariance, histogramDistance
from io.XmlWriter import XmlWriter

class HistogramTest(TestCase):
    def testEmptyBins(self):
        x = Histogram(10, 2)
        self.assertEqual(x.size(), 10)
        self.assertEqual(x.min(), float('inf'))
        self.assertEqual(x.max(), 1.)
        self.assertEqual(x._upperBound(), 10.)
        self.assertEqual(len(x.histograms), 2)
        p = x.getProbabilities()
        self.assertTrue((p == numpy.zeros(10, numpy.float64)).all())

    def testConstantValueBins(self):
        # Basic tests with constant-value bins.
        size = 8
        values = range(size)
        weights = [1.] * size
        mean, variance = _computeMeanAndVariance(values, weights)
        x = Histogram(size, 2)
        x.setCurrentToMinimum()
        for i in range(8):
            x.accumulate(i, 1.)
        self.assertAlmostEqual(x.getMean(), mean)
        self.assertAlmostEqual(x.getUnbiasedVariance(), variance)
        self.assertEqual(x.size(), 8)
        self.assertEqual(x.min(), 0.)
        self.assertEqual(x.max(), 8.)
        self.assertEqual(x._upperBound(), 8.)
        p = x.getProbabilities()
        self.assertTrue((p == 1./8.).all())

    def testMoreBasic(self):
        # More basic tests.
        x = Histogram(8, 2)
        x.setCurrentToMinimum()
        x.accumulate(4, 1.)
        assert x.getMean() == 4
        assert x.getUnbiasedVariance() == float('inf')
        assert x.size() == 8
        assert x.min() == 4.
        assert x.max() == 5.
        assert x._upperBound() == 8.

    def testAccumulate1(self):
        x = Histogram(8, 2)
        x.setCurrentToMinimum()
        x.accumulate(0, 1)
        assert x.size() == 8
        assert x.min() == 0.
        assert x.max() == 1.
        assert x._upperBound() == 8.
        x.accumulate(7, 1)
        assert x.size() == 8
        assert x.min() == 0.
        assert x.max() == 8.
        assert x._upperBound() == 8.
        x.accumulate(8, 1)
        assert x.size() == 8
        assert x.min() == 0.
        assert x.max() == 10.
        assert x._upperBound() == 16.
        x.accumulate(15, 1)
        assert x.size() == 8
        assert x.min() == 0.
        assert x.max() == 16.
        assert x._upperBound() == 16.

    def testAccumulate2(self):
        x = Histogram(8, 2)
        x.setCurrentToMinimum()
        x.accumulate(10, 1)
        assert x.size() == 8
        assert x.min() == 10.
        assert x.max() == 11.
        assert x._upperBound() == 18.
        x.accumulate(5, 1)
        assert x.size() == 8
        assert x.min() == 5.
        assert x.max() == 11.
        assert x._upperBound() == 13.
        x.accumulate(0, 1)
        assert x.size() == 8
        assert x.min() == 0.
        assert x.max() == 12.
        assert x._upperBound() == 16.

    def testPoisson1(self):
        # Poisson with mean 4. 20 bins. PMF = e^-lambda lambda^n / n!
        # Store in the first array of bins.
        lam = 4.
        size = 20
        poisson = [math.exp(-lam)]
        for n in range(1, size):
            poisson.append(poisson[-1] * lam / n)
        cardinality = size
        sumOfWeights = sum(poisson)
        mean = 0.
        for i in range(size):
            mean += poisson[i] * i
        mean /= sumOfWeights
        summedSecondCenteredMoment = 0.
        for i in range(size):
            summedSecondCenteredMoment += poisson[i] * (i - mean)**2

        stream = StringIO(repr(cardinality) + '\n' + 
                          repr(sumOfWeights) + '\n' +
                          repr(mean) + '\n' + 
                          repr(summedSecondCenteredMoment) + '\n' + 
                          '0\n1\n' +
                          ''.join([repr(_x) + ' ' for _x in poisson]) + '\n' +
                          '0 ' * len(poisson) + '\n')

        x = Histogram()
        x.read(stream, 2)
        assert x.cardinality == cardinality
        assert x.sumOfWeights == sumOfWeights
        assert x.mean == mean
        assert x.summedSecondCenteredMoment == summedSecondCenteredMoment
        assert x.size() == len(poisson)
        self.assertAlmostEqual(sum(x.getProbabilities()), 1)

        stream = StringIO()
        writer = XmlWriter(stream)
        writer.beginDocument()
        x.writeXml(writer, 0, 0)
        writer.endDocument()

    def testPoisson2(self):
        # Poisson with mean 4. Use accumulate.
        lam = 4.
        size = 20
        x = Histogram(size, 2)
        x.setCurrentToMinimum()
        weight = math.exp(-lam)
        for i in range(50):
            x.accumulate(i, weight)
            weight *= lam / (i + 1)
        assert abs(x.getMean() - 4) < 1e-4
        assert abs(x.getUnbiasedVariance() - 4) < 1e-1

        # Uniform, 20 bins.
        size = 20
        weights = [1.] * size
        cardinality = size
        sumOfWeights = sum(weights)
        mean = 0.
        for i in range(size):
            mean += weights[i] * i
        mean /= sumOfWeights
        summedSecondCenteredMoment = 0.
        for i in range(size):
            summedSecondCenteredMoment += weights[i] * (i - mean)**2

        stream = StringIO(repr(cardinality) + '\n' + 
                          repr(sumOfWeights) + '\n' +
                          repr(mean) + '\n' + 
                          repr(summedSecondCenteredMoment) + '\n' + 
                          '0\n1\n' +
                          ''.join([str(_x) + ' ' for _x in [1.] * size]) +
                          '\n' +
                          '0 ' * size + '\n')
        y = Histogram()
        y.read(stream, 2)
        # Distance between Poisson and uniform.
        assert histogramDistance(x, y) > 0
        # Distance between Poisson and Poisson.
        assert histogramDistance(x, x) == 0

        # Uniform, 30 bins.
        x.merge(y)

if __name__ == '__main__':
    main()
