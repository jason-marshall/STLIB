"""Tests the HistogramAverage class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from StringIO import StringIO

from unittest import TestCase, main

from HistogramAverage import HistogramAverage
from Histogram import Histogram
from io.XmlWriter import XmlWriter

class HistogramAverageTest(TestCase):
    def test(self):
        numberOfBins = 4
        multiplicity = 2
        h = Histogram(numberOfBins, multiplicity)
        h.setCurrentToMinimum()
        h.accumulate(0, 1)
        h.accumulate(1, 2)
        h.accumulate(2, 2)
        h.accumulate(3, 1)

        recordedSpecies = [0, 1, 2]
        x = HistogramAverage(numberOfBins, multiplicity, recordedSpecies)
        for i in range(2):
            x.setCurrentToMinimum()
            for i in range(len(recordedSpecies)):
                x.histograms[i].merge(h)
        assert not x.hasErrors()

        stream = StringIO()
        writer = XmlWriter(stream)
        writer.beginDocument()
        x.writeXml(writer, 'model', 'method')
        writer.endDocument()

if __name__ == '__main__':
    main()
