"""Tests the HistogramFrames class."""

import sys
if __name__ == '__main__':
    sys.path.insert(1, '..')
    sys.path.insert(1, '../..')
else:
    sys.path.insert(1, 'state')

from StringIO import StringIO

from unittest import TestCase, main

from HistogramFrames import HistogramFrames
from Histogram import Histogram
from io.XmlWriter import XmlWriter

class HistogramFramesTest(TestCase):
    def test(self):
        numberOfBins = 4
        multiplicity = 2
        h = Histogram(numberOfBins, multiplicity)
        h.setCurrentToMinimum()
        h.accumulate(0, 1)
        h.accumulate(1, 2)
        h.accumulate(2, 2)
        h.accumulate(3, 1)

        frameTimes = [0, 1]
        recordedSpecies = [0, 1, 2]
        x = HistogramFrames(numberOfBins, multiplicity, recordedSpecies)
        x.setFrameTimes(frameTimes)
        for i in range(2):
            x.setCurrentToMinimum()
            for i in range(len(frameTimes)):
                for j in range(len(recordedSpecies)):
                    x.histograms[i][j].merge(h)
        assert not x.hasErrors()

        stream = StringIO()
        writer = XmlWriter(stream)
        writer.beginDocument()
        x.writeXml(writer, 'model', 'method')
        writer.endDocument()

if __name__ == '__main__':
    main()
